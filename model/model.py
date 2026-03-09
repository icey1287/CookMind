"""
MiniMind 模型主体。

这个文件实现的是一个简化版 decoder-only Transformer，核心组件包括：
1. 配置类 MiniMindConfig
2. RMSNorm
3. RoPE 位置编码
4. GQA 注意力
5. 前馈网络 FFN
6. 多层 Transformer Block 堆叠
7. Causal LM 输出头与 loss

阅读建议：
1. 先看 MiniMindConfig，理解模型有哪些超参数。
2. 再看 Attention 和 FeedForward，理解单层 block 的组成。
3. 最后看 MiniMindModel 和 MiniMindForCausalLM，理解整个前向流程。
"""

from typing import Tuple,Optional,List,Union
from transformers import  PretrainedConfig,PreTrainedModel,GenerationMixin
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn.init import init

class MiniMindConfig(PretrainedConfig):
    """
    模型配置类。

    这个类本身不做计算，它只是把“模型长什么样”这件事描述清楚。
    后面的每一个模块都会从 config 里读取所需超参数。
    """

    model_type = "minimind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        
        ############ MoE ############
        use_moe:bool=False,
        num_experts_per_tok:int=2,
        n_routed_experts:int=4,
        n_shared_experts:int=1,
        scoring_func:str='softmax',
        aux_loss_alpha:float=0.1,
        seq_aux:bool=True,
        norm_topk_prob:bool=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # 下面这些字段会在模型构建阶段被各个模块读取。
        # 它们共同定义了词表大小、隐藏层维度、层数、注意力头数等核心结构。
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe=use_moe
        self.num_experts_per_tok=num_experts_per_tok
        self.n_routed_experts=n_routed_experts
        self.n_shared_experts=n_shared_experts
        self.seq_aux=seq_aux
        self.norm_topk_prob=norm_topk_prob
        self.aux_loss_alpha=aux_loss_alpha
        self.scoring_func=scoring_func

        # inference_rope_scaling=True 时，构造 YaRN 风格的 RoPE 缩放参数。
        # 这通常用于推理阶段想把上下文扩得更长的场景。
        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


class RMSNorm(nn.Module):
    """
    RMSNorm 归一化。

    和 LayerNorm 相比，RMSNorm 不做均值中心化，只按均方根进行缩放。
    这样计算更简单，在很多大模型里也很常见。
    """

    def __init__(self, dim:int,eps:float=1e-5):
        super().__init__()
        self.dim=dim
        self.eps=eps

        # 每个隐藏维度都有一个可学习缩放系数。
        self.weight=nn.Parameter(torch.ones(dim))

    def _norm(self,x):
        # x.pow(2).mean(-1, keepdim=True) 计算最后一维的均方值。
        # rsqrt(...) 等价于 1 / sqrt(...)。
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)

    def forward(self,x):
        # 先在 float 精度下算归一化，再 cast 回原始 dtype，兼顾稳定性和性能。
        return self.weight * self._norm(x.float()).type_as(x)

def precompute_freqs_cis(dim:int,end:int=int(32*1024),rope_base:float=1e6,rope_scaling:Optional[dict]=None):
    """
    预先计算 RoPE 所需的 cos/sin 表。

    为什么要预计算：
    - 训练时每一层、每个 batch 都会用到位置编码。
    - 如果每次前向都重新算一遍，会重复开销。
    - 预先算好并注册成 buffer，后面直接切片取用更高效。
    """

    # 先生成每一对维度对应的基础频率。
    freqs,attn_factor=1.0/(rope_base**(torch.arange(0,dim,2)[:dim//2].float()/dim)),1.0
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    # t 表示位置下标 0,1,2,...,end-1。
    t = torch.arange(end, device=freqs.device)

    # outer(t, freqs) 得到“位置 × 频率”的网格。
    freqs = torch.outer(t, freqs).float()

    # RoPE 会把 cos/sin 在最后一个维度复制一份，便于和 q/k 的偶数/奇数位配对。
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q,k,cos,sin,position_ids=None,unsqueeze_dim=1):
    """
    把 RoPE 应用到 query 和 key 上。

    核心思想：
    把最后一维看成一个个二维向量，然后在二维平面里做旋转。
    这样不同位置会得到不同相位，从而把位置信息编码进注意力计算里。
    """

    def rotate_half(x:torch.Tensor):
        # 把最后一维一分为二，形如 [x1, x2]，旋转成 [-x2, x1]。
        x1=x[..., :x.shape[-1]//2]
        x2=x[..., x.shape[-1]//2:]
        return torch.stack((-x2,x1),dim=-1).flatten(-2)

    # 这里就是标准 RoPE 公式：x*cos(theta) + rotate_half(x)*sin(theta)
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

def repeat_kv(x:torch.Tensor,n_rep:int)->torch.Tensor:
        """
        GQA/MQA 中重复 key/value heads。

        当 query 头数大于 key/value 头数时，
        需要把较少的 kv heads 复制多份，才能和 query 头数对齐。
        """

        bs,slen,n_key_value_heads,head_dim=x.shape
        if n_rep==1:
            return x
        
        return x[:, :, :, None, :].expand(bs,slen,n_key_value_heads,n_rep,head_dim).reshape(bs,slen,n_key_value_heads*n_rep,head_dim)

class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss
    
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0: y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else: y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache
    


class Attention(nn.Module):     #GQA
    """
    注意力模块。

    这里实现的是 GQA: Grouped Query Attention。
    它的思路是：
    - 保持较多的 query heads，保证表达能力。
    - 减少 key/value heads，降低显存和计算开销。
    """

    def __init__(self,args:MiniMindConfig):
        super().__init__()

        # 如果没有单独指定 num_key_value_heads，就退化成普通多头注意力。
        self.num_key_value_heads=args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads

        assert args.num_attention_heads%self.num_key_value_heads==0, "num_attention_heads must be divisible by num_key_value_heads"

        # n_rep 表示每个 kv head 需要复制给多少个 query head 使用。
        self.n_local_heads=args.num_attention_heads
        self.n_local_kv_heads=self.num_key_value_heads
        self.n_rep=self.n_local_heads//self.n_local_kv_heads
        self.head_dim=args.hidden_size//args.num_attention_heads

        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)     # Query投影
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)     # Key投影
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)     # Value投影
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)     # 输出投影
        
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # 如果当前 PyTorch 版本支持 scaled_dot_product_attention，且配置允许，
        # 就优先走 flash/SDPA 快路径。
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attention

    def forward(self,x:torch.Tensor,
                position_embeddings:Tuple[torch.Tensor,torch.Tensor],
                past_key_value:Optional[Tuple[torch.Tensor,torch.Tensor]]=None,
                use_cache=False,
                attention_mask:Optional[torch.Tensor]=None)->torch.Tensor:
        """
        输入 x 的形状: [batch_size, seq_len, hidden_size]

        输出：
        - output: 当前层注意力结果，形状仍然是 [batch_size, seq_len, hidden_size]
        - past_kv: 如果 use_cache=True，则返回缓存好的 k/v，供增量推理复用
        """

        bsz,seq_len,_=x.shape

        # 先把隐藏状态投影成 q/k/v。
        xq,xk,xv=self.q_proj(x),self.k_proj(x),self.v_proj(x)

        # 把最后一维 reshape 成“头数 × 每头维度”。
        xq=xq.view(bsz,seq_len,self.n_local_heads,self.head_dim)
        xk=xk.view(bsz,seq_len,self.n_local_kv_heads,self.head_dim)
        xv=xv.view(bsz,seq_len,self.n_local_kv_heads,self.head_dim)

        cos,sin=position_embeddings

        # 给 q/k 注入旋转位置编码。
        xq,xk=apply_rotary_pos_emb(xq,xk,cos,sin)
        if past_key_value is not None:
            # 增量推理时，把历史缓存的 k/v 和当前步新产生的 k/v 拼起来。
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 调整维度顺序，方便做注意力矩阵乘法。
        xq=xq.transpose(1,2)  # bsz, n_local_heads, seq_len, head_dim
        xk=repeat_kv(xk,self.n_rep).transpose(1,2)  # bsz, n_local_heads, seq_len, head_dim
        xv=repeat_kv(xv,self.n_rep).transpose(1,2)  # bsz, n_local_heads, seq_len, head_dim

        if self.flash and seq_len>1 and (attention_mask is None or torch.all(attention_mask == 1)):
            # 快路径：交给 PyTorch 原生高性能实现。
            output=F.scaled_dot_product_attention(xq,xk,xv,dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # 慢路径：手动实现标准 causal self-attention。
            scores=(xq@xk.transpose(-2,-1))/math.sqrt(self.head_dim)

            # 因果 mask：当前位置不能看未来 token。
            scores[:,:,:,-seq_len:]+=torch.triu(torch.full((seq_len,seq_len),float('-inf'),device=scores.device),diagonal=1)

            if attention_mask is not None:
                # 把 [bsz, seq_len] 的 mask 扩展到可和 scores 广播相加的形状。
                extended_attention_mask=attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask=(1.0-extended_attention_mask)*-1e9
                scores=scores+extended_attention_mask
            
            # softmax 后得到注意力权重，再乘 v 得到上下文表示。
            scores=F.softmax(scores.float(),dim=-1).type_as(xq)
            scores=self.attn_dropout(scores)
            output=scores@xv
        
        # 把多头结果拼回 hidden_size，并做输出投影。
        output=output.transpose(1,2).reshape(bsz,seq_len,-1) # bsz, seq_len, n_local_heads*head_dim
        output=self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    """
    Transformer block 里的前馈网络。

    这里采用的是常见的 gated FFN 形式：
    act(gate_proj(x)) * up_proj(x)，然后再经过 down_proj 投回 hidden_size。
    """

    def __init__(self,config:MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            # 常见经验做法：中间层维度约为 hidden_size * 8 / 3，
            # 再向上对齐到 64 的倍数，方便底层算子实现。
            intermediate_size=int(config.hidden_size*8/3)
            config.intermediate_size=64*((intermediate_size+63)//64)
        self.gate_proj=nn.Linear(config.hidden_size,config.intermediate_size,bias=False)
        self.down_proj=nn.Linear(config.intermediate_size,config.hidden_size,bias=False)
        self.up_proj=nn.Linear(config.hidden_size,config.intermediate_size,bias=False)
        self.dropout=nn.Dropout(config.dropout)

        # ACT2FN 是 transformers 里维护的“激活函数名字 -> 真正函数”的映射表。
        self.act_fn=ACT2FN[config.hidden_act]
        

    def forward(self,x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

class MiniMindBlock(nn.Module):
    """
    一个完整的 Transformer block。

    结构顺序是：
    1. RMSNorm
    2. Self-Attention
    3. 残差连接
    4. RMSNorm
    5. FFN
    6. 残差连接
    """

    def __init__(self,layer_id:int,config:MiniMindConfig):
        super().__init__()
        self.num_attention_heads=config.num_attention_heads
        self.hidden_size=config.hidden_size
        self.head_dim=config.hidden_size//config.num_attention_heads
        self.self_attn=Attention(config)

        self.layer_id=layer_id
        self.input_layernorm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.post_attention_layernorm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        self.mlp=FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self,hidden_states,postion_embeddings,past_key_value=None,use_cache=False,attention_mask=None):
        # 先保存残差分支输入。
        residual=hidden_states
        hidden_states,present_key_value=self.self_attn(
            self.input_layernorm(hidden_states),postion_embeddings,
            past_key_value,use_cache,attention_mask
        )

        # attention 输出加回残差。
        hidden_states+=residual

        # FFN 部分同样采用 Pre-Norm + 残差结构。
        hidden_states=hidden_states+self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states,present_key_value

class MiniMindModel(nn.Module):
    """
    只包含 Transformer 主干，不包含语言模型输出头。

    它的输出是最后一层隐藏状态 hidden_states。
    如果要真正做文本生成或算语言模型 loss，需要再接一个 lm_head。
    """

    def __init__(self,config:MiniMindConfig):
        super().__init__()
        self.config=config
        self.vocab_size=config.vocab_size
        self.num_hidden_layers=config.num_hidden_layers

        # token embedding: 把离散 token id 映射成连续向量。
        self.embed_tokens=nn.Embedding(config.vocab_size,config.hidden_size)
        self.dropout=nn.Dropout(config.dropout)

        # 堆叠多个 Transformer block。
        self.layers=nn.ModuleList([MiniMindBlock(i,config) for i in range(config.num_hidden_layers)])
        self.norm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)

        # 预先计算整段上下文范围内 RoPE 所需的 cos/sin 表。
        freqs_cos,freqs_sin=precompute_freqs_cis(
            dim=config.hidden_size//config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )

        # register_buffer 表示：
        # - 它们不是可训练参数
        # - 但会跟着模型一起移动到 GPU / 保存到 state_dict
        self.register_buffer("freqs_cos",freqs_cos,False)
        self.register_buffer("freqs_sin",freqs_sin,False)
    
    def forward(self,
                input_ids:Optional[torch.Tensor]=None,
                attention_mask:Optional[torch.Tensor]=None,
                past_key_values:Optional[List[Tuple[torch.Tensor,torch.Tensor]]]=None,
                use_cache:bool=False,
                **kwargs,                
                ):
        """
        主干网络前向传播。

        输入：
        - input_ids: [batch_size, seq_len]
        - attention_mask: 可选，标记哪些位置有效
        - past_key_values: 推理缓存

        输出：
        - hidden_states: 最后一层隐藏状态
        - presents: 每一层的缓存 k/v
        - aux_loss: MoE 门控的辅助负载均衡损失；Dense 模型时为 0
        """

        batch_size,seq_len=input_ids.shape

        if hasattr(past_key_values,'layers'):
            past_key_values=None

        # 训练时 past_key_values 通常是 None，这里统一转成一个长度等于层数的列表。
        past_key_values=past_key_values or [None]*len(self.layers)

        # 如果存在缓存，说明当前是在增量推理。
        # start_pos 表示当前 token 在整条序列中的起始位置。
        start_pos=past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # token id -> embedding vector。
        hidden_states=self.dropout(self.embed_tokens(input_ids))

        # 根据当前起始位置和序列长度，从预计算表里切出这一段位置编码。
        position_embeddings=(self.freqs_cos[start_pos:start_pos+seq_len],self.freqs_sin[start_pos:start_pos+seq_len])
        
        presents=[]
        for layer_idx,(layer,past_key_values) in enumerate(zip(self.layers,past_key_values)):
            # 每层都吃入同一份 attention_mask，但 past_key_value 是按层分别维护的。
            hidden_states,present=layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_values,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        
        # 末尾再做一次归一化，得到最终隐藏状态。
        hidden_states=self.norm(hidden_states)

        # 如果启用了 MoE，这里把每一层专家门控产生的辅助损失加总起来；
        # Dense 模型时列表为空，sum 的初值会保证返回一个 0 张量而不是 None。
        aux_loss = sum(
            [layer.mlp.aux_loss for layer in self.layers if isinstance(layer.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze()
        )
        return hidden_states, presents, aux_loss

class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    完整的因果语言模型。

    它在 MiniMindModel 主干之上再加一个 lm_head，
    用于把 hidden_states 映射回词表大小，得到每个位置的 logits。
    """

    config_class = MiniMindConfig

    def __init__(self,config:MiniMindConfig):
        self.config=config or MiniMindConfig()
        super().__init__(self.config)
        self.model=MiniMindModel(self.config)

        # lm_head 把 hidden_size 投影到 vocab_size，得到对每个 token 的预测分数。
        self.lm_head=nn.Linear(self.config.hidden_size,self.config.vocab_size,bias=False)
        #输出层和嵌入层权重共享
        self.model.embed_tokens.weight=self.lm_head.weight

    def forward(self,
                input_ids:Optional[torch.Tensor]=None,
                attention_mask:Optional[torch.Tensor]=None,
                labels:Optional[torch.Tensor]=None,
                past_key_values:Optional[List[Tuple[torch.Tensor,torch.Tensor]]]=None,
                use_cache:bool=False,
                logits_to_keep:Union[int,torch.Tensor]=0,
                **args):
        """
        因果语言模型前向。

        如果 labels 不为 None，就会额外计算训练时使用的交叉熵损失。
        """
        
        hidden_states,past_key_values,aux_loss=self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )

        # logits_to_keep 允许只保留最后若干个位置的 logits。
        # 训练时通常保留全部；生成时可能只关心最后一个位置。
        slice_indices=slice(-logits_to_keep,None) if isinstance(logits_to_keep,int)  else logits_to_keep 
        logits=self.lm_head(hidden_states)[:,slice_indices,:]

        loss=None
        if labels is not None:
            # 因果语言模型的训练目标是：
            # 用位置 t 的输出去预测位置 t+1 的真实 token。
            # 所以 logits 去掉最后一个时间步，labels 去掉第一个时间步。
            shifted_logits=logits[...,:-1,:].contiguous()
            shifted_labels=labels[...,1:].contiguous()

            # ignore_index=-100 对应数据集里 pad 位置的标签设置。
            loss=F.cross_entropy(shifted_logits.view(-1,shifted_logits.size(-1)),shifted_labels.view(-1),ignore_index=-100)

        # 返回 Hugging Face 标准输出对象，便于兼容现有训练/推理接口。
        output = CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        output.aux_loss = aux_loss
        return output

