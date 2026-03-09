"""
GRPO（Group Relative Policy Optimization）训练脚本。

GRPO 是一种面向推理模型训练的强化学习方法，可以看成 PPO 的轻量变种。
它最重要的特点是：
1. 不需要单独训练 value model（评论家 / critic）。
2. 对同一个 prompt 一次生成多条回答，在组内做相对比较。
3. 用 reward model 或规则函数给这组回答打分。
4. 根据组内相对优势（advantage）更新策略模型。

直觉上，它像这样工作：
- 同一个问题，模型一次写出 8 份答案。
- 奖励系统给这 8 份答案评分。
- 不去问“绝对上该得多少分”，而是只看“这 8 份里谁更好”。
- 比组内平均更好的答案被强化，比平均更差的答案被抑制。

这种做法尤其适合长链推理（reasoning）场景，因为它能在没有人工 chosen/rejected
标注的情况下，通过模型自采样 + 奖励打分不断提升推理质量。
"""

import os
import sys

# 设置包名和搜索路径，便于从项目根目录导入内部模块。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import warnings
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model import CookMindConfig
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model

# 屏蔽无关警告，保持训练日志更清晰。
warnings.filterwarnings('ignore')


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """整合所有奖励函数计算总奖励"""
    def reasoning_model_reward(rewards):
        # 对 reasoning 模型额外加入“格式奖励”：
        # 回答是否遵循 <think>...</think><answer>...</answer> 模板。
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            # 只要格式完全匹配，就给一个基础奖励 0.5。
            if match_pattern or match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        def mark_num(text):
            # 更细粒度的标签计数奖励：
            # 每个关键标签出现且只出现一次，各加 0.25 分。
            # 这样即便整体格式不完全匹配，也能鼓励模型逐步学会模板。
            reward = 0
            if text.count("<think>") == 1:
                reward += 0.25
            if text.count("</think>") == 1:
                reward += 0.25
            if text.count("<answer>") == 1:
                reward += 0.25
            if text.count("</answer>") == 1:
                reward += 0.25
            return reward

        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    # 每条 response 对应一个最终奖励，初始全为 0。
    rewards = torch.zeros(len(responses), device=args.device)
    if args.reasoning == 1:
        # 推理模型额外叠加格式奖励。
        rewards = reasoning_model_reward(rewards)

    with torch.no_grad():
        reward_model_scores = []
        # prompts 的长度是 batch_size；responses 的长度是 batch_size * num_generations。
        batch_size = len(prompts)
        # 对 reward model 的打分进行裁剪，防止极端分值破坏训练稳定性。
        scale = 3.0

        for i in range(batch_size):
            for j in range(args.num_generations):
                # 第 i 个 prompt 对应的第 j 条采样回答在 responses 中的下标。
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                # 把 chat template 重新解析回 role/content 结构，供 reward model 评分。
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]

                # 将本次 response 作为 assistant 回答拼回对话上下文中。
                tmp_chat = messages + [{"role": "assistant", "content": response}]
                score = reward_model.get_score(reward_tokenizer, tmp_chat)
                score = max(min(score, scale), -scale)

                if args.reasoning == 1:
                    # 对 reasoning 模型，如果回答里有 <answer>...</answer>，
                    # 再单独取 answer 部分打一次分，减少 think 过程中的噪声干扰。
                    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                    if answer_match:
                        answer_content = answer_match.group(1).strip()
                        tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                        answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                        answer_score = max(min(answer_score, scale), -scale)
                        # 把“完整回复得分”和“最终答案得分”加权融合。
                        score = score * 0.4 + answer_score * 0.6

                reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        # 总奖励 = 格式奖励 + reward model 语义奖励。
        rewards += reward_model_scores

    return rewards


def grpo_train_epoch(epoch, loader, iters, ref_model, reward_model, reward_tokenizer, start_step=0, wandb=None):
    """
    执行一个 epoch 的 GRPO 训练。

    一个 batch 的核心流程：
    1. 对每个 prompt 采样生成 num_generations 条回答。
    2. 计算当前策略模型对这些回答每个 token 的 log prob。
    3. 计算参考模型对这些回答每个 token 的 log prob。
    4. 用 reward system 给每条回答打分。
    5. 在每个 prompt 的组内做 reward 标准化，得到 advantage。
    6. 用 policy gradient + KL 惩罚更新当前模型。
    """
    for step, batch in enumerate(loader, start=start_step + 1):
        # prompts 是一个长度为 B 的字符串列表，每个元素是一条对话 prompt。
        prompts = batch['prompt']  # list[str], length B

        # tokenizer 把文本 prompt 编码成 token id，并左侧 padding。
        # 左 padding 的原因是：生成时我们关心序列末尾，左 padding 对自回归生成更自然。
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)  # input_ids: [B, P], attention_mask: [B, P]
        if args.max_seq_len:
            # 截断过长 prompt，只保留最后 max_seq_len 个 token，
            # 避免上下文太长导致显存爆炸。
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        with torch.no_grad():
            # DDP 模型需要使用 .module 访问 generate 方法
            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            # 对每个 prompt 采样生成 num_generations 条回答。
            # do_sample=True 表示不是贪心解码，而是按概率采样，
            # 这样同一个 prompt 才能得到一组不同的候选回答。
            outputs = model_for_gen.generate(
                **prompt_inputs, max_new_tokens=args.max_gen_len, do_sample=True, temperature=0.8,
                num_return_sequences=args.num_generations, pad_token_id=tokenizer.pad_token_id)  # [B*num_gen, P+R]

        # completion_ids 只取新生成部分，不含原始 prompt。
        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]  # [B*num_gen, R]
        
        def get_per_token_logps(mdl, input_ids, n_keep):
            """
            计算模型对最后 n_keep 个 token 的逐 token log probability。

            为什么只保留最后 n_keep 个 token？
            因为 GRPO 只关心生成出来的 completion 部分，不关心 prompt 部分。
            """
            input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
            # logits_to_keep=n_keep+1 是因为 logits 需要和下一时刻 token 对齐，
            # 后面再通过 [:, :-1, :] 去掉最后一个无对应标签的位置。
            logits = mdl(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
            per_token_logps = []
            for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
                ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
                # 取出每个位置真实生成 token 对应的 log prob。
                per_token_logps.append(torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1))
            return torch.stack(per_token_logps)

        with autocast_ctx:
            # 当前策略模型对生成结果的逐 token log prob。
            per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))  # [B*num_gen, R]
            # 如果是 MoE，额外跑一次完整前向拿 aux_loss；Dense 模型则没有这个损失。
            res = model(outputs) if lm_config.use_moe else None
            aux_loss = res.aux_loss if res is not None else torch.tensor(0.0, device=args.device)
        
        with torch.no_grad():
            # 参考模型提供 KL 惩罚的基准分布，不参与更新。
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))  # [B*num_gen, R]

        # 把生成 token 解码成字符串，供奖励函数使用。
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer).to(args.device)  # [B*num_gen]

        # ========== 组内标准化：GRPO 的关键 ==========
        # 把每个 prompt 的 num_generations 条回答分成一组。
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        # 组内平均分和标准差，然后广播回每个回答。
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]

        # advantage 表示“该回答比同组平均水平高多少”。
        # 这一步替代了 PPO 中的 value model 估计，是 GRPO 最大的简化点。
        advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
        # 再做一次全 batch 标准化，进一步稳定训练。
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # [B*num_gen]

        # 找出每条生成序列的 eos 位置，只对 eos 前的 token 计算 loss。
        is_eos = completion_ids == tokenizer.eos_token_id  # [B*num_gen, R]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()  # [B*num_gen, R]

        # ========== KL 惩罚 ==========
        # ref - policy 的 log prob 差，衡量当前模型偏离参考模型的程度。
        kl_div = ref_per_token_logps - per_token_logps

        # 使用一个常见的逐 token KL 近似形式，保证数值稳定且始终非负。
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]

        # ========== GRPO policy loss ==========
        # exp(per_token_logps - per_token_logps.detach()) 的值在前向时等于 1，
        # 但它保留了对 per_token_logps 的梯度，相当于经典 policy gradient 中的
        # importance ratio 技巧的一个稳定实现。
        #
        # 若 advantage > 0，则该 token 所在回答优于组内平均，应提高其概率。
        # 若 advantage < 0，则该 token 所在回答低于组内平均，应降低其概率。
        per_token_loss = -(torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) - args.beta * per_token_kl)  # [B*num_gen, R]

        # 只对有效 completion 区域求平均，再对整个 batch 求平均。
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        loss = (policy_loss + aux_loss) / args.accumulation_steps  # scalar
        loss.backward()

        if (step + 1) % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                # 对策略模型的全部参数做梯度裁剪。
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Actor Loss: {policy_loss_val:.4f}, Aux Loss: {current_aux_loss:.4f}, Reward: {avg_reward_val:.4f}, '
                   f'Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')

            if wandb and is_main_process():
                wandb.log({
                    "policy_loss": policy_loss_val,
                    "aux_loss": current_aux_loss,
                    "reward": avg_reward_val,
                    "avg_response_len": avg_len_val,
                    "advantages_mean": advantages.mean().item(),
                    "learning_rate": current_lr
                })

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 保存完整策略模型权重，方便后续直接加载推理。
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        # 主动释放本 step 的大张量，GRPO 的显存占用很高，这一步很有必要。
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask


if __name__ == "__main__":
    # ======================== 命令行参数定义 ========================
    parser = argparse.ArgumentParser(description="CookMind GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    # 强化学习阶段学习率要非常小，否则策略会迅速崩坏。
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    # prompt 最大长度和生成最大长度会共同决定模型的 max_seq_len。
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=66, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", help="RLAIF数据路径")
    # 每个 prompt 采样多少条回答，是 GRPO 的关键超参数。
    # 数值越大，组内比较越稳定，但显存和时间开销也线性增长。
    parser.add_argument("--num_generations", type=int, default=8, help="每个prompt生成的样本数")
    # beta 控制 KL 惩罚强度：
    # 越大 -> 越不允许策略偏离参考模型；
    # 越小 -> 策略更新更激进。
    parser.add_argument("--beta", type=float, default=0.02, help="KL惩罚系数")
    # reasoning=1 时，会启用 <think>/<answer> 的格式奖励和 answer 单独打分。
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="CookMind-GRPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # max_seq_len 需要覆盖 prompt + completion 的总长度。
    lm_config = CookMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"CookMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    # reasoning 模型一般基于 reason 权重继续 RL；
    # 普通对话模型则基于 full_sft 权重继续 RL。
    base_weight = "reason" if args.reasoning == 1 else "full_sft"

    # Policy 模型：当前要被训练更新的策略模型。
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')

    # Reference 模型：冻结的基准模型，用于计算 KL 惩罚。
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    # Reward 模型：专门负责对回答质量打分，不参与训练。
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)

    # RLAIFDataset 只提供 prompt；回答由当前模型在线生成。
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # 先构建一个只用于统计长度的 DataLoader，计算每个 epoch 有多少步。
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    # 余弦退火调度器：学习率从初始值缓慢下降到十分之一。
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        # RoPE 的频率缓存不需要同步。
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # SkipBatchSampler 用于断点续训时跳过已经训练完的 batch。
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, ref_model, reward_model, reward_tokenizer, start_step, wandb)
        else:
            grpo_train_epoch(epoch, loader, len(loader), ref_model, reward_model, reward_tokenizer, 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()