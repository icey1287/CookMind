"""
LoRA（Low-Rank Adaptation）模块。

核心思想：
  大模型全量微调需要更新全部参数，显存开销巨大。
  LoRA 的做法是：冻结原模型所有权重，只在选定的线性层旁边
  "并联"一个极小的低秩旁路 ΔW = B × A，只训练 A 和 B 两个小矩阵。

  原始前向：   y = Wx
  LoRA 前向：  y = Wx + BAx
  其中 A ∈ R^{d×r}，B ∈ R^{d×r}（r << d），参数量从 d² 降到 2dr。

  举个直觉例子：
  如果 hidden_size=512，一个 Linear 层有 512×512 = 262,144 个参数。
  rank=8 时，A 只有 512×8=4096，B 也是 8×512=4096，
  总共 8,192 个参数，仅为原来的 3.1%，却能让模型学会新任务。
"""

import torch
from torch import nn


class LoRA(nn.Module):
    """
    LoRA 低秩适配器模块。

    结构：输入 x → A（降维到 rank）→ B（升维回原始维度）→ 输出 ΔW·x
    最终效果等价于给原权重加了一个低秩扰动：W' = W + B·A

    参数说明：
    - in_features:  输入特征维度，和原始 Linear 层的输入维度一致。
    - out_features: 输出特征维度，和原始 Linear 层的输出维度一致。
    - rank:         低秩的秩（r），越小参数越少但表达能力越弱，通常取 4~16。
    """

    def __init__(self, in_features, out_features, rank):
        super().__init__()
        # 低秩的秩 r，决定了 A 和 B 中间的"瓶颈"维度。
        self.rank = rank

        # 矩阵 A：把输入从 in_features 维度压缩到 rank 维度（降维）。
        # 不加 bias，因为 LoRA 论文中证明无 bias 效果更好且更简洁。
        self.A = nn.Linear(in_features, rank, bias=False)

        # 矩阵 B：把 rank 维度的中间表示重新映射回 out_features 维度（升维）。
        self.B = nn.Linear(rank, out_features, bias=False)

        # 初始化策略非常关键：
        # A 用小幅高斯随机初始化，让不同的 LoRA 模块起点略有差异。
        self.A.weight.data.normal_(mean=0.0, std=0.02)

        # B 全零初始化。这保证了训练刚开始时 ΔW = B·A = 0，
        # 即 LoRA 旁路的输出为 0，模型行为和原始预训练模型完全一致。
        # 训练过程中 B 的权重会逐渐从零"长出来"，实现平滑微调。
        self.B.weight.data.zero_()

    def forward(self, x):
        # 先经过 A 降维，再经过 B 升维，得到低秩旁路的输出 ΔW·x。
        # 这个输出会和原始 Linear 的输出相加（在 apply_lora 中实现）。
        return self.B(self.A(x))


def apply_lora(model, rank=8):
    """
    遍历模型所有模块，找到"方阵"形状的 Linear 层，给它们挂载 LoRA 旁路。

    为什么只选方阵（in == out）？
    因为在 Transformer 中，QKV 投影和 Attention 输出投影通常是方阵
    （hidden_size → hidden_size），这些层对模型行为影响最大。
    而 lm_head（hidden_size → vocab_size）等非方阵层不会被改动。

    参数说明：
    - model: 要应用 LoRA 的基础模型（已加载预训练权重）。
    - rank:  LoRA 的秩，默认 8。
    """
    for name, module in model.named_modules():
        # 只对方阵 Linear 层应用 LoRA（输入维度 == 输出维度）。
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 创建 LoRA 模块并放到和模型相同的设备上（CPU/GPU）。
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)

            # 把 LoRA 模块作为子模块挂到原始 Linear 上，
            # 这样 model.named_parameters() 能自动发现它的参数。
            setattr(module, "lora", lora)

            # 保存原始 forward 方法的引用。
            original_forward = module.forward

            # 用默认参数绑定（Python 闭包陷阱的经典解法）：
            # 如果不用 layer1=original_forward, layer2=lora 显式绑定，
            # 循环中所有 forward_with_lora 都会指向最后一个 module 的 forward 和 lora。
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                # 核心公式：y = W·x + ΔW·x = layer1(x) + layer2(x)
                return layer1(x) + layer2(x)

            # 替换原始 Linear 的 forward 方法为带 LoRA 旁路的版本。
            module.forward = forward_with_lora


def load_lora(model, path):
    """
    从磁盘加载 LoRA 权重并注入到模型中。

    注意：调用此函数之前，必须先调用 apply_lora(model) 把 LoRA 结构挂上去，
    否则 module 上没有 .lora 属性，权重无处可放。

    参数说明：
    - model: 已经 apply_lora 过的模型。
    - path:  LoRA 权重文件路径（.pth）。
    """
    # 加载权重字典，map_location 确保在正确的设备上加载。
    state_dict = torch.load(path, map_location=model.device)

    # 兼容 DDP 保存的权重：DDP 会给所有 key 加上 'module.' 前缀，这里去掉。
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    # 遍历模型所有模块，找到有 .lora 属性的层，把对应权重加载进去。
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 从全局 state_dict 中筛选出属于当前 module 的 LoRA 权重，
            # 并把 key 中的前缀去掉，只留下 'A.weight' / 'B.weight'。
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """
    只保存模型中 LoRA 部分的权重（不保存原始大模型的权重）。

    这使得 LoRA 权重文件非常小（通常只有几 MB），
    而原始大模型的权重可以复用，不需要每次都保存一份完整副本。

    参数说明：
    - model: 训练中的模型（可能被 DDP 或 torch.compile 包装过）。
    - path:  保存路径。
    """
    # 如果模型被 torch.compile 包装过，需要取出原始模型。
    # torch.compile 会生成一个 wrapper，原始模型存在 _orig_mod 属性里。
    raw_model = getattr(model, '_orig_mod', model)

    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            # 去掉 DDP 可能添加的 'module.' 前缀，保持 key 的纯净。
            clean_name = name[7:] if name.startswith("module.") else name

            # 把 LoRA 子模块的参数存入字典，key 格式如：
            # 'layers.0.attention.wq.lora.A.weight'
            # 'layers.0.attention.wq.lora.B.weight'
            lora_state = {f'{clean_name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)

    # 只保存 LoRA 参数，文件极小。
    torch.save(state_dict, path)