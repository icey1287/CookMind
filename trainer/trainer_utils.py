"""
训练工具函数集合。

这里放的是训练脚本会反复调用、但不属于某一个模型模块本身的公共能力：
1. 打印参数量。
2. 判断当前是不是主进程。
3. 学习率调度。
4. DDP 初始化。
5. 随机种子设置。
6. 模型权重与断点恢复。
7. 跳过前若干 batch 的采样器。

把这些工具函数拆出来有两个好处：
1. train_pretrain.py 会更短，更容易读主流程。
2. 以后做 SFT、DPO、MoE 训练时可以复用同一套工具。
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model import CookMindForCausalLM

def get_model_params(model, config):
    """
    统计模型参数量。

    这里既计算总参数量，也尝试估算 MoE 场景下“每次前向真正激活的参数量”。
    对普通 dense 模型来说，active 和 total 基本一样。
    对 MoE 来说，总参数可能很大，但每个 token 只会走部分专家，所以活跃参数量更小。
    """
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: 
        Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else:
        Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    """
    判断当前进程是不是主进程。

    单卡训练时，没有分布式进程组，默认当前就是主进程。
    多卡 DDP 时，rank=0 通常负责打印日志、保存权重。
    这样可以避免每张卡都重复打印和重复保存。
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    """只在主进程打印日志，减少多卡训练时的重复输出。"""
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    计算当前 step 的学习率。

    这里使用的是一个简化版余弦退火：
    - 一开始学习率较高。
    - 随着训练推进逐步下降。
    - 最低会衰减到初始学习率的 10%。

    公式里的 0.1 + 0.45 * (1 + cos(...)) 展开后范围大约是 [0.1, 1.0]。
    也就是说学习率不会衰减到 0，而是保留一个较小但非零的下限。
    """
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    """
    初始化 PyTorch DDP 环境。

    如果环境变量里没有 RANK，说明当前不是通过 torchrun 启动的，
    那就直接走单卡/单进程模式。
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    # 使用 NCCL 作为 GPU 场景下的高性能通信后端。
    dist.init_process_group(backend="nccl")

    # LOCAL_RANK 表示当前进程应绑定到哪一张本地 GPU。
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    """
    固定随机种子，尽可能让实验结果可复现。

    注意：
    完全复现并不是 100% 保证的，尤其在不同硬件、不同 CUDA 版本下。
    但统一 random / numpy / torch 的种子后，结果会稳定很多。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    """
    统一处理模型保存与断点恢复。

    两种调用方式：
    1. 传入 model，表示“保存”。
    2. 不传 model，表示“加载 resume checkpoint”。

    会同时维护两个文件：
    - 主权重文件: 只保存模型参数，便于部署或继续训练。
    - resume 文件: 除模型外还保存优化器、step、epoch 等训练状态，便于断点续训。
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        # 如果模型外面包了一层 DDP，真正的裸模型在 model.module 里。
        # torch.compile 也可能在外面套一层 _orig_mod，这里一起剥掉。
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)

        # state_dict 是最标准的 PyTorch 存权重方式。
        state_dict = raw_model.state_dict()

        # 保存前转成 half + cpu，可以显著减小 checkpoint 体积。
        # 这里保存的是“文件里的精度”，不会影响当前内存中的训练精度。
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}

        # 先写临时文件，再原子替换正式文件。
        # 这样即便训练中途崩掉，也不容易把旧 checkpoint 写坏。
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        # resume_data 除了模型参数外，还会记录优化器状态、epoch、step 等信息。
        # 这些信息是“继续训练”必须的；只靠模型权重是不够的。
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }

        # kwargs 里额外传入的对象，如果也有 state_dict，就一起保存。
        # 比如 scaler、lr_scheduler 等，都可以通过这个入口扩展。
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data

        # 保存完后主动清理一下 CUDA cache，减少显存碎片。
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            # map_location='cpu' 表示先加载到 CPU，避免恢复时瞬间占满显存。
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                # 如果上次保存时的 GPU 数量和现在不同，step 的含义也会变化。
                # 这里做一个简单折算，让续训位置尽量合理。
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    """
    初始化 tokenizer 和模型。

    参数说明：
    - lm_config: 模型结构配置。
    - from_weight: 如果不是 'none'，则尝试加载已有权重继续训练。
    - tokenizer_path: tokenizer 所在目录，或者 Hugging Face 模型名。
    - save_dir: 权重文件目录。
    - device: 把模型放到哪个设备上。
    """

    # tokenizer 负责把文本转成 token id，也是数据预处理的重要组成部分。
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = CookMindForCausalLM(lm_config)

    if from_weight!= 'none':
        # 根据当前模型规模和是否使用 MoE，拼出权重文件名。
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)

        # strict=False 允许“部分参数匹配”。
        # 这在你后面逐步补 MoE 或调整结构时会更灵活。
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')

    # 返回已经放到目标设备上的模型，以及配套 tokenizer。
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    """
    可以“跳过前若干 batch”的采样器。

    这个采样器主要服务于断点续训。
    比如上次训练到第 300 个 batch 中断了，这次恢复时就可以直接跳过前 300 个 batch，
    而不是从 epoch 开头重新跑。
    """

    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        # 手动把底层 sampler 产出的单个样本索引，拼成一个个 batch。
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    # 还在“需要跳过”的范围里时，直接丢掉这个 batch。
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            # 处理最后一个不足 batch_size 的尾 batch。
            yield batch

    def __len__(self):
        # 先算总 batch 数，再减去要跳过的 batch 数。
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)