"""
全量监督微调（Full SFT）训练脚本。

预训练阶段让模型学会了"语言"和"知识"，但它只会续写文本，不会对话。
SFT 阶段用人工标注的对话数据（user/assistant 格式），教模型学会：
1. 理解用户的提问意图。
2. 按照对话模板格式组织回答。
3. 只对 assistant 的回复部分计算 loss（user 部分被 mask 成 -100）。

Full SFT 与 LoRA SFT 的区别：
- Full SFT 更新模型的所有参数，训练更彻底但显存开销大。
- LoRA SFT 只更新旁路参数，适合小数据集的轻量定制。
"""

import os
import sys

# 设置包名和搜索路径，使跨目录的相对导入能正常工作。
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model import CookMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    执行一个 epoch 的全量 SFT 训练。

    与 LoRA 版 train_epoch 的核心区别：
    - 梯度裁剪是对 model.parameters()（全部参数），而不是仅 lora_params。
    - 保存的是完整模型权重（而不只是 LoRA 旁路）。
    """
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        # 动态学习率：用 get_lr 计算本步的学习率（预热 + 余弦退火）。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ---- 前向传播（混合精度上下文中执行）----
        with autocast_ctx:
            # 模型接收 input_ids 和 labels，内部自动：
            # 1. 将 input_ids 过 Embedding → Transformer Blocks → lm_head 得到 logits。
            # 2. 将 logits 和 labels 做错位对齐后计算 CrossEntropyLoss。
            # 3. labels 中为 -100 的位置（user 的话和 padding）不参与 loss 计算。
            res = model(input_ids, labels=labels)
            # loss = 语言模型损失 + MoE 辅助损失（非 MoE 时 aux_loss 为 0）。
            loss = res.loss + res.aux_loss
            # 梯度累积：除以步数，使累加后的梯度等效于大 batch。
            loss = loss / args.accumulation_steps

        # ---- 反向传播 ----
        scaler.scale(loss).backward()

        # ---- 每累积 N 步后执行一次参数更新 ----
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            # 全量 SFT：对所有参数做梯度裁剪（和 LoRA 不同，LoRA 只裁剪 lora_params）。
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        # ---- 日志打印 ----
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb:
                wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # ---- 模型保存（保存完整模型权重）----
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            # 文件名中加上 _moe 后缀以区分 MoE 和 Dense 模型。
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 如果用了 DDP 包装，真实模型在 model.module 里面。
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            # 如果用了 torch.compile，真实模型在 _orig_mod 属性里。
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 保存时转成 float16 以节省磁盘空间（权重精度损失极小）。
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 同时保存完整训练状态（优化器、scaler 等），用于断点续训。
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()
            del state_dict

        # 释放本步临时变量，加速垃圾回收。
        del input_ids, labels, res, loss


if __name__ == "__main__":
    # ======================== 命令行参数定义 ========================
    parser = argparse.ArgumentParser(description="CookMind Full SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    # Full SFT 通常只需 2~3 个 epoch，因为更新全部参数，学习效率高。
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    # Full SFT 学习率极低（1e-6），因为模型已有预训练知识，
    # 学习率太高会导致灾难性遗忘（catastrophic forgetting），把预训练学到的知识覆盖掉。
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl", help="训练数据路径")
    # from_weight='pretrain' 表示基于预训练权重继续训练。
    # 如果设为 'none'，则从随机初始化开始（不推荐，效果会很差）。
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="CookMind-Full-SFT", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查 checkpoint ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = CookMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置实验跟踪 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"CookMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    # 加载模型和分词器，默认从 pretrain 权重开始（预训练 → Full SFT 是标准流程）。
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    # SFTDataset 会将对话数据格式化为 chat template，并只给 assistant 的回复生成 labels。
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # Full SFT：优化器接收模型的所有参数（和 LoRA 不同）。
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从 checkpoint 恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP 包装模型 ==========
    if dist.is_initialized():
        # RoPE 的预计算正弦/余弦缓冲区无需跨 GPU 同步。
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练循环 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布式进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()