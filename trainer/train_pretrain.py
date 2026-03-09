"""
预训练主脚本。

这个文件做的事情可以概括成 9 步：
1. 解析命令行参数。
2. 初始化随机种子和分布式环境。
3. 创建模型配置。
4. 初始化混合精度环境。
5. 初始化日志系统（可选 wandb/swanlab）。
6. 构建模型、数据集、DataLoader、优化器。
7. 如果存在断点，则恢复训练状态。
8. 进入 epoch / step 训练循环。
9. 保存权重并在结束时清理 DDP。

如果你刚开始学训练脚本，建议优先看这两个函数：
1. train_epoch: 单个 epoch 内部到底执行了什么。
2. main 里的训练主循环：一个完整训练过程是如何被串起来的。
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model import CookMindConfig
from dataset.lm_dataset import PretrainDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')

def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    跑完一个 epoch 的训练。

    参数说明：
    - epoch: 当前是第几个 epoch。
    - loader: DataLoader，按 batch 产出训练数据。
    - iters: 当前 epoch 预计总 step 数，用于日志和学习率计算。
    - start_step: 断点续训时，本 epoch 从第几个 step 开始。
    - wandb: 可选的实验记录器。
    """

    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader,start=start_step+1):
        # 把 batch 数据搬到训练设备上，通常是 GPU。
        input_ids=input_ids.to(args.device)
        labels=labels.to(args.device)

        # 这里每个 step 都重新计算并设置学习率。
        # 所以优化器里的 lr 实际上是“动态变化”的，而不是固定常数。
        lr=get_lr(epoch*iters+step,args.epochs*iters,args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr']=lr
        
        with autocast_ctx:
            # 前向传播：得到 logits、loss，以及预留给 MoE 的 aux_loss。
            res=model(input_ids,labels=labels)

            # 注意：如果 aux_loss 还是 None，这里直接相加会报错。
            # 这说明当前脚本默认假设 aux_loss 已经是一个 tensor。
            loss=res.loss+res.aux_loss

            # 梯度累积时，每一步只反传总 loss 的 1 / accumulation_steps。
            # 否则累计多步后，等效梯度会被放大很多倍。
            loss=loss/args.accumulation_steps

        # 用 GradScaler 包一层 backward，是混合精度训练的标准写法。
        scaler.scale(loss).backward()

        # 只有累计够指定步数，才真正执行一次参数更新。
        if (step + 1) % args.accumulation_steps == 0:
            # 先把缩放过的梯度还原，再做裁剪，避免裁剪结果失真。
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 更新参数并推进 scaler 状态。
            scaler.step(optimizer)
            scaler.update()

            # 清空梯度，准备下一轮累积。
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            # 日志里展示的是“恢复成原始尺度后的 loss”。
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']

            # 粗略估算本 epoch 还需要多少分钟。
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            # 保存前切到 eval，避免某些层在 train/eval 模式下行为不同。
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'

            # 兼容 DDP 和 torch.compile 场景，取到真正的裸模型。
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()

            # 保存一份纯模型权重，方便后续推理或继续训练。
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)

            # 同时保存 resume checkpoint，里面还包含优化器、epoch、step 等训练状态。
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        # 手动删除局部变量，有助于尽早释放显存引用。
        del input_ids, labels, res, loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CookMind Pretraining")

    # 下面是训练时可调的超参数和路径参数。
    # 它们都可以通过命令行覆盖默认值。
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数（建议1轮zero或2-6轮充分训练）")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="CookMind-Pretrain", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # init_distributed_mode 会判断当前是不是用 torchrun 启动的多卡训练。
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"

    # 多卡时给不同 rank 使用不同随机种子，可以减少完全相同的随机行为。
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)

    # 根据命令行参数创建模型配置。
    lm_config = CookMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))

    # 如果开启自动续训，就尝试从 resume checkpoint 恢复训练状态。
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    # device_type 影响 autocast 的具体行为。
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # CPU 上通常不使用 autocast，因此直接用 nullcontext 占位。
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        # 这里只在主进程初始化实验记录，避免重复创建 run。
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"CookMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    # init_model 会同时返回模型和 tokenizer。
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        # torch.compile 会尝试对模型进行图编译，以换取更好的执行性能。
        model = torch.compile(model)
        Logger('torch.compile enabled')

    # 构建预训练数据集。
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    # 多卡时使用 DistributedSampler，让不同进程处理不同数据分片。
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    # 只有 float16 混合精度时 GradScaler 才真正启用。
    # bfloat16 一般不需要梯度缩放。
    scaler = torch.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 恢复模型、优化器和 scaler 的内部状态。
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        # 这两个 buffer 是预计算位置编码表，不需要参与 DDP 参数同步。
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # DistributedSampler 每个 epoch 都要 set_epoch，保证多卡 shuffle 一致且可复现。
        train_sampler and train_sampler.set_epoch(epoch)

        # 这里单机模式下手动生成随机索引列表，作为 sampler 的替代。
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()

        # 如果是断点恢复且恢复位置还在当前 epoch，就跳过已经训练过的 batch。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)

        # 注意这里传了 batch_sampler，因此不再需要单独的 batch_size/shuffle 参数。
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    # DDP 训练结束后显式销毁进程组，是一个比较干净的收尾动作。
    if dist.is_initialized(): dist.destroy_process_group()