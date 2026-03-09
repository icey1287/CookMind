"""
LoRA 微调训练脚本。

整体流程：
1. 加载已经经过 Full SFT 的基础模型权重。
2. 调用 apply_lora() 给模型的方阵 Linear 层挂上 LoRA 旁路。
3. 冻结所有非 LoRA 参数（requires_grad=False），只训练 LoRA 的 A、B 矩阵。
4. 用 SFT 对话数据训练（和 Full SFT 使用相同格式的数据集）。
5. 只保存 LoRA 权重（文件极小，通常只有几 MB）。

与 Full SFT 的关键区别：
- Full SFT 更新模型所有参数，LoRA 只更新约 2~5% 的参数。
- Full SFT 保存完整模型权重，LoRA 只保存旁路权重。
- LoRA 更适合在小数据集上快速定制模型能力（如身份认知、医疗问答等）。
"""

import os
import sys

# 设置包名，使相对导入（如 from trainer.xxx）能正常工作。
__package__ = "trainer"
# 把项目根目录加到 Python 搜索路径，这样能找到 model/、dataset/ 等顶层包。
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
from model.model_lora import save_lora, apply_lora
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

# 屏蔽不影响训练的警告信息（如 tokenizer 并行警告），保持控制台输出整洁。
warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    """
    执行一个 epoch 的 LoRA 训练。

    参数说明：
    - epoch:       当前的 epoch 编号（从 0 开始）。
    - loader:      DataLoader，每次迭代取出 (input_ids, labels)。
    - iters:       本 epoch 的总步数（用于学习率调度和日志）。
    - lora_params: LoRA 参数列表，只对这些参数做梯度裁剪。
    - start_step:  断点续训时跳过的步数。
    - wandb:       可选的实验跟踪工具（这里用的是 swanlab）。
    """
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 把数据搬到 GPU（或 CPU），和模型在同一个设备上。
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)

        # 动态调整学习率：根据当前全局步数，按照预热 + 余弦退火策略计算 lr。
        # epoch * iters + step 是全局步数，args.epochs * iters 是总步数。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ---- 前向传播 ----
        with autocast_ctx:
            # 模型前向：输入 token 序列和对应标签，模型内部会自动计算交叉熵损失。
            # res.loss = 语言模型的 next-token 预测损失。
            # res.aux_loss = MoE 架构的辅助负载均衡损失（非 MoE 时为 0 或 None）。
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss

            # 梯度累积：把 loss 除以累积步数，这样多次 backward 累加的梯度
            # 等效于一个大 batch 的梯度。
            loss = loss / args.accumulation_steps

        # ---- 反向传播 ----
        # scaler.scale(loss) 在 float16 混合精度下放大 loss 防止梯度下溢。
        # 在 bfloat16 模式下 scaler 实际是禁用的（enabled=False），不影响计算。
        scaler.scale(loss).backward()

        # ---- 参数更新（每累积 accumulation_steps 步后执行一次）----
        if (step + 1) % args.accumulation_steps == 0:
            # 把 scaler 缩放过的梯度还原回真实尺度。
            scaler.unscale_(optimizer)

            # 梯度裁剪：只裁剪 LoRA 参数的梯度（不裁剪冻结参数，它们没有梯度）。
            # 限制梯度的 L2 范数不超过 grad_clip，防止训练不稳定。
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)

            # 执行一步优化器更新（只有 LoRA 参数会被更新，因为只有它们有梯度）。
            scaler.step(optimizer)
            scaler.update()

            # 清零梯度，准备下一轮累积。set_to_none=True 比 zero_grad() 更省显存。
            optimizer.zero_grad(set_to_none=True)

        # ---- 日志打印 ----
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            # 还原真实的 loss 值（之前除以了 accumulation_steps）。
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            # logits_loss = 纯语言模型损失（不含 MoE 辅助损失）。
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            # 估算本 epoch 剩余时间（单位：分钟）。
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb:
                wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # ---- 模型保存（只保存 LoRA 权重）----
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}.pth'
            # 与 Full SFT 不同：这里只保存 LoRA 的 A、B 矩阵权重，文件极小。
            save_lora(model, lora_save_path)
            # 同时保存完整的训练状态（含优化器等），用于断点续训。
            lm_checkpoint(lm_config, weight=args.lora_name, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()

        # 手动释放本步的临时变量，帮助 Python 垃圾回收更快回收显存。
        del input_ids, labels, res, loss


if __name__ == "__main__":
    # ======================== 命令行参数定义 ========================
    # 这些参数控制了 LoRA 训练的所有可调配置。
    parser = argparse.ArgumentParser(description="CookMind LoRA Fine-tuning")
    parser.add_argument("--save_dir", type=str, default="../out/lora", help="模型保存目录")
    # lora_name 决定了保存文件名和 checkpoint 名，
    # 不同的 LoRA 任务（身份认知、医疗问答等）可以用不同名字区分。
    parser.add_argument("--lora_name", type=str, default="lora_identity", help="LoRA权重名称(如lora_identity/lora_medical等)")
    # LoRA 通常需要较多 epoch（50），因为可训练参数很少，需要更多迭代才能学会。
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    # LoRA 的学习率通常比 Full SFT 高一个量级（1e-4 vs 1e-6），
    # 因为只更新少量参数，较大的步长才能产生足够的效果。
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/lora_identity.jsonl", help="LoRA训练数据路径")
    # LoRA 默认基于 full_sft 权重训练（先全量微调，再 LoRA 微调特定能力）。
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练，默认full_sft")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="CookMind-LoRA", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 检测是否通过 torchrun 启动了多 GPU 分布式训练，返回当前进程的本地 rank。
    local_rank = init_distributed_mode()
    # 多 GPU 时，每个进程绑定到自己负责的那张卡。
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    # 固定随机种子保证可复现性，不同 rank 用不同种子避免数据重复。
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查 checkpoint ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 用命令行参数构建模型配置（架构超参数）。
    lm_config = CookMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 如果开启了断点续训（from_resume=1），尝试从 checkpoints 目录加载之前的训练状态。
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # bfloat16 可以表示更大的数值范围（和 float32 一样的指数位），不容易溢出。
    # float16 范围较小，搭配 GradScaler 使用以防止梯度下溢。
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # CPU 不支持自动混合精度，使用 nullcontext() 跳过。
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置实验跟踪（swanlab / wandb）==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        # 如果有之前的 wandb_id，用 resume='must' 接续之前的实验。
        resume = 'must' if wandb_id else None
        wandb_run_name = f"CookMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、应用 LoRA、冻结非 LoRA 参数 ==========
    # 加载基础模型和分词器（默认加载 full_sft 的预训练权重）。
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    if args.use_compile == 1:
        # torch.compile 会对模型做图编译优化，加速前向和反向传播。
        model = torch.compile(model)
        Logger('torch.compile enabled')

    # 核心步骤：给模型的方阵 Linear 层挂上 LoRA 旁路。
    # 调用后，每个被选中的 Linear 层的 forward 都变成 y = Wx + BAx。
    apply_lora(model)
    
    # 统计并打印参数量，直观展示 LoRA 的参数效率。
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")
    
    # 冻结所有非 LoRA 参数，只允许 LoRA 的 A、B 矩阵接收梯度。
    # 这是 LoRA 的核心：基础模型完全不动，只训练旁路。
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            # LoRA 参数：开启梯度，收集到列表中供优化器使用。
            param.requires_grad = True
            lora_params.append(param)
        else:
            # 非 LoRA 参数：冻结，不参与梯度计算，节省显存。
            param.requires_grad = False
    
    # ========== 6. 定义数据和优化器 ==========
    # 使用 SFTDataset 加载对话数据（和 Full SFT 共用数据格式）。
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # GradScaler 只在 float16 时启用；bfloat16 不需要缩放。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 优化器只接收 LoRA 参数，不会浪费计算在冻结参数上。
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    
    # ========== 7. 从 checkpoint 恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # strict=False：因为 checkpoint 可能包含 LoRA 参数的 key，
        # 而如果 LoRA 结构刚通过 apply_lora 挂上去，key 可能不完全匹配。
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 8. DDP 包装模型（多 GPU 训练）==========
    if dist.is_initialized():
        # freqs_cos / freqs_sin 是 RoPE 的预计算缓冲区，不需要跨 GPU 同步。
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 9. 开始训练循环 ==========
    for epoch in range(start_epoch, args.epochs):
        # 分布式训练时，每个 epoch 设置不同的 seed 让各 GPU 的采样顺序不同。
        train_sampler and train_sampler.set_epoch(epoch)
        # 固定 epoch 种子后打乱数据顺序。
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        # 断点续训时跳过已训练的步数。
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # SkipBatchSampler 跳过前 skip 个 batch，实现断点续训。
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        # pin_memory=True 把数据预先锁定在内存中，加速 CPU→GPU 的数据传输。
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, lora_params, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)
    
    # ========== 10. 清理分布式进程 ==========
    # 训练结束后销毁进程组，释放多 GPU 通信资源。
    if dist.is_initialized():
        dist.destroy_process_group()