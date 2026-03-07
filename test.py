import torch
print("PyTorch版本：", torch.__version__)
print("CUDA可用：", torch.cuda.is_available())
print("MPS可用（Mac Metal GPU）：", torch.backends.mps.is_available())