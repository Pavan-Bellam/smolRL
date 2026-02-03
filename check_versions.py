import sys
import platform
import torch

print("python:", sys.version.replace("\n", " "))
print("python_exe:", sys.executable)
print("platform:", platform.platform())
print("machine:", platform.machine())

print("torch:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("gpu_name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

# Critical for choosing cxx11abiTRUE vs FALSE wheels
print("cxx11abi:", torch._C._GLIBCXX_USE_CXX11_ABI)

# Optional: helpful for matching "cu12" expectation
try:
    import subprocess
    print("nvidia-smi:", subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"]).decode().strip())
except Exception as e:
    print("nvidia-smi: unavailable", repr(e))
