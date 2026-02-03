# save as: flashattn_sanity.py
# run: python flashattn_sanity.py

import os, sys, time
import torch
import torch.nn.functional as F

def main():
    torch.manual_seed(0)

    # Basic environment print
    print("python:", sys.version.split()[0])
    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    # Confirm flash-attn import + version
    try:
        import flash_attn
        print("flash_attn:", getattr(flash_attn, "__version__", "unknown"))
    except Exception as e:
        print("flash_attn import FAILED:", repr(e))
        raise

    # Force a config that should exercise flash attention kernels
    device = "cuda"
    dtype = torch.bfloat16  # good default for H100
    B, H, T, D = 2, 8, 1024, 64  # modest but non-trivial
    assert D % 8 == 0

    # Create QKV (packed as expected by flash-attn kernels)
    # Shape: [B, T, 3, H, D]
    qkv = torch.randn(B, T, 3, H, D, device=device, dtype=dtype, requires_grad=True)

    # Use flash-attn's fused attention
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func

    out = flash_attn_qkvpacked_func(
        qkv,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
    )  # [B, T, H, D]

    # Simple loss + backward
    target = torch.zeros_like(out)
    loss = F.mse_loss(out, target)
    print("loss:", float(loss))

    loss.backward()

    # Validate finite values
    checks = {
        "out_isfinite": torch.isfinite(out).all().item(),
        "loss_isfinite": torch.isfinite(loss).item(),
        "qkv_grad_isfinite": torch.isfinite(qkv.grad).all().item(),
        "qkv_grad_norm": float(qkv.grad.float().norm().item()),
    }
    print("checks:", checks)

    if not (checks["out_isfinite"] and checks["loss_isfinite"] and checks["qkv_grad_isfinite"]):
        raise RuntimeError("FlashAttention sanity check failed: found NaN/Inf.")

    # Optional: run a few steps to catch intermittent NaNs
    qkv2 = qkv.detach().clone().requires_grad_(True)
    opt = torch.optim.AdamW([qkv2], lr=1e-2)
    for step in range(10):
        opt.zero_grad(set_to_none=True)
        out2 = flash_attn_qkvpacked_func(qkv2, dropout_p=0.0, causal=True)
        loss2 = (out2.float() ** 2).mean()
        loss2.backward()
        opt.step()

        ok = torch.isfinite(loss2).item() and torch.isfinite(qkv2.grad).all().item()
        print(f"step {step:02d} loss {float(loss2):.6f} finite {ok}")
        if not ok:
            raise RuntimeError(f"NaN/Inf detected at step {step}.")

    print("OK: flash-attn forward/backward stayed finite.")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available.")
    main()
