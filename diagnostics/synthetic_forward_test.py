
import json
import time
from pathlib import Path

import torch
from models.select_network import define_G

# 项目根目录
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "options" / "train_baseline.json"
WEIGHT = Path(
    r"D:\科研项目\RGB_Thermal\assets\weights\SwinFuSR\baseline"
)


def main():
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，停止测试。")

    # 读取官方配置，不修改原始 JSON
    with open(CONFIG, "r", encoding="utf-8") as f:
        opt = json.load(f)
    opt["is_train"] = False

    # 构建模型并严格加载权重
    model = define_G(opt)
    checkpoint = torch.load(
        WEIGHT, map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint, strict=True)
    print("Strict checkpoint loading: PASS")
    print("Model parameters:", sum(
        p.numel() for p in model.parameters()
    ))

    model = model.to("cuda").eval()

    # 合成输入，不是真实实验数据
    thermal = torch.rand(
        1, 1, 64, 64, dtype=torch.float32, device="cuda"
    )
    guide = torch.rand_like(thermal)

    print("Input shape:", tuple(thermal.shape))
    print("Input device:", thermal.device)

    with torch.inference_mode():
        # 预热
        _ = model(thermal, guide)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()

        for _ in range(3):
            output = model(thermal, guide)

        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 3

    print("Output shape:", tuple(output.shape))
    print("Output dtype:", output.dtype)
    print("Output device:", output.device)
    print("Output min/max:",
          output.min().item(), output.max().item())
    print("Output finite:",
          torch.isfinite(output).all().item())
    print("Average forward time (s):", round(elapsed, 4))
    print("Peak GPU memory (MiB):", round(
        torch.cuda.max_memory_allocated() / 1024**2, 1
    ))

    assert output.shape == thermal.shape, "输出尺寸异常"
    assert torch.isfinite(output).all(), "输出含 NaN 或 Inf"

    print("SYNTHETIC_FORWARD_TEST=PASS")


if __name__ == "__main__":
    main()
