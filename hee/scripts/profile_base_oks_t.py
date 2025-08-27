import torch
from vd.archs.net_arch import TemporalNet
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import time

if __name__ == "__main__":
    model_t = TemporalNet().cuda()
    model_t.eval()

    # 💡 TemporalNet expects (B*3, C, H, W)
    B = 1
    T = 3
    C = 64
    H, W = 60, 80

    feat_l1 = torch.randn(B * T, C, H, W).cuda()
    feat_l2 = torch.randn(B * T, C, H // 2, W // 2).cuda()
    feat_l3 = torch.randn(B * T, C, H // 4, W // 4).cuda()

    # FLOPs 분석
    print("🔍 FLOPs 계산 중...")
    flops = FlopCountAnalysis(model_t, (feat_l1, feat_l2, feat_l3))
    print(f"⚙️ TemporalNet FLOPs: {flops.total() / 1e9:.2f} GFLOPs")

    # 파라미터 수
    print("\n📦 Parameters:")
    print(parameter_count_table(model_t))

    # 추론 시간 측정
    print("\n⏱️ 추론 시간 측정:")
    for _ in range(10):  # warm-up
        _ = model_t(feat_l1, feat_l2, feat_l3)

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        _ = model_t(feat_l1, feat_l2, feat_l3)
    torch.cuda.synchronize()
    end = time.time()

    print(f"🚀 Inference time: {(end - start) * 1000:.2f} ms")

    output_path = "scripts/profile_result_t.txt"
    with open(output_path, "w") as f:
        f.write(f"⚙️ FLOPs: {flops.total() / 1e9:.2f} GFLOPs\n")
        f.write(parameter_count_table(model_t) + "\n")
        f.write(f"🚀 Inference time: {(end - start) * 1000:.2f} ms\n")

    print(f"✅ 프로파일 결과 저장됨: {output_path}")
