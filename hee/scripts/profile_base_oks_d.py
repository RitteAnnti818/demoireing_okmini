import torch
from vd.archs.net_arch import DemoireNet
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import time

if __name__ == "__main__":
    # 모델 초기화
    model_d = DemoireNet().cuda()
    model_d.eval()

    # 더미 입력 (멀티프레임: B=1, T=3, C=3, H=480, W=640)
    dummy_input = torch.randn(1, 3, 3, 480, 640).cuda()

    # FLOPs 분석
    print("🔍 FLOPs 계산 중...")
    flops = FlopCountAnalysis(model_d, dummy_input)
    print(f"⚙️ DemoireNet FLOPs: {flops.total() / 1e9:.2f} GFLOPs")

    # 파라미터 수
    print("\n📦 Parameters:")
    print(parameter_count_table(model_d))

    # 추론 시간 측정
    print("\n⏱️ 추론 시간 측정:")
    for _ in range(10):  # warm-up
        _ = model_d(dummy_input)

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        _ = model_d(dummy_input)
    torch.cuda.synchronize()
    end = time.time()

    print(f"🚀 Inference time: {(end - start) * 1000:.2f} ms")

    output_path = "scripts/profile_result_d.txt"
    with open(output_path, "w") as f:
        f.write(f"⚙️ FLOPs: {flops.total() / 1e9:.2f} GFLOPs\n")
        f.write(parameter_count_table(model_d) + "\n")
        f.write(f"🚀 Inference time: {(end - start) * 1000:.2f} ms\n")

    print(f"✅ 프로파일 결과 저장됨: {output_path}")