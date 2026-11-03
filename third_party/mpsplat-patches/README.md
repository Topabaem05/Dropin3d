# mpsplat 패치: gsplat 1.5.3과 같은 합성 규칙

[mpsplat](https://github.com/zacsmms/mpsplat)(커밋 `94c8949`, Apache-2.0)의 3DGS 래스터라이저를 gsplat 1.5.3 CUDA 커널(`RasterizeToPixels3DGSFwd.cu`, `RasterizeToPixels3DGSBwd.cu`)과 같은 규칙으로 바꾸는 patch입니다.
gsplat 1.5.3으로 학습한 결과(예: 4DGS360)를 Mac에서 재현하려고 만들었습니다.

## 바뀌는 규칙

| 항목 | mpsplat 원본 | 패치 후 (= gsplat 1.5.3 CUDA) |
|---|---|---|
| alpha 상한 | 0.99 | 0.999 |
| alpha < 1/255인 Gaussian | 합성에 포함 | 건너뜀 |
| sigma < 0 | alpha = 0으로 포함 (`last_ids`가 갱신됨) | 건너뜀 |
| `1 - alpha` 하한 | 1e-7로 clamp | clamp 없음 (alpha ≤ 0.999라서 필요 없음) |
| 조기 종료 | 없음 | 다음 T ≤ 1e-4이면 그 Gaussian을 빼고 멈춤 |
| `last_ids` | 마지막으로 순회한 index | 마지막으로 합성된 Gaussian의 index |

## 바뀌는 파일
- `gsplat/mps/_shaders/rasterize_to_pixels_3dgs_fwd.metal`: Metal forward
- `gsplat/mps/_shaders/rasterize_to_pixels_3dgs_bwd.metal`: Metal backward (forward와 짝을 맞춤)
- `gsplat/cuda/_torch_rasterize.py`: 3DGS PyTorch 경로. 3채널이 아니거나 background가 있을 때 쓰이는 경로라서 같은 규칙으로 맞췄습니다. 2DGS 등 다른 래스터라이저는 바꾸지 않았습니다.

projection 단계(Gaussian 반경 계산)는 1.5.3과 이미 같아서 바꾸지 않았습니다.

## 적용 방법 (Mac)

1. mpsplat을 받아 기준 커밋으로 맞춥니다.
   ```bash
   git clone https://github.com/zacsmms/mpsplat && cd mpsplat && git checkout 94c89496f99d74d81e2a298e1f6159cc3978af34
   ```
2. patch를 적용합니다.
   ```bash
   git apply /path/to/dropin3d/third_party/mpsplat-patches/0001-gsplat-1.5.3-compositing-parity.patch
   ```
3. 설치합니다.
   ```bash
   pip install -e .
   ```
4. Metal 경로와 PyTorch 경로가 같은 결과를 내는지 확인합니다.
   ```bash
   python /path/to/dropin3d/tools/check_mpsplat_parity.py
   ```

## 검증 상태
- ✅ CPU 테스트(`tests/test_gsplat153_compositing.py`):
  - patch의 벡터화 forward가 gsplat 1.5.3 CUDA forward의 픽셀 루프와 같은 결과를 냅니다.
  - patch한 Metal backward의 공식을 그대로 옮긴 코드가 autograd gradient와 일치합니다.
  - 테스트 장면에서 조기 종료, 건너뛰기, alpha 상한 분기가 모두 실행됩니다.
- ⚠️ Metal 셰이더 컴파일과 실행은 Mac에서 확인해야 합니다(`tools/check_mpsplat_parity.py`).

## 남는 차이
- **gradient 누적 순서**: Metal backward는 스레드마다 float atomic으로 더합니다. CUDA는 warp 단위로 합친 뒤 더합니다. 그래서 gradient의 마지막 몇 비트가 실행마다 다를 수 있습니다. CUDA도 atomic을 쓰므로 원래 완전히 결정적이지는 않습니다.
- **`exp` 구현**: CUDA는 `__expf`(빠른 근사), Metal은 `exp`를 씁니다. float 반올림 수준의 차이가 날 수 있습니다.
- **background가 있을 때의 backward**: Metal 커널은 background가 없을 때만 쓰이므로 background 항이 필요 없습니다. background는 `dropin3d.raster`가 따로 합성하고, 그 gradient는 autograd가 계산합니다.
