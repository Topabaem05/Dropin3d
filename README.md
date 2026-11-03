# Dropin3d

**Drop-in replacements for CUDA-only libraries in 3D vision research code.**
On NVIDIA GPUs each module calls the original library, so CUDA results do not change.
On Apple Silicon (MPS) or CPU it switches to a portable implementation (PyTorch / NumPy / scikit-learn) with the same API.

3D 비전 연구 코드(3DGS, 4DGS, 3D 트래킹 등)에서 자주 쓰는 **CUDA 전용 라이브러리의 대체 구현 모음**입니다.
NVIDIA GPU에서는 원래 라이브러리를 그대로 호출하므로 결과가 바뀌지 않습니다.
Apple Silicon(MPS)이나 CPU에서는 같은 API를 가진 이식 가능한 구현으로 자동 전환합니다.

## 대체하는 라이브러리

| 원래 API | dropin3d | Mac/CPU 구현 | 원본과 같은 결과인가 |
|---|---|---|---|
| `pytorch3d.ops.knn_points` | `dropin3d.knn.knn_points` | 정확한 brute-force kNN (PyTorch, gradient 지원) | ✅ 같음 (거리가 완전히 같은 점끼리는 순서가 다를 수 있음) |
| `pointops2.knnquery` (Point Transformer, TAPIP3D) | `dropin3d.knn.knnquery` | 정확한 kNN, CUDA 커널과 같은 패딩 규칙 | ✅ 같음 |
| `cuml.KMeans`, `cuml.HDBSCAN` | `dropin3d.cluster` | scikit-learn | ⚠️ KMeans 초기화 알고리즘이 달라 label이 다름 |
| CuPy 배열 코드 | `dropin3d.array` (`xp`) | NumPy | ✅ 같음 |
| `fused_ssim` | `dropin3d.ssim.fused_ssim` | fused-ssim의 MPS 커널, 없으면 PyTorch SSIM | ✅ 같은 정의 |
| `gsplat.rasterization` | `dropin3d.raster.rasterization` | [mpsplat](https://github.com/zacsmms/mpsplat)의 Metal 커널을 쓰도록 3채널 분할 + background 직접 합성 | ⚠️ mpsplat 패치를 적용하면 gsplat 1.5.3과 같은 합성 규칙 (아래 참고) |
| `torch.autocast("cuda", bfloat16)` | `dropin3d.precision` | 하드웨어에 맞는 dtype 선택 (sm_80 미만, MPS, CPU는 fp32) | ✅ 같거나 더 정밀함 |
| `torch.device("cuda")`, `torch.cuda.empty_cache()` | `dropin3d.device` | cuda → mps → cpu 자동 선택 | ✅ |

## 설치

```bash
pip install "dropin3d[cluster] @ git+https://github.com/Topabaem05/Dropin3d"
```

Mac에서 gsplat 대신 mpsplat을 설치합니다(`gsplat` 패키지 이름을 그대로 씁니다).
```bash
pip install "gsplat @ git+https://github.com/zacsmms/mpsplat@94c89496f99d74d81e2a298e1f6159cc3978af34"
```

MPS에 없는 연산은 CPU로 보냅니다.
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## 사용법

import 한 줄만 바꾸면 됩니다.

```python
# from pytorch3d.ops import knn_points
from dropin3d.knn import knn_points

# from fused_ssim import fused_ssim
from dropin3d.ssim import fused_ssim

# from gsplat.rendering import rasterization
from dropin3d.raster import rasterization   # 키워드 인자로 호출해야 함

# import third_party.pointops2.functions.pointops as pointops
from dropin3d import knn as pointops         # pointops.knnquery(...)

# from cuml import KMeans
from dropin3d.cluster import kmeans_labels   # labels = kmeans_labels(x, n_clusters)

from dropin3d import get_device, empty_cache, manual_seed_all
from dropin3d.precision import resolve_precision, autocast_context, no_autocast
```

## 환경변수

| 변수 | 값 | 설명 |
|---|---|---|
| `DROPIN3D_DEVICE` | `cuda` / `mps` / `cpu` | device를 강제로 지정 |
| `DROPIN3D_KNN_BACKEND` | `native` / `torch` | kNN 백엔드 (native = pytorch3d, pointops2) |
| `DROPIN3D_POINTOPS_MODULE` | 모듈 경로 | pointops2 CUDA 모듈 경로를 직접 지정 |
| `DROPIN3D_CLUSTER_BACKEND` | `cuml` / `sklearn` | 클러스터링 백엔드 |
| `DROPIN3D_SSIM_BACKEND` | `torch` | PyTorch SSIM을 강제로 사용 |
| `DROPIN3D_CHUNKED_RASTER` | `1` / `0` / `force` | MPS 3채널 분할 렌더링 (`force` = 모든 device, 테스트용) |

## mpsplat과 gsplat 1.5.3의 차이
mpsplat의 래스터라이저는 gsplat 1.5.3 CUDA 커널과 합성 규칙이 다릅니다: alpha 상한 0.99(CUDA는 0.999), alpha < 1/255 건너뛰기 없음, T ≤ 1e-4 조기 종료 없음.
[`third_party/mpsplat-patches/`](third_party/mpsplat-patches/)의 patch를 적용하면 Metal 커널과 PyTorch 경로가 모두 gsplat 1.5.3과 같은 규칙을 따릅니다.
Mac에서 [`tools/check_mpsplat_parity.py`](tools/check_mpsplat_parity.py)로 두 경로를 비교할 수 있습니다.

## 작업 계획
남은 마일스톤(M2~M8)의 목표, 작업 순서, 통과 기준은 [`docs/ROADMAP.md`](docs/ROADMAP.md)에 있습니다.

## 통합 예시
- [4DGS360](https://github.com/Jaewon040/4DGS360): [`integrations/4dgs360/`](integrations/4dgs360/), 상세 문서는 [`docs/4dgs360.md`](docs/4dgs360.md)

## 테스트
GPU 없이 CPU에서 실행합니다. 각 대체 구현을 기준 구현과 비교합니다.
```bash
pip install -e ".[test]"
```
```bash
pytest
```

## 라이선스
Apache-2.0. 이 레포는 직접 작성한 코드만 포함합니다. gsplat, mpsplat, pytorch3d, pointops2, cuML, fused-ssim은 각자의 라이선스를 따르며 이 레포에 포함되어 있지 않습니다.
