# 작업 계획: M2 ~ M8

이 문서는 dropin3d로 4DGS360을 Mac(MPS)과 V100에서 "성능 손상 없이" 돌리기 위해 남은 작업을 정리한다.
마일스톤마다 목표, 선행 조건, 작업 순서, 통과 기준, 산출물을 적는다.

- 기준 버전: 4DGS360 `099cec4`, mpsplat `94c8949`, dropin3d `main`
- 상태 표시: ✅ 완료, 🟡 코드 완료(실기기 확인 필요), ⏳ 남음
- 통과 기준의 수치는 **제안값**이다. 첫 측정 결과를 보고 조정한다.

## 0. 한눈에 보기

| # | 마일스톤 | 상태 | 필요한 장비 | 선행 조건 |
|---|---|---|---|---|
| M1 | 학습 코드에서 CUDA 의존 제거 | ✅ | — | — |
| M2 | gsplat → mpsplat 렌더링 래퍼 실기기 확인 | 🟡 | Mac | M4 |
| M3 | AnchorTAP3D(3D 트래킹) 실기기 확인 | 🟡 | Mac, CUDA(비교용) | — |
| M4 | mpsplat 셰이더 patch 실기기 확인 | 🟡 | Mac | — |
| M5 | Shape of Motion 전처리 MPS 포팅 | ⏳ | Mac, CUDA(비교용) | — |
| M6 | 단안 영상의 카메라 추정 경로 | ⏳ | CUDA(권장) 또는 Mac | M5 |
| M7 | CUDA 대 Mac 학습 결과 비교 | ⏳ | Mac, CUDA | M2, M4 |
| M8 | V100 대응 확인 | 🟡 | V100, A100급(비교용) | — |

### 진행 순서

```
M4 ──▶ M2 ──▶ M7          (Mac 학습 경로: 가장 먼저)
M3                         (Mac 전처리 경로, 독립)
M5 ──▶ M6                  (커스텀 데이터가 필요할 때만)
M8                         (V100, 독립)
```

1. **M4 → M2 → M7**: 공개된 iPhone360 데이터로 Mac 학습 경로를 끝까지 확인한다. 셰이더가 컴파일되지 않으면 이후 단계가 모두 막히므로 M4를 먼저 한다.
2. **M3, M8**: 장비가 준비되는 대로 따로 진행한다.
3. **M5 → M6**: 새 영상으로 학습할 때만 필요하다. iPhone/iPhone360 데이터만 쓴다면 건너뛴다.

---

## M2. gsplat → mpsplat 렌더링 래퍼 실기기 확인

**목표:** `dropin3d.raster.rasterization`이 Mac에서 mpsplat의 Metal 커널을 쓰고, PyTorch 경로와 같은 결과를 내는지 확인한다.

**선행 조건:** M4 (patch를 적용한 mpsplat 설치)

**작업 순서**
1. Mac 환경을 설치한다([`docs/4dgs360.md`](4dgs360.md) 2절).
2. 비교 스크립트의 [2]번 항목(16채널 + depth + background)을 확인한다.
   ```bash
   python tools/check_mpsplat_parity.py
   ```
3. 속도를 비교한다. 같은 렌더링을 두 경로로 실행하고 시간을 잰다.
   - 분할 렌더링 켬(Metal): `DROPIN3D_CHUNKED_RASTER=1`
   - 분할 렌더링 끔(PyTorch 경로): `DROPIN3D_CHUNKED_RASTER=0`
4. 4DGS360 학습을 짧게 실행한다(아래 M7의 학습 명령에서 `--num-epochs 5`).

**통과 기준**
- 비교 스크립트 [2]번의 상대 오차 ≤ 1e-4
- 분할 렌더링이 PyTorch 경로보다 빠름 (예상: 수십 배)
- 짧은 학습에서 loss가 줄어들고 NaN이 없음

**산출물:** 비교 스크립트 출력, 속도 측정표, 짧은 학습 로그

**위험과 대응**
- Metal 경로가 쓰이지 않으면(속도 차이가 없으면) mpsplat의 dispatch 조건(`tile_size=16`, `packed=False`)과 PyTorch 버전(`torch.mps.compile_shader` 지원 여부)을 확인한다.
- 학습 중 MPS에 없는 연산이 나오면 `PYTORCH_ENABLE_MPS_FALLBACK=1`로 우회하고, 해당 연산을 기록한다.

---

## M3. AnchorTAP3D(3D 트래킹) 실기기 확인

**목표:** AnchorTAP3D 추론이 Mac에서 돌아가고, 3D 트랙이 CUDA 결과와 일치하는지 확인한다.

**선행 조건:** 없음

**작업 순서**
1. 4DGS360에 [`integrations/4dgs360/`](../integrations/4dgs360/) patch를 적용한다.
2. TAPIP3D의 `third_party/`를 `preproc/AnchorTAP3D/`로 복사한다(cotracker 등). pointops2는 빌드하지 않아도 된다.
3. TAPIP3D checkpoint(`tapip3d_final.pth`)를 받는다.
4. iPhone360 장면 1개에서 추론을 실행한다.
   ```bash
   python inference.py --work_dir <장면 경로> --img_res 1 --checkpoint checkpoints/tapip3d_final.pth --iphone --use_2dtrack --track_vis_threshold 0.5 --filter_2dtrack_with_mask --prune_query 1000 --device auto --precision auto
   ```
5. 같은 장면을 CUDA(A100급, `--precision auto` = bf16)에서도 실행한다.
6. 두 결과의 `track3d_*.npy`를 비교한다.

**통과 기준**
- Mac에서 오류 없이 끝남
- 보이는 점의 3D 위치 차이 중앙값 ≤ 장면 크기의 0.5%
- visibility 판정 일치율 ≥ 99%
- 참고: Mac은 fp32, CUDA는 bf16이므로 비트 단위로 같을 수는 없다. 차이는 bf16 반올림 수준이어야 한다.

**산출물:** 비교 스크립트(`tools/compare_tracks.py`, 새로 작성), 비교 결과표

**위험과 대응**
- 공개된 iPhone360 데이터에 AnchorTAP3D 입력(BootsTAPIR 2D 트랙, 정렬된 깊이)이 들어 있는지 아직 확인하지 않았다. 없으면 M5가 먼저 필요하다.
- kNN 단계가 느리면 `dropin3d.knn.torch_knnquery`의 `chunk_size`를 조정한다.

---

## M4. mpsplat 셰이더 patch 실기기 확인

**목표:** gsplat 1.5.3 합성 규칙으로 고친 Metal 셰이더가 컴파일되고, PyTorch 경로와 같은 결과를 내는지 확인한다.

**선행 조건:** 없음 (가장 먼저 할 일)

**작업 순서**
1. mpsplat에 patch를 적용하고 설치한다([`third_party/mpsplat-patches/README.md`](../third_party/mpsplat-patches/README.md)).
2. 비교 스크립트를 실행한다.
   ```bash
   python tools/check_mpsplat_parity.py
   ```
3. 첫 줄에 `parity patch applied: True`가 나오는지 확인한다.
4. mpsplat 자체 테스트(`tests/test_mps_*`)도 실행해서 다른 기능이 깨지지 않았는지 확인한다.

**통과 기준**
- 셰이더가 오류 없이 컴파일됨
- [1]번 항목(3채널, Metal 대 PyTorch)의 색, alpha, gradient 상대 오차 ≤ 1e-4
- mpsplat 자체 테스트가 patch 전과 같은 결과(새로 실패하는 테스트 없음)

**산출물:** 스크립트 출력, 필요하면 patch 수정

**위험과 대응**
- 셰이더 컴파일 오류: 오류 메시지를 받아 patch를 고친다.
- mpsplat 테스트 중 alpha 상한 0.99를 전제로 한 테스트가 실패할 수 있다. 이 경우 테스트 기대값만 1.5.3 규칙에 맞게 고친다.

---

## M5. Shape of Motion 전처리 MPS 포팅

**목표:** 새 영상에 필요한 전처리(마스크, 단안 깊이, 2D 트랙)를 Mac에서 실행한다.

**선행 조건:** 없음. 커스텀 데이터가 필요할 때만 진행한다.

**작업 순서**

| 단계 | 도구 | 할 일 |
|---|---|---|
| 1. 마스크 | SAM + XMem (Shape of Motion `preproc`) | device를 `mps`로 바꾸는 patch. SAM2를 쓴다면 CUDA 확장(구멍 채우기)은 건너뛰고 OpenCV connected components로 대신함 |
| 2. 단안 깊이 | Video-Depth-Anything | device를 `mps`로 바꾸는 patch, 16-bit PNG export 스크립트 작성 |
| 3. 깊이 정렬 | `preproc/align_monodepth_with_lidar.py` | 변경 없음. 출력 이름을 `depth_anything_lidar`로 짓지 않음(직접 깊이로 해석되기 때문) |
| 4. 2D 트랙 | BootsTAPIR (tapnet PyTorch) | device를 `mps`로 바꾸는 patch. 5-D `grid_sample`에는 torch ≥ 2.9가 필요함. 다른 tracker로 바꾸지 않음(visibility 공식이 TAPIR 출력을 전제함) |

각 patch는 `integrations/<도구 이름>/`에 patch 파일과 README로 넣는다.

**통과 기준** (같은 영상을 CUDA와 Mac에서 처리해서 비교)
- 마스크 IoU ≥ 0.99
- 깊이 상대 오차 중앙값 ≤ 1%
- 2D 트랙 위치 차이 중앙값 ≤ 0.5 px, occlusion 판정 일치율 ≥ 99%

**산출물:** `integrations/shape-of-motion/`, `integrations/video-depth-anything/`, `integrations/tapnet/`

**위험과 대응**
- SAM2는 MPS 지원을 "예비 단계이며 수치가 다를 수 있음"이라고 밝히고 있다. 원래의 SAM + XMem을 유지한다.

---

## M6. 단안 영상의 카메라 추정 경로

**목표:** 카메라를 모르는 영상에서 카메라와 깊이를 얻는 방법을 정한다.

**선행 조건:** M5. iPhone/iPhone360 데이터에는 필요 없다(카메라를 `camera/*.json`에서 읽음).

**선택지**

| 방법 | 장비 | 결과 영향 | 작업량 |
|---|---|---|---|
| A. MegaSaM을 CUDA에서 한 번 실행하고 결과만 가져옴 (**권장**) | CUDA | 없음 | 작음 |
| B. Depth Anything 3, π³, VGGT 같은 순수 PyTorch 모델로 대체 | Mac | **큼**: 카메라와 깊이가 달라져 최종 결과도 달라짐 | 중간 |
| C. MegaSaM 전체를 포팅 (DROID `droid_backends`) | Mac | 작음 | 큼. 알려진 Mac 포팅이 없음 |

**작업 순서 (A)**
1. CUDA 장비에서 MegaSaM을 실행한다.
2. 결과를 DyCheck 형식(`camera/*.json`과 깊이)으로 변환하는 스크립트를 작성한다(`tools/megasam_to_dycheck.py`).
3. 변환한 데이터로 iPhone 경로(`--iphone`)를 사용한다.

**알려진 문제:** AnchorTAP3D `inference.py`의 `droid_recon` 로더(400~417행)는 프레임 *i*에 `kf_tstamps[i]`의 pose를 준다. 모든 프레임이 keyframe이 아니면 pose가 틀린다. A 방법은 이 로더를 쓰지 않으므로 영향이 없다.

**통과 기준 (A):** 변환한 카메라로 다시 투영한 점이 원래 MegaSaM 결과와 0.1 px 이내로 일치

**산출물:** `tools/megasam_to_dycheck.py`, 사용 문서

**위험과 대응**
- lietorch를 Mac에서 쓰게 되면 반드시 CPU에서 실행한다. MPS 텐서를 넣으면 오류 없이 입력을 그대로 돌려준다.

---

## M7. CUDA 대 Mac 학습 결과 비교

**목표:** Mac 학습 결과가 CUDA 결과와 통계적으로 구별되지 않는지 확인한다. 이것이 "성능 손상 없음"의 최종 판정이다.

**선행 조건:** M2, M4

**준비 작업**
1. **seed 옵션 추가**: 지금 `run_training.py`는 seed를 42로 고정한다(`set_seed(42)`). 환경변수(예: `FOURDGS_SEED`)로 바꿀 수 있게 4DGS360 patch를 수정한다.
2. 비교 스크립트를 작성한다(`tools/compare_runs.py`). 여러 실행의 지표를 모아 평균, 표준편차, 차이를 표로 만든다.

**작업 순서**
1. 장면: iPhone360에서 2~3개(예: jacket과 움직임이 큰 장면 1개)
2. 각 장면을 CUDA(gsplat 1.5.3)와 Mac(patch한 mpsplat)에서 seed 3개로 학습한다.
   ```bash
   python run_training.py --work-dir ./outputs/iphone360/jacket --port 8888 data:iphone360 --data.data-dir ./data/iphone360/jacket/ --data.camera_type original
   ```
3. 렌더링하고 지표를 계산한다.
   ```bash
   python run_evaluation.py --work-dir outputs/iphone360/jacket/ --ckpt-path outputs/iphone360/jacket/checkpoints/last.ckpt data:iphone360 --data.data-dir ./data/iphone360/jacket
   ```
   ```bash
   PYTHONPATH="." python scripts/evaluate_iphone360.py --data_dir ./data/iphone360/jacket --result_dir ./outputs/iphone360/jacket
   ```
4. 학습 시간과 최대 메모리도 기록한다.

**통과 기준**
- 장면마다 Mac 평균과 CUDA 평균의 차이 ≤ CUDA seed 표준편차의 2배 (PSNR, SSIM, LPIPS 각각)
- 기준을 넘으면 원인을 하나씩 분리한다. 순서: KMeans 백엔드(`DROPIN3D_CLUSTER_BACKEND`로 CUDA에서도 scikit-learn을 써서 비교) → 렌더러 → SSIM 백엔드

**산출물:** 결과표(`docs/results/m7.md`), `tools/compare_runs.py`, seed 옵션 patch

**위험과 대응**
- KMeans 초기화 차이(scikit-learn 대 cuML)가 결과를 흔들 수 있다. CUDA에서 `DROPIN3D_CLUSTER_BACKEND=sklearn`으로도 돌려서 이 요인만 따로 측정한다.

---

## M8. V100 대응 확인

**목표:** V100(sm_70)에서 4DGS360 전체 파이프라인이 돌아가고, A100급 결과와 같은지 확인한다.

**선행 조건:** 없음

**작업 순서**
1. CUDA 확장을 sm_70용으로 빌드한다.
   ```bash
   export TORCH_CUDA_ARCH_LIST="7.0"
   ```
   ```bash
   CUDA_ARCHITECTURES=70 pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation --force-reinstall
   ```
   gsplat과 pointops2도 같은 설정으로 다시 빌드하고, 이전 JIT 캐시(`~/.cache/torch_extensions`)를 지운다.
2. 바이너리에 sm_70이 들어 있는지 확인한다.
   ```bash
   cuobjdump --list-elf <확장 모듈 .so 경로>
   ```
3. AnchorTAP3D를 `--precision auto`로 실행한다. V100에서는 fp32가 선택되는지 로그에서 확인한다.
4. 학습의 최대 메모리를 잰다. V100 16GB에서 부족하면 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`부터 적용한다. batch size와 densification 설정은 바꾸지 않는다(결과가 달라짐).
5. M7과 같은 방법으로 A100급 결과와 비교한다.

**통과 기준**
- `no kernel image is available` 오류 없음
- M7과 같은 기준으로 A100급 결과와 차이 없음

**산출물:** V100 설치 문서(`docs/v100.md`), 결과표

**위험과 대응**
- 16GB에서 OOM이 나고 할당자 설정으로도 해결되지 않으면, 카메라별로 나눠 backward하는 방법을 검토한다. 이때 `means2d` gradient 누적과 `batch_size` 배율이 원래와 같은지 수치로 확인해야 한다.

---

## 결과 보고 형식

각 마일스톤이 끝나면 다음을 PR로 올린다.
- 실행한 장비, OS, PyTorch 버전
- 실행한 명령과 출력(또는 로그 파일)
- 통과 기준별 측정값과 판정
- 기준을 넘은 항목이 있으면 원인 분석과 다음 조치
