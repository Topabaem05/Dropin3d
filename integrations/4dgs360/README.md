# 4DGS360 통합

[4DGS360](https://github.com/Jaewon040/4DGS360)(커밋 `099cec4`)을 dropin3d로 Mac(MPS), CPU, V100에서 돌리기 위한 patch입니다.
4DGS360 코드는 이 레포에 포함되어 있지 않습니다. 원본 레포를 받은 뒤 patch를 적용합니다.

## 적용 방법

1. 4DGS360을 받고 기준 커밋으로 맞춥니다.
   ```bash
   git clone https://github.com/Jaewon040/4DGS360 && cd 4DGS360 && git checkout 099cec4
   ```
2. patch를 적용합니다.
   ```bash
   git apply /path/to/Dropin3d/integrations/4dgs360/4dgs360-dropin3d.patch
   ```
3. Mac 환경을 설치합니다.
   ```bash
   pip install -r requirements-mac.txt
   ```
4. 통합 테스트를 실행합니다.
   ```bash
   PYTHONPATH=. pytest tests/test_dropin3d_integration.py
   ```

## patch 내용
- `pytorch3d`, `cuML`, `CuPy`, `fused_ssim`, `gsplat` import를 dropin3d로 교체
- 하드코딩된 `"cuda"` device를 `dropin3d.get_device()`로 교체 (학습, 렌더링, 평가 스크립트)
- AnchorTAP3D: pointops2 kNN 교체, `--device auto`, `--precision auto` 추가, fp32 보호 구간을 모든 device에 적용
- `requirements-mac.txt`, `tests/test_dropin3d_integration.py` 추가

자세한 설명과 남은 작업은 [`docs/4dgs360.md`](../../docs/4dgs360.md)를 보세요.
