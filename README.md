# Real-time image enhancement: latency vs quality across backends

Short self-directed engineering study. A small residual U-Net enhances dark, noisy 640×480 frames; the question is whether it fits a **30 fps budget (33.3 ms per frame)** and what speed costs in quality.

- Three model widths, trained on synthetic low-light + noise degradation
- Per-frame latency at batch size 1: PyTorch CPU, PyTorch GPU (Apple MPS / CUDA) in FP32 and FP16, ONNX Runtime
- Correctness checks for every backend (max output difference vs PyTorch FP32) and PSNR for FP32 vs FP16
- End-to-end pipeline profile: camera frame → preprocessing → transfer → inference → postprocessing

## Results
_To be filled in after the run._

## Run it
```bash
conda activate emg
pip install -r requirements.txt
```
Open `notebooks/realtime_enhancement_benchmark.ipynb`, select the kernel, set `QUICK=True` for a smoke test, then Run All.

## Author
Jahid Hasan Jony
