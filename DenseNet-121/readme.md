
# DenseNet Benchmarking Project

This project benchmarks DenseNet on a custom dataset using PyTorch. It evaluates multiple model variants:

- Baseline FP32
- AMP (Automatic Mixed Precision / FP16)
- TorchScript
- Dynamic Quantization (CPU)

Metrics measured include latency, throughput, VRAM usage, CPU utilization, and accuracy.



## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
````

**Sample `requirements.txt`:**

torch>=2.2
torchvision
numpy
pandas
tqdm
tensorboard


## Dataset

Organize your dataset like this:

```
datasets/train/class1/
datasets/train/class2/
datasets/val/class1/
datasets/val/class2/
```



## Running the Benchmark

1. Place the dataset in `/kaggle/working/datasets_folder` (or adjust paths in `main.py`).
2. Run the script:


python src/main.py

This will:

* Train and benchmark DenseNet variants
* Measure performance metrics
* Save results in CSV/JSON
* Save models and logs in `outputs/`



## Outputs

* CSV/JSON Results: `outputs/benchmark_results.csv`, `outputs/benchmark_results.json`
* Saved Models: `outputs/models/`
* TensorBoard Logs: `outputs/tensorboard_logs/`
* Profiler Traces: `outputs/profiler/`

---

## Notes

* AMP requires GPU. CPU will skip AMP variant.
* Dynamic quantization runs on CPU only.
* TorchScript variant may fail if unsupported ops exist.

---
