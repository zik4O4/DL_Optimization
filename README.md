# Deep Learning Model Optimization

MNIST experiments demonstrating magnitude pruning, post-training quantization, quantization-aware training and TensorFlow Lite export.

## Problem and solution

The notebook explores reducing a handwritten-digit model's deployment size while retaining classification performance. A dense network (`Flatten → Dense(100, ReLU) → Dense(10, Softmax)`) is trained, pruned toward 50% sparsity, and converted to TFLite before and after quantization.

**Technologies:** Python, TensorFlow/Keras, TensorFlow Model Optimization, NumPy, Matplotlib and Jupyter.

## Repository contents

| Path | Contents |
|---|---|
| `prg/DL opt.ipynb` | Training, pruning, quantization and export experiment |
| `prg/tflite_model.tflite` | Existing baseline TFLite artifact |
| `prg/tflite_quant_model.tflite` | Existing post-training quantized artifact |
| `prg/tflite_qaware_model.tflite` | Existing quantization-aware artifact |
| `vedio.mp4` | Existing project video |

## Setup and usage

```bash
git clone https://github.com/zik4O4/DL_Optimization.git
cd DL_Optimization
python -m venv .venv
source .venv/bin/activate
python -m pip install tensorflow tensorflow-model-optimization numpy matplotlib jupyter
jupyter notebook "prg/DL opt.ipynb"
```

The command lists the notebook's dependencies, not a validated lockfile. Use mutually compatible TensorFlow, Keras and TensorFlow Model Optimization releases; newer combinations may require environment adjustments. MNIST is downloaded on first use. Run cells in order and check the notebook working directory before exporting, because filenames are relative and may replace local artifacts.

## Existing results

| Evaluated Keras model | Saved MNIST test accuracy |
|---|---:|
| Baseline | 0.9747 |
| Pruned | 0.9790 |
| Quantization-aware | 0.9758 |

| Existing TFLite file | Size in bytes |
|---|---:|
| Baseline | 319,936 |
| Post-training quantized | 84,816 |
| Quantization-aware | 82,712 |

These values come from saved notebook outputs and committed artifacts. Accuracy is measured on Keras models, not independently on each exported TFLite file. No new training or runtime benchmark is claimed.

## Limitations

- Post-training conversion uses `Optimize.DEFAULT` without a representative calibration dataset; this is not demonstrated fully integer end-to-end inference.
- The notebook uses the test set as validation data during QAT, so reported final evaluation is not an untouched holdout protocol.
- Inference latency and on-device performance are not measured.
- The notebook checkpoint, video and original outputs are preserved.

## License

No repository-level license has been selected. Framework, dataset and third-party asset terms apply separately.
