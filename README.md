# RAJ DL ESE - Deep Learning Experiments Lab

## Project Overview
A comprehensive collection of 10 deep learning experiments implementing fundamental neural network architectures and techniques: MLPs, CNNs, Transfer Learning, RNNs, LSTMs, GRUs, Object Detection, GANs, and performance analysis. All experiments use PyTorch and are optimized for reproducibility and educational clarity.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Setup](#environment-setup)
3. [Dataset Information](#dataset-information)
4. [Running Experiments](#running-experiments)
5. [Offline Mode & Library Downloading](#offline-mode--library-downloading)
6. [Expected Outputs](#expected-outputs)
7. [Troubleshooting](#troubleshooting)
8. [Documentation](#documentation)

---

## Quick Start

### Prerequisites
- **Python**: 3.9 or higher (tested on 3.11)
- **OS**: Linux, macOS, or Windows
- **Storage**: ~2 GB free space (for models and data)
- **RAM**: 4 GB minimum (8 GB recommended)
- **GPU** (optional): NVIDIA GPU with CUDA support for faster training

### 30-Second Setup

```bash
# 1. Navigate to project directory
cd /home/raj_99/Projects/Raj_DL_ESE

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install torch torchvision torchaudio
pip install matplotlib numpy
pip install opencv-python Pillow
pip install ultralytics  # For YOLO

# 4. Run your first experiment
python mlp.py
```

---

## Environment Setup

### Step 1: Create Virtual Environment

We recommend using Python virtual environments to isolate dependencies.

**Using venv (Standard):**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate  # Windows
```

**Using Conda (Alternative):**
```bash
conda create -n dl_env python=3.11
conda activate dl_env
```

### Step 2: Install PyTorch

Install the appropriate PyTorch version for your system:

**CPU-only (Recommended for beginners):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**GPU (NVIDIA CUDA 11.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**GPU (NVIDIA CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Visit [pytorch.org](https://pytorch.org/get-started/locally/) to find the correct command.

### Step 3: Install Additional Dependencies

```bash
pip install numpy matplotlib pillow opencv-python
pip install ultralytics  # For YOLO object detection
pip install tensorboard scikit-learn  # Optional
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'Torchvision: {torchvision.__version__}')"
python -c "from ultralytics import YOLO; print('YOLO: OK')"
```

---

## Dataset Information

### Overview

This suite automatically downloads datasets on first run if internet is available. All data is cached locally for offline access.

### Datasets Used

#### 1. MNIST (Images)
- **Size**: ~11 MB
- **Resolution**: 28x28 grayscale  
- **Classes**: 10 (digits 0-9)
- **Source**: torchvision.datasets.MNIST
- **Cache**: `~/.cache/torch/datasets/MNIST/`
- **Used in**: EXP 1, 2, 4, 5, 7, 8, 9, 10

#### 2. MobileNetV2 (Pretrained Model)
- **Size**: ~13 MB
- **Type**: CNN pretrained on ImageNet
- **Classes**: 1,000 (ImageNet)
- **Source**: torchvision.models
- **Cache**: `~/.cache/torch/hub/`
- **Used in**: EXP 3, 4

#### 3. YOLOv8n (Object Detection)
- **Size**: ~100 MB
- **Type**: Object detector pretrained on COCO
- **Classes**: 80 (COCO)
- **Source**: ultralytics
- **Cache**: `~/.cache/ultralytics/`
- **Used in**: EXP 6

#### 4. Professional Text Corpus (Custom)
- **Size**: ~1 KB
- **Format**: Plain text
- **Content**: Machine learning evaluation discourse
- **Source**: Local `data.txt`
- **Used in**: EXP 7, 8, 9

---

## Offline Mode & Library Downloading

### First-Time Setup (Internet Required)

When running experiments for the first time:

1. **MNIST downloads automatically** (~11 MB, ~30 seconds)
    - To: `~/.cache/torch/datasets/MNIST/`
    - In: EXP 1, 2, 4, 5, 7, 8, 9, 10

2. **MobileNetV2 downloads automatically** (~13 MB, ~20 seconds)
    - To: `~/.cache/torch/hub/`
    - In: EXP 3, 4

3. **YOLOv8n downloads automatically** (~100 MB, ~2 minutes)
    - To: `~/.cache/ultralytics/`
    - In: EXP 6

4. **Text corpus** (no download, local file)
    - Already in: `data.txt`
    - In: EXP 7, 8, 9

### Offline Mode (No Internet)

After first-time setup, experiments run completely offline:

```bash
# All of these work without internet (after first run)
python mlp.py      # Uses cached MNIST
python transfer_learning.py  # Uses cached MobileNetV2
python object_detection.py   # Uses cached YOLOv8n
python rnn.py      # Uses local data.txt
```

**If data not cached:**
- Connect to internet
- Run experiment to trigger download
- Disconnect internet for subsequent runs

### Manual Dataset Download

To explicitly download datasets before running:

```bash
# Download MNIST
python -c "
from torchvision.datasets import MNIST
MNIST(root='~/.cache/torch/datasets/', download=True)
print('MNIST downloaded')
"

# Download MobileNetV2
python -c "
import torchvision.models as models
models.mobilenet_v2(pretrained=True)
print('MobileNetV2 downloaded')
"

# Download YOLOv8n
python -c "
from ultralytics import YOLO
YOLO('yolov8n.pt')
print('YOLOv8n downloaded')
"
```

### Cache Configuration

To customize cache locations:

```bash
export TORCH_HOME=/custom/path/torch
export ULTRALYTICS_HOME=/custom/path/ultralytics
python mlp.py
```

---

## Running Experiments

### Quick Reference Table

| Exp | File | Title | Time | Output |
|-----|------|-------|------|--------|
| 1 | `mlp.py` | MLP Analysis | ~20s | PNG plot |
| 2 | `cnn.py` | CNN Analysis | ~45s | PNG plot |
| 3 | `transfer_learning.py` | Transfer Learning | ~10s | PNG plot |
| 4 | `cnn_vs_transfer.py` | CNN vs Transfer | ~60s | PNG plot |
| 5 | `performance_analysis.py` | Hyperparameters | ~80s | PNG plot |
| 6 | `object_detection.py` | Object Detection | ~90s | PNG plots |
| 7 | `rnn.py` | RNN | ~15s | PNG plot |
| 8 | `lstm.py` | LSTM | ~25s | PNG plots |
| 9 | `gru.py` | GRU | ~20s | PNG plots |
| 10 | `gan.py` | GAN | ~120s | PNG plots |

### Running Individual Experiments

**EXP 1: Multilayer Perceptron**
```bash
python mlp.py
# Output: results/exp1_mlp.png, results/exp1_depth_analysis.png
# Purpose: Activation functions, depth, convergence analysis
```

**EXP 2: Convolutional Neural Network**
```bash
python cnn.py
# Output: results/exp2_cnn.png
# Purpose: CNN vs FC, pooling, feature maps, kernels
```

**EXP 3: Transfer Learning**
```bash
python transfer_learning.py
# Output: results/exp3_transfer_learning.png
# Purpose: Frozen features, fine-tuning, efficiency analysis
```

**EXP 4: CNN vs Transfer Comparison**
```bash
python cnn_vs_transfer.py
# Output: results/exp4_cnn_vs_transfer.png
# Purpose: Accuracy, timing, dataset size impact
```

**EXP 5: Hyperparameter Analysis**
```bash
python performance_analysis.py
# Output: results/exp5_performance_analysis.png
# Purpose: Learning rate, epochs, train/val curves, configs
```

**EXP 6: Object Detection**
```bash
python object_detection.py
# Output: results/exp6_object_detection.png
# Purpose: YOLO detection, confidence, multi-object, image quality
```

**EXP 7: Recurrent Neural Network**
```bash
python rnn.py
# Output: results/exp7_rnn.png
# Purpose: Sequence prediction, length effect, long-term dependencies
```

**EXP 8: LSTM**
```bash
python lstm.py
# Output: results/exp8_lstm.png
# Purpose: LSTM vs RNN, convergence, units, memory mechanisms
```

**EXP 9: Gated Recurrent Unit**
```bash
python gru.py
# Output: results/exp9_gru.png
# Purpose: GRU vs LSTM, speed, efficiency, real-time suitability
```

**EXP 10: Generative Adversarial Network**
```bash
python gan.py
# Output: results/exp10_generated_final.png, exp10_progression.png, exp10_gan_training.png
# Purpose: Generation, training dynamics, sample quality
```

### Run All Experiments

**Sequentially (~9 minutes total):**
```bash
for script in mlp.py cnn.py transfer_learning.py cnn_vs_transfer.py \
                  performance_analysis.py object_detection.py \
                  rnn.py lstm.py gru.py gan.py; do
     python $script
done
```

**Parallel (~4 minutes):**
```bash
for script in mlp.py cnn.py transfer_learning.py cnn_vs_transfer.py \
                  performance_analysis.py object_detection.py \
                  rnn.py lstm.py gru.py gan.py; do
     python $script &
done
wait
```

---

## Expected Outputs

All plots saved to `results/` directory. See terminal output for metrics.

### Output Files

```
results/
├── exp1_mlp.png                    # MLP analysis
├── exp1_depth_analysis.png         # Depth effects
├── exp2_cnn.png                    # CNN analysis
├── exp3_transfer_learning.png      # Transfer learning
├── exp4_cnn_vs_transfer.png        # Comparison
├── exp5_performance_analysis.png   # Hyperparameters
├── exp6_object_detection.png       # YOLO detection
├── exp7_rnn.png                    # RNN analysis
├── exp8_lstm.png                   # LSTM analysis
├── exp9_gru.png                    # GRU analysis
├── exp10_generated_final.png       # GAN samples
├── exp10_progression.png           # GAN progression
├── exp10_gan_training.png          # GAN curves
├── VIVA_MASTER_NOTES.txt           # Quick reference
└── TECHNICAL_EXPLANATION.txt       # Detailed explanations
```

---

## Troubleshooting

### PyTorch not found
```bash
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Dataset download error
```bash
# Connect to internet, run any experiment
python mlp.py
# Then disconnect and run again
```

### Out of memory
```bash
# Edit experiment file, reduce batch_size from 32 to 16
# Or switch to CPU: export CUDA_VISIBLE_DEVICES=""
```

### YOLO import error
```bash
pip install ultralytics
```

### results/ directory not found
```bash
mkdir -p results
python mlp.py
```

---

## Documentation

### Quick Reference
**File:** `results/VIVA_MASTER_NOTES.txt`
- Summary of all 10 experiments
- Key metrics and results
- Viva Q&A format

### Detailed Explanations
**File:** `results/TECHNICAL_EXPLANATION.txt`
- 40 sub-experiments (4 per experiment)
- Implementation details
- Expected results with values
- Viva answers and trade-offs

### Recommended Learning Path
1. EXP 1 (MLP) - Activation, depth, convergence
2. EXP 2 (CNN) - Convolution, pooling, features
3. EXP 5 (Performance) - Hyperparameter tuning
4. EXP 3 (Transfer) - Practical deep learning
5. EXP 7-9 (RNN/LSTM/GRU) - Sequence models
6. EXP 10 (GAN) - Generative models
7. EXP 4, 6 (Comparison, Detection) - Advanced topics

---

## File Structure

```
Raj_DL_ESE/
├── README.md                       # This file
├── main.py
├── pyproject.toml
├── data.txt                        # Text corpus
├── mlp.py                          # EXP 1
├── cnn.py                          # EXP 2
├── transfer_learning.py            # EXP 3
├── cnn_vs_transfer.py              # EXP 4
├── performance_analysis.py         # EXP 5
├── object_detection.py             # EXP 6
├── rnn.py                          # EXP 7
├── lstm.py                         # EXP 8
├── gru.py                          # EXP 9
├── gan.py                          # EXP 10
└── results/                        # Output directory
```

---

## References

- PyTorch: [pytorch.org](https://pytorch.org)
- MNIST: LeCun et al. (1998)
- MobileNetV2: Sandler et al. (2018)
- YOLO: Redmon & Farhadi (2018)
- LSTM: Hochreiter & Schmidhuber (1997)
- GRU: Cho et al. (2014)
- GAN: Goodfellow et al. (2014)

---

**Status**: Production Ready ✅  
**Last Updated**: 2024  
**Python**: 3.9+  
**PyTorch**: 2.0+
| torchvision | Vision datasets (MNIST, pretrained models) | First dataset load |
| torchaudio | Audio processing (if needed) | First import |
| numpy | Numerical computation | First import |
| matplotlib | Visualization & plot generation | First import |
| opencv-python | Image processing | First import |
| ultralytics | YOLO object detection | First model load |
| scikit-learn | ML utilities | First import |
| pillow | Image I/O | First import |

---

## Running Individual Experiments

### EXP 1: MLP (Multilayer Perceptron)
```bash
uv run mlp.py
```
**Expected output:**
- Console: Activation function accuracies, depth convergence metrics
- Plots: `results/exp1_mlp.png`, `results/exp1_depth_analysis.png`
**Runtime:** ~30 seconds | **Dataset:** MNIST (5000 train, 1000 test)

### EXP 2: CNN (Convolutional Neural Network)
```bash
uv run cnn.py
```
**Expected output:**
- Console: CNN vs FC comparison, pooling analysis, kernel effect metrics
- Plot: `results/exp2_cnn.png`
**Runtime:** ~45 seconds | **Dataset:** MNIST (5000 train, 1000 test)

### EXP 3: Transfer Learning
```bash
uv run transfer_learning.py
```
**Expected output:**
- Console: Frozen vs fine-tuned performance, efficiency metrics
- Plot: `results/exp3_transfer_learning.png`
**Runtime:** ~60 seconds | **Model:** MobileNetV2 (pretrained, ~13 MB download on first run)
**Note:** Uses synthetic image data (no actual dataset download required after MobileNetV2)

### EXP 4: CNN vs Transfer Learning
```bash
uv run cnn_vs_transfer.py
```
**Expected output:**
- Console: Accuracy & timing comparison across dataset sizes
- Plot: `results/exp4_cnn_vs_transfer.png`
**Runtime:** ~120 seconds | **Dataset:** MNIST (scaled to 500-5000 samples)

### EXP 5: Performance Analysis
```bash
uv run performance_analysis.py
```
**Expected output:**
- Console: Hyperparameter tuning results
- Plot: `results/exp5_performance_analysis.png`
**Runtime:** ~90 seconds | **Dataset:** MNIST (5000 train, 1000 test)

### EXP 6: Object Detection (YOLO)
```bash
uv run object_detection.py
```
**Expected output:**
- Console: Detection counts, confidence statistics, image quality impact
- Plot: `results/exp6_object_detection.png`
**Runtime:** ~45 seconds | **Model:** YOLOv8n (~100 MB download on first run)
**Fallback:** If YOLO unavailable, uses mock detection data

### EXP 7: Simple RNN
```bash
uv run rnn.py
```
**Expected output:**
- Console: Character-level prediction accuracy, sequence-length impact
- Plot: `results/exp7_rnn.png`
**Runtime:** ~50 seconds | **Dataset:** Professional text corpus from `data.txt` (in-memory)

### EXP 8: LSTM
```bash
uv run lstm.py
```
**Expected output:**
- Console: LSTM vs RNN comparison, convergence metrics, memory analysis
- Plot: `results/exp8_lstm.png`
**Runtime:** ~60 seconds | **Dataset:** Professional text corpus from `data.txt`

### EXP 9: GRU
```bash
uv run gru.py
```
**Expected output:**
- Console: GRU vs LSTM speed/accuracy trade-off, efficiency configs, real-time suitability
- Plot: `results/exp9_gru.png`
**Runtime:** ~70 seconds | **Dataset:** Professional text corpus from `data.txt`

### EXP 10: GAN (Generative Adversarial Network)
```bash
uv run gan.py
```
**Expected output:**
- Console: Training stability metrics, generator/discriminator performance
- Plots: `results/exp10_generated_final.png`, `results/exp10_progression.png`, `results/exp10_gan_training.png`
**Runtime:** ~80 seconds | **Dataset:** MNIST (1000 train samples)

---

## Run All Experiments (Sequentially)
```bash
for exp in mlp cnn transfer_learning cnn_vs_transfer performance_analysis \
           object_detection rnn lstm gru gan; do
    echo "Running EXP: $exp"
    uv run ${exp}.py
done
```
**Total runtime:** ~8-10 minutes

---

## Output Organization

All results are saved in the `results/` folder:

```
results/
├── VIVA_MASTER_NOTES.txt          # Quick reference for viva preparation
├── TECHNICAL_EXPLANATION.txt      # Detailed sub-experiment explanations
├── exp1_mlp.png                   # MLP activation & depth analysis
├── exp1_depth_analysis.png        # Additional depth visualization
├── exp2_cnn.png                   # CNN architecture comparison
├── exp3_transfer_learning.png     # Transfer learning performance
├── exp4_cnn_vs_transfer.png       # CNN vs Transfer comparison
├── exp5_performance_analysis.png  # Hyperparameter tuning results
├── exp6_object_detection.png      # YOLO detection analysis
├── exp7_rnn.png                   # RNN sequence prediction
├── exp8_lstm.png                  # LSTM vs RNN comparison
├── exp9_gru.png                   # GRU efficiency analysis
├── exp10_generated_final.png      # GAN final samples
├── exp10_progression.png          # GAN training progression
└── exp10_gan_training.png         # GAN loss curves
```

---

## Dataset Details & Offline Mode

### MNIST Dataset
- **Source:** torchvision.datasets.MNIST
- **Size:** ~11 MB (train: 9.5 MB, test: 1.6 MB)
- **First-run behavior:** Automatically downloads to `./data/` folder
- **Offline use:** Once downloaded, all subsequent runs work offline
- **Subset used:** 5000 train samples, 1000 test samples (for speed)

### Professional Text Corpus
- **Source:** Local file `data.txt` (generated/updated per session)
- **Content:** Professional discussion of ML evaluation, not toy repetitive sentences
- **Size:** ~1 KB (in-memory, no large file I/O)
- **Used in:** RNN (EXP 7), LSTM (EXP 8), GRU (EXP 9)
- **Offline:** 100% local, no download required

### Pretrained Models
| Model | Size | First-run Download | Location |
|-------|------|-------------------|----------|
| MobileNetV2 | ~13 MB | Yes (from torchvision) | `~/.cache/torch/` |
| YOLOv8n | ~100 MB | Yes (from ultralytics) | `./yolov8n.pt` or cache |

### Synthetic Data
- **Transfer Learning (EXP 3):** torch.randn() generated images (no external download)
- **GAN (EXP 10):** MNIST subsets (already covered above)

---

## Environment Variables (Optional)

Set cache directories for faster offline access:
```bash
export TORCH_HOME=/path/to/torch/cache
export ULTRALYTICS_HOME=/path/to/yolo/cache
```

Default locations will be used if not set.

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution:** Ensure virtual environment is activated
```bash
source .venv/bin/activate
uv pip list  # verify torch is installed
```

### Issue: "MNIST download fails / no internet"
**Solution:** If MNIST was previously downloaded, delete cache and pre-download offline
```bash
rm -rf ./data/  # Remove old MNIST
# First run with internet to cache MNIST
uv run mlp.py  # This will download MNIST
# Subsequent runs: no internet needed
```

### Issue: "YOLO model not found"
**Solution:** The script has a fallback to mock detection. For real YOLO:
```bash
uv add ultralytics
uv run object_detection.py  # Downloads YOLOv8n on first run
```

### Issue: "matplotlib backend error"
**Solution:** Ensure display is available or use non-interactive backend
```bash
export MPLBACKEND=Agg
uv run mlp.py
```

---

## File Structure
```
Raj_DL_ESE/
├── README.md                    # This file
├── requirements.txt             # PyTorch + dependencies (no versions)
├── data.txt                     # Professional text corpus
├── .venv/                       # Virtual environment
├── data/                        # MNIST downloaded here
├── results/                     # All experiment outputs
├── mlp.py                       # EXP 1
├── cnn.py                       # EXP 2
├── transfer_learning.py         # EXP 3
├── cnn_vs_transfer.py           # EXP 4
├── performance_analysis.py      # EXP 5
├── object_detection.py          # EXP 6
├── rnn.py                       # EXP 7
├── lstm.py                      # EXP 8
├── gru.py                       # EXP 9
└── gan.py                       # EXP 10
```

---

## Key Features

✅ **Lightweight & Fast:** Each experiment runs in <2 minutes  
✅ **Fully Reproducible:** No randomness in dataset; PyTorch seed can be set  
✅ **Offline-Ready:** After first run, 100% offline (no internet needed)  
✅ **Professional Quality:** Real evaluation pipeline with proper metrics  
✅ **Viva-Friendly:** Clear code, documented sub-experiments, easy to modify on-the-fly  

---

## Citation & References

**Framework:** PyTorch (torch.org)  
**Datasets:** MNIST via torchvision  
**Pretrained Models:** MobileNetV2 via torchvision, YOLOv8 via Ultralytics  

---

## For Detailed Sub-Experiment Explanations

Refer to `results/TECHNICAL_EXPLANATION.txt` for complete breakdown of each sub-experiment requirement and implementation details matching the lab problem statement.

---

**Last Updated:** May 11, 2026  
**Lab Version:** RAJ DL ESE v1.0
