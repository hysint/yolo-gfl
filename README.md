# Project Setup

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/hysint/yolo-gfl.git
```

### 2. Install PyTorch with CUDA 12.6
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 3. Install other dependencies
```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:
```
ultralytics
onnx
onnxruntime-gpu
onnxslim
opencv-python
pandas
matplotlib
seaborn
plotly
scikit-learn
pandas-stubs
```

## System Requirements

- **GPU**: NVIDIA GPU with CUDA 12.6 support
- **Python**: 3.8 or higher