# 🏥 Generative Rationale-VLM: Transparent Medical Visual Question Answering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 11.8+](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**Generative Rationale-VLM** is an explainable medical visual question answering (VQA) framework that replaces "black-box" predictions with transparent 6-step clinical reasoning chains aligned with diagnostic protocols.

## 🎯 Key Features

- **6-Step Clinical Reasoning**: Generates transparent diagnostic chains: morphology → location → size → density → infiltration → malignancy risk
- **Explainability Metrics**: Novel evaluation metrics (RIO, RQR, CLC) for medical interpretability
- **Hallucination Detection**: Real-time verification against medical knowledge bases
- **Cross-Modal Distillation**: Balances inference efficiency with explanation quality
- **Clinical Validation**: Reduces physician decision time by 27%, improves diagnostic accuracy by 13.8%

## 📊 Performance Highlights

| Metric | Rationale-VLM | CNN-Attention Baseline | Improvement |
|--------|---------------|------------------------|-------------|
| Accuracy (PathVQA) | **84.7%** | 76.3% | +8.4% |
| RIO (Image Alignment) | **0.83** | 0.58 | +43% |
| RQR (Semantic Relevance) | **0.87** | 0.62 | +40% |
| Physician Decision Time | **-27%** | Baseline | |
| Diagnostic Accuracy | **+13.8%** | Baseline | |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/cx20030628/Generative_VLM.git
cd Generative_VLM

# Install dependencies
pip install -r requirements.txt
```

## ⚡ GPU Acceleration Setup

**Note**: Running on CPU is extremely slow (~10-20x slower). Follow these steps to configure CUDA environment for GPU acceleration:

### Step 1: Check Your GPU Compatibility

```bash
# Check if you have NVIDIA GPU
nvidia-smi

# Expected output should show GPU model and CUDA version
# If not found, you may not have NVIDIA GPU or drivers installed
```

### Step 2: Install Miniconda (if not installed)

Download Miniconda from [Miniconda website](https://docs.conda.io/en/latest/miniconda.html)

**Windows:**
```powershell
# Download and run the Miniconda installer
# After installation, restart your terminal
```

**Linux/Mac:**
```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# Install
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts, then restart terminal
```

### Step 3: Create Conda Environment with CUDA Support

```bash
# Create new conda environment with Python 3.10
conda create -n generative_vlm python=3.10 -y

# Activate environment
conda activate generative_vlm

# Install PyTorch with CUDA 11.8 (adjust based on your CUDA version)
# Check your CUDA version with: nvidia-smi
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### Step 4: Install Project Dependencies in Conda Environment

```bash
# Navigate to project directory
cd Generative_VLM

# Install remaining dependencies
pip install -r requirements.txt

# Install additional CUDA-optimized packages
pip install nvidia-cudnn-cu11==8.9.4.25  # For cuDNN acceleration
pip install nvidia-cublas-cu11==11.11.3.6  # For cuBLAS acceleration
```

### Step 5: Verify GPU Setup

```bash
# Run verification script
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Memory allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB')
    print(f'Memory cached: {torch.cuda.memory_reserved(0)/1e9:.2f} GB')
else:
    print('WARNING: CUDA not available. Training will be VERY slow on CPU!')
"

# Expected output if successful:
# PyTorch version: 2.0.1
# CUDA available: True
# GPU device: NVIDIA GeForce RTX 4090 (or your GPU model)
# CUDA version: 11.8
```

### Step 6: Performance Comparison

**CPU vs GPU Training Time:**
- **CPU only**: ~8-12 hours per epoch (estimated)
- **GPU (NVIDIA RTX 4090)**: ~20-30 minutes per epoch
- **Speedup**: 20-30x faster with GPU

## 🗂️ Dataset Preparation

The framework supports multiple medical VQA datasets. Choose your approach:

### Option 1: Using Preprocessed Features (Recommended for Training)

If you want to use the pre-extracted image features:

1. Download the processed dataset from our repository
2. Place `.npy` feature files in `data/processed/images/`
3. Place metadata in `data/raw/metadata.csv`

### Option 2: Download and Preprocess Raw Datasets

For PathVQA dataset:

#### Step 1: Get Hugging Face Access Token

1. Visit [Hugging Face](https://huggingface.co/) and create an account
2. Go to Settings → Access Tokens
3. Create a new token with "read" permissions
4. Set the token as environment variable:
   ```bash
   # Windows
   set HF_TOKEN=your_token_here
   
   # Linux/Mac
   export HF_TOKEN=your_token_here
   ```

#### Step 2: Download Raw Data

```bash
# Run the download script (automatically handles authentication)
python download_data.py
```

#### Step 3: Extract Image Features

```bash
# Extract visual features using BLIP-2 encoder (GPU-accelerated)
python data/preprocess.py --dataset pathvqa --output_dir data/processed --device cuda
```

## 🏋️ Model Training

### Activate Conda Environment Before Training

```bash
# Always activate conda environment first
conda activate generative_vlm

# Navigate to project directory
cd D:\MedVQAProjects\Generative_VLM
```

### Train Rationale-VLM (GPU-accelerated)

```bash
# Train Rationale-VLM with CUDA
python experiments/train.py --config experiments/config.yaml --device cuda --gpu_id 0

# Monitor GPU usage during training
nvidia-smi -l 1  # Updates every second
```

### Train Baseline Model

```bash
# Train baseline model
python experiments/train.py --config experiments/config_baseline.yaml --device cuda
```

### Training with Multiple GPUs (if available)

```bash
# Use DataParallel for multiple GPUs
python experiments/train.py --config experiments/config.yaml --device cuda --gpu_ids 0,1

# Use DistributedDataParallel for larger scale
python experiments/train.py --config experiments/config.yaml --device cuda --distributed
```

## 🔍 Inference

```bash
# Interactive inference (GPU-accelerated)
python experiments/infer.py --image_path <path_to_image> --question "<clinical_question>" --device cuda

# Batch inference
python experiments/batch_infer.py --input_csv test_cases.csv --output_csv results.csv --device cuda

# Performance benchmark
python experiments/benchmark.py --model rationale_vlm --device cuda --batch_sizes 1,4,8,16
```

## 📁 Project Structure

```
Generative_VLM/
├── .venv/                         # Python virtual environment
├── data/                          # Data processing
│   ├── __init__.py               # Data package init
│   ├── preprocess.py             # Data preprocessing script
│   ├── processed/                # Processed features
│   └── raw/                      # Raw datasets
├── experiments/                   # Training and evaluation
│   ├── baseline_infer.py         # Baseline inference script
│   ├── compare_results.py        # Result comparison
│   ├── config.yaml               # Main configuration
│   ├── debug_dataloader.py       # DataLoader debugging
│   ├── infer.py                  # Inference script
│   ├── optimization_report.txt   # Optimization report
│   ├── physician_evaluation.py   # Physician evaluation script
│   ├── start_training.py         # Training entry point
│   ├── train.py                  # Main training script
│   └── verify_config.py          # Config verification
├── metrics/                       # Evaluation metrics
│   ├── __init__.py              # Metrics package init
│   ├── clc.py                   # Clinical Logic Consistency
│   ├── rio.py                   # Rationale-Image Overlap
│   ├── robustness.py            # Robustness evaluation
│   └── rqr.py                   # Rationale-Question Relevance
├── models/                        # Model architectures
│   ├── __init__.py              # Models package init
│   ├── rationale_vlm.py         # Main Rationale-VLM
│   ├── student_modules.py       # Distilled modules
│   └── modules/                 # Sub-modules
│       ├── __init__.py          # Modules package init
│       ├── dynamic_weight.py    # Dynamic weighting module
│       └── hallucination.py     # Hallucination detection module
├── tests/                        # Unit tests
│   ├── __init__.py              # Tests package init
│   └── test_imports.py          # Import testing
├── utils/                        # Utility functions
├── .gitattributes               # Git attributes
├── .gitignore                   # Git ignore file
├── download_data.py             # Data download script
├── install_final.py             # Installation script
├── LICENSE                      # MIT License
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
└── setup.py                     # Package setup
```

## 🚨 Troubleshooting GPU Issues

### Common Problems and Solutions:

1. **"CUDA not available" error**
   ```bash
   # Check CUDA toolkit installation
   nvcc --version
   
   # Reinstall PyTorch with correct CUDA version
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

2. **Out of memory error**
   ```bash
   # Reduce batch size in config.yaml
   # training:
   #   batch_size: 8  # Reduce from 16
   
   # Use gradient accumulation
   # training:
   #   gradient_accumulation_steps: 2
   ```

3. **Slow GPU performance**
   ```bash
   # Enable cuDNN benchmarking
   export CUDNN_BENCHMARK=1
   
   # Use mixed precision training
   python experiments/train.py --config experiments/config.yaml --fp16
   ```

4. **Driver compatibility issues**
   ```bash
   # Update NVIDIA drivers
   # Visit: https://www.nvidia.com/Download/index.aspx
   
   # Check compatibility
   nvidia-smi
   # Driver Version should be >= 525.60.11 for CUDA 11.8
   ```

## 📊 Supported Datasets

| Dataset | Modality | Samples | Questions | GPU Preprocessing Time |
|---------|----------|---------|-----------|------------------------|
| PathVQA | Pathology | 4,998 | 32,799 | ~15 mins (GPU) / ~2 hours (CPU) |
| VQA-RAD | Radiology | 315 | 3,515 | ~2 mins (GPU) / ~20 mins (CPU) |
| SLAKE | Multimodal | 642 | 14,028 | ~5 mins (GPU) / ~45 mins (CPU) |

## 🔧 Configuration

Edit `experiments/config.yaml` to customize:

```yaml
# Hardware configuration
hardware:
  device: "cuda"  # or "cpu"
  gpu_id: 0
  num_workers: 4  # DataLoader workers
  pin_memory: true  # Faster data transfer to GPU

# Mixed precision training (faster, less memory)
training:
  use_amp: true  # Automatic Mixed Precision
  gradient_accumulation_steps: 1
  
# Model configuration
model:
  name: "rationale_vlm"
  hidden_size: 768
  num_attention_heads: 12

# Training parameters
training:
  batch_size: 16  # Adjust based on GPU memory
  learning_rate: 1e-5
  num_epochs: 20

# Data configuration
data:
  dataset: "pathvqa"
  image_size: 224
  max_question_length: 64
```

## 📈 Evaluation

```bash
# Run comprehensive evaluation (GPU-accelerated)
python experiments/evaluate.py --model rationale_vlm --dataset pathvqa --device cuda

# Generate detailed report
python experiments/generate_report.py --output report.pdf

# Benchmark performance
python experiments/benchmark.py --compare cpu cuda
```

## 🧪 Experiments Reproducibility

To reproduce the paper results with GPU acceleration:

```bash
# 1. Setup conda environment with CUDA
conda create -n generative_vlm python=3.10 pytorch=2.0.1 torchvision=0.15.2 cudatoolkit=11.8 -c pytorch -c nvidia
conda activate generative_vlm

# 2. Install project dependencies
pip install -r requirements.txt

# 3. Download and preprocess data (GPU accelerated)
python download_data.py
python data/preprocess.py --device cuda

# 4. Train models (GPU accelerated)
python experiments/train.py --config experiments/config.yaml --device cuda  # Rationale-VLM
python experiments/train.py --config experiments/config_baseline.yaml --device cuda  # Baseline

# 5. Run evaluations
python experiments/run_all_experiments.py --device cuda
```

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{chen2025generative,
  title={Comparative Study of Explainability in Generative VLM vs CNN Baseline Models for Medical Visual Question Answering},
  author={Chen, Xi and Zhuo, Ziyue},
  journal={Advance Machine Learning (WOA7015)},
  year={2025}
}
```

## 👥 Authors

- **CHEN XI** (25053692)
- **ZHUO ZIYUE** (24083635)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PathVQA, VQA-RAD, and SLAKE dataset creators
- Hugging Face for dataset hosting
- The 5 participating physicians for clinical evaluations
- NVIDIA for CUDA acceleration technology
- All contributors to open-source medical AI research

---

**Note**: This is a research project. The model outputs should be used as decision support only, not as definitive medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.
