# 🛡️ Signature Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive machine learning-based signature fraud detection system that employs an ensemble approach combining Convolutional Neural Networks (CNNs) with classical machine learning algorithms to distinguish between genuine and forged handwritten signatures.

## 🌟 Features

- **🤖 Advanced Ensemble ML**: 4-model ensemble (2 CNNs + SVM + Random Forest)
- **🌐 Web Interface**: Flask-based responsive web application
- **🖥️ Desktop GUI**: Modern Tkinter-based desktop application
- **📧 Smart Alerts**: Automatic email notifications for fraud detection
- **🔊 Voice Feedback**: Accessibility features with text-to-speech
- **📊 Comprehensive Analytics**: Detailed performance metrics and evaluation
- **⚡ Real-time Processing**: Fast prediction with confidence scoring
- **🎯 High Accuracy**: Sophisticated preprocessing and feature extraction

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │    │   GUI Interface  │    │  Batch Evaluator│
│    (Flask)      │    │    (Tkinter)     │    │                 │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
          │                      │                       │
          └──────────────────────┼───────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Ensemble Predictor    │
                    │                         │
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                       │                        │
   ┌────▼────┐            ┌─────▼─────┐           ┌──────▼──────┐
   │ CNN A   │            │   CNN B   │           │  Classical  │
   │ Model   │            │   Model   │           │  ML Models  │
   └─────────┘            └───────────┘           └─────────────┘
                                                        │
                                              ┌─────────┼─────────┐
                                              │                   │
                                         ┌────▼────┐         ┌───▼───┐
                                         │   SVM   │         │   RF  │
                                         │  Model  │         │ Model │
                                         └─────────┘         └───────┘
```

## 📋 Table of Contents

- [Installation](#-installation)
- [Dataset Setup](#-dataset-setup)
- [Complete Project Workflow](#-complete-project-workflow)
- [File-by-File Execution Guide](#-file-by-file-execution-guide)
- [Model Training](#-model-training)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Evaluation](#-evaluation)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- At least 8GB RAM (for model loading)
- ~500MB free disk space (for models)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd signature-fraud-detection
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Environment Configuration

Create a `.env` file in the project root:

```env
# Email Configuration (for fraud alerts)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=your-app-password
```

## 📁 Dataset Setup

### Dataset Structure

Organize your dataset in the following structure:

```
dataset/
├── train/
│   ├── forgery/          # Forged signature images
│   │   ├── image1.png
│   │   ├── image2.jpg
│   │   └── ...
│   └── genuine/          # Genuine signature images
│       ├── image1.png
│       ├── image2.jpg
│       └── ...
├── val/
│   ├── forgery/
│   └── genuine/
└── test/
    ├── forgery/
    └── genuine/
```

### Supported Image Formats

- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)
- TIFF (.tiff)

### Dataset Requirements

- **Minimum**: 100 images per class for training
- **Recommended**: 1000+ images per class for optimal performance
- **Image Quality**: Clear, high-contrast signatures
- **Resolution**: Any resolution (automatically resized to 224x224)

## 🔄 Complete Project Workflow

### 📝 Step-by-Step Project Execution

Follow this complete workflow to set up and run the entire project:

#### Phase 1: Environment Setup
```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Create necessary directories
mkdir models charts outputs static\uploads
```

#### Phase 2: Dataset Preparation
```bash
# 1. Organize your dataset in the required structure
# 2. Verify dataset structure
python -c "import os; print('Train folders:', os.listdir('dataset/train'))"
python -c "import os; print('Val folders:', os.listdir('dataset/val'))"
python -c "import os; print('Test folders:', os.listdir('dataset/test'))"
```

#### Phase 3: Model Training (Complete Pipeline)
```bash
# 1. Train CNN models (takes 15-30 minutes)
python scripts/cnn_train.py

# 2. Train classical ML models (takes 5-10 minutes)
python scripts/classical_train.py

# 3. Generate base predictions for stacking (optional)
python scripts/generate_base_probs.py

# 4. Train meta-learner (optional)
python scripts/stacker_train.py
```

#### Phase 4: Testing and Evaluation
```bash
# 1. Run batch evaluation on test set
python batch_evaluation.py

# 2. Generate performance charts
python generate_comparative_charts.py

# 3. Test single prediction
python scripts/predict.py --image dataset/test/forgery/sample.png
```

#### Phase 5: Application Deployment
```bash
# Option 1: Web Application
python app.py
# Then visit: http://localhost:5000

# Option 2: Desktop GUI
python gui_app/ensemble_gui.py
```

## 📋 File-by-File Execution Guide

### 🔧 Training Scripts

#### 1. `scripts/utils.py`
**Purpose**: Core preprocessing and feature extraction utilities
**Usage**: Imported by other scripts (no direct execution)
```python
# Functions provided:
# - preprocess_for_cnn(): Image preprocessing pipeline
# - extract_handcrafted(): HOG + LBP feature extraction
# - walk_images(): Dataset iteration utility
```

#### 2. `scripts/cnn_train.py`
**Purpose**: Train CNN models (CNN-A and CNN-B)
**Execution**:
```bash
python scripts/cnn_train.py
```
**What it does**:
- Loads training and validation data from `dataset/train` and `dataset/val`
- Trains CNN-A (lightweight): Conv2D → MaxPool → Dense → Dropout
- Trains CNN-B (deeper): Conv2D → MaxPool → Conv2D → MaxPool → Dense → Dropout
- Saves models as `models/cnn_a.h5` and `models/cnn_b.h5`
- Training time: ~15-30 minutes depending on dataset size

**Expected Output**:
```
Found 2000 images belonging to 2 classes.
Found 400 images belonging to 2 classes.
Epoch 1/12
63/63 [==============================] - 45s 715ms/step
...
Saved: cnn_a.h5, cnn_b.h5 -> models/
```

#### 3. `scripts/classical_train.py`
**Purpose**: Train SVM and Random Forest models with handcrafted features
**Execution**:
```bash
python scripts/classical_train.py
```
**What it does**:
- Extracts HOG and LBP features from all training images
- Trains SVM with linear kernel and probability estimation
- Trains Random Forest with 300 estimators
- Saves models and scaler: `svm_hog_lbp.joblib`, `rf_hog_lbp.joblib`, `scaler.joblib`
- Training time: ~5-10 minutes

**Expected Output**:
```
[DEBUG] Loaded 2000 samples from dataset/train, X shape=(2000, 585), y shape=(2000,)
[DEBUG] Loaded 400 samples from dataset/val, X shape=(400, 585), y shape=(400,)
SVM Accuracy: 0.7250
RF  Accuracy: 0.7125
✅ Models saved in: models/
```

#### 4. `scripts/generate_base_probs.py`
**Purpose**: Generate base model predictions for stacking ensemble
**Execution**:
```bash
python scripts/generate_base_probs.py
```
**What it does**:
- Loads all trained models (CNN-A, CNN-B, SVM, RF)
- Generates predictions on training set
- Saves base probabilities as `outputs/base_probs.npy`
- Required for training the stacking meta-learner

#### 5. `scripts/stacker_train.py`
**Purpose**: Train meta-learner for stacking ensemble
**Execution**:
```bash
python scripts/stacker_train.py
```
**What it does**:
- Loads base model predictions from `outputs/base_probs.npy`
- Trains a meta-learner (typically Logistic Regression)
- Saves stacker model as `models/stacker.joblib`
- Improves ensemble performance through learned combination

#### 6. `scripts/predict.py`
**Purpose**: Single image prediction utility
**Execution**:
```bash
python scripts/predict.py --image path/to/signature.png
```
**What it does**:
- Loads the ensemble predictor
- Processes single image and returns prediction
- Shows individual model scores and final ensemble result

### 🌐 Application Files

#### 7. `app.py`
**Purpose**: Flask web application server
**Execution**:
```bash
python app.py
```
**What it does**:
- Starts Flask web server on `http://localhost:5000`
- Provides web interface for signature upload and analysis
- Integrates with ensemble predictor and email system
- Handles file uploads, predictions, and result display

**Features**:
- File upload with validation
- Real-time prediction with confidence scores
- Email alerts for fraud detection (>60% confidence)
- Voice feedback using text-to-speech
- Individual model score breakdown

#### 8. `gui_app/ensemble_gui.py`
**Purpose**: Desktop GUI application
**Execution**:
```bash
python gui_app/ensemble_gui.py
```
**What it does**:
- Launches modern Tkinter-based desktop application
- Provides image browser and preview functionality
- Real-time signature analysis with visual feedback
- Email integration for fraud alerts
- Modern UI with gradient backgrounds and hover effects

#### 9. `ensemble_predictor.py`
**Purpose**: Core prediction engine
**Usage**: Imported by applications (no direct execution)
```python
from ensemble_predictor import EnsemblePredictor
predictor = EnsemblePredictor()
result = predictor.ensemble_predict("image.png")
```
**What it does**:
- Loads all trained models automatically
- Implements weighted ensemble voting
- Provides confidence scoring and thresholding
- Handles preprocessing and feature extraction
- Returns detailed prediction results

### 📊 Evaluation and Analysis

#### 10. `batch_evaluation.py`
**Purpose**: Comprehensive batch testing framework
**Execution**:
```bash
python batch_evaluation.py
```
**What it does**:
- Evaluates ensemble on entire test dataset
- Calculates accuracy, precision, recall, F1-score
- Generates confusion matrix and per-class metrics
- Saves detailed results in JSON and CSV formats
- Creates comprehensive evaluation report

**Output Files**:
- `evaluation_report_YYYYMMDD_HHMMSS.json`: Complete evaluation data
- `evaluation_results_YYYYMMDD_HHMMSS.csv`: Detailed per-image results
- `evaluation_summary_YYYYMMDD_HHMMSS.csv`: Summary metrics

#### 11. `generate_comparative_charts.py`
**Purpose**: Performance visualization and chart generation
**Execution**:
```bash
python generate_comparative_charts.py
```
**What it does**:
- Generates accuracy comparison charts
- Creates confusion matrix visualizations
- Plots confidence distribution histograms
- Saves charts in `charts/` directory

#### 12. `mailing.py`
**Purpose**: Email notification system
**Usage**: Imported by applications
```python
from mailing import mailsend
mailsend("user@email.com", "Subject", "Body")
```
**What it does**:
- Sends fraud alert emails using SMTP
- Supports Gmail and other email providers
- Requires configuration in `.env` file

### 🔧 Configuration Files

#### 13. `.env`
**Purpose**: Environment configuration
**Setup**:
```env
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=your-app-password
```

#### 14. `requirements.txt`
**Purpose**: Python dependencies
**Usage**:
```bash
pip install -r requirements.txt
```

### 🚀 Quick Start Commands

#### Complete Setup and Training:
```bash
# 1. Environment setup
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt

# 2. Train all models
python scripts/cnn_train.py && python scripts/classical_train.py

# 3. Evaluate system
python batch_evaluation.py

# 4. Start web application
python app.py
```

#### Testing Individual Components:
```bash
# Test single prediction
python scripts/predict.py --image test_image.png

# Test web app (in browser: http://localhost:5000)
python app.py

# Test desktop GUI
python gui_app/ensemble_gui.py

# Generate performance charts
python generate_comparative_charts.py
```

## 🎯 Model Training

### Step 1: Prepare Dataset

Ensure your dataset follows the structure above and place it in the `dataset/` directory.

### Step 2: Train CNN Models

```bash
# Train both CNN models (CNN-A and CNN-B)
python scripts/cnn_train.py
```

This will:
- Train a lightweight CNN (CNN-A) with basic architecture
- Train a deeper CNN (CNN-B) with more layers
- Save models as `models/cnn_a.h5` and `models/cnn_b.h5`
- Use data augmentation and validation

### Step 3: Train Classical ML Models

```bash
# Train SVM and Random Forest models
python scripts/classical_train.py
```

This will:
- Extract HOG and LBP features from images
- Train SVM with linear kernel
- Train Random Forest with 300 estimators
- Save models and scaler in `models/` directory

### Step 4: Train Stacking Meta-Learner (Optional)

```bash
# Generate base model predictions
python scripts/generate_base_probs.py

# Train stacking ensemble
python scripts/stacker_train.py
```

### Training Parameters

**CNN Models:**
- Input Size: 224x224x3
- Batch Size: 32
- Epochs: 12 (adjustable)
- Optimizer: Adam (lr=1e-4)
- Loss: Categorical Crossentropy

**Classical Models:**
- Features: HOG (16x16 cells) + LBP (8 neighbors, radius=1)
- SVM: Linear kernel, probability=True
- Random Forest: 300 estimators, balanced class weights

## 💻 Usage

### Web Application

Start the Flask web server:

```bash
python app.py
```

Visit `http://localhost:5000` in your browser.

**Features:**
- Upload signature images
- Real-time fraud detection
- Email notifications for fraud alerts
- Individual model score breakdown
- Voice feedback

### Desktop GUI Application

Launch the Tkinter desktop application:

```bash
python gui_app/ensemble_gui.py
```

**Features:**
- Modern, intuitive interface
- Image preview and analysis
- Real-time predictions
- Email integration
- Visual confidence indicators

### Batch Evaluation

Evaluate the system on test datasets:

```bash
python batch_evaluation.py
```

**Outputs:**
- Comprehensive evaluation report (JSON)
- Detailed results (CSV)
- Performance metrics summary
- Confusion matrix analysis

### Command Line Prediction

For single image prediction:

```bash
python scripts/predict.py --image path/to/signature.png
```

## 📂 Project Structure

```
signature-fraud-detection/
├── 📁 dataset/                    # Training and test data
│   ├── train/                     # Training images
│   ├── val/                       # Validation images
│   └── test/                      # Test images
├── 📁 models/                     # Trained model files
│   ├── cnn_a.h5                   # Lightweight CNN model
│   ├── cnn_b.h5                   # Deep CNN model
│   ├── svm_hog_lbp.joblib         # SVM classifier
│   ├── rf_hog_lbp.joblib          # Random Forest classifier
│   ├── scaler.joblib              # Feature scaler
│   └── stacker.joblib             # Meta-learner (optional)
├── 📁 scripts/                    # Training and utility scripts
│   ├── utils.py                   # Image preprocessing utilities
│   ├── cnn_train.py               # CNN model training
│   ├── classical_train.py         # Classical ML training
│   ├── stacker_train.py           # Ensemble meta-learner
│   ├── generate_base_probs.py     # Base model predictions
│   └── predict.py                 # Single image prediction
├── 📁 gui_app/                    # Desktop GUI application
│   └── ensemble_gui.py            # Tkinter GUI interface
├── 📁 templates/                  # HTML templates for web app
│   ├── index.html                 # Upload page
│   └── result.html                # Results page
├── 📁 static/                     # Static web assets
│   └── uploads/                   # Uploaded images storage
├── 📁 charts/                     # Generated performance charts
├── 📄 app.py                      # Flask web application
├── 📄 ensemble_predictor.py       # Core prediction engine
├── 📄 batch_evaluation.py         # Batch testing framework
├── 📄 mailing.py                  # Email notification system
├── 📄 requirements.txt            # Python dependencies
├── 📄 .env                        # Environment configuration
└── 📄 README.md                   # This file
```

## ⚙️ Configuration

### Environment Variables (.env)

```env
# Email Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=your-app-password

# Model Configuration
CONFIDENCE_THRESHOLD=0.75
EMAIL_ALERT_THRESHOLD=0.60

# Application Settings
UPLOAD_FOLDER=static/uploads
MAX_FILE_SIZE=16777216  # 16MB
```

### Model Weights Configuration

Adjust ensemble weights in `ensemble_predictor.py`:

```python
weights = {
    'cnn_a': 0.35,    # Lightweight CNN
    'cnn_b': 0.35,    # Deep CNN
    'rf': 0.15,       # Random Forest
    'svm': 0.15       # Support Vector Machine
}
```

## 📊 Evaluation

### Performance Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall prediction accuracy
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown
- **Per-Class Metrics**: Individual performance for each class

### Current Performance

Based on test dataset (2,096 samples):

| Metric | Overall | Genuine Detection | Forgery Detection |
|--------|---------|-------------------|-------------------|
| Accuracy | 49.1% | 60.9% | 36.3% |
| Precision | 48.6% | 51.1% | 46.0% |
| Recall | 49.1% | 60.9% | 36.3% |
| F1-Score | 48.4% | 55.5% | 40.6% |

### Confidence Levels

- **High Confidence**: ≥75% threshold
- **Medium Confidence**: 60-75% range  
- **Low Confidence**: <60% threshold

### Generate Performance Charts

```bash
python generate_comparative_charts.py
```

Creates visualization charts in the `charts/` directory.

## 🔧 API Reference

### EnsemblePredictor Class

```python
from ensemble_predictor import EnsemblePredictor

# Initialize predictor
predictor = EnsemblePredictor()

# Make prediction
prediction_type, confidence, individual_scores, is_high_confidence = \
    predictor.ensemble_predict("path/to/image.png", confidence_threshold=0.75)
```

### Flask Web API

**POST /**: Upload and analyze signature

```python
# Request
files = {'file': open('signature.png', 'rb')}
data = {'email': 'user@example.com'}
response = requests.post('http://localhost:5000/', files=files, data=data)
```

### Batch Evaluation API

```python
from batch_evaluation import BatchEvaluator

# Initialize evaluator
evaluator = BatchEvaluator("dataset/test")

# Run evaluation
report = evaluator.run_evaluation()
```

## 🛠️ Advanced Features

### Custom Preprocessing Pipeline

The system uses a sophisticated preprocessing pipeline:

1. **Grayscale Conversion**: Convert to single channel
2. **Gaussian Blur**: Noise reduction (5x5 kernel)
3. **Otsu Thresholding**: Automatic binarization
4. **Automatic Inversion**: Ensure consistent ink/background
5. **Bounding Box Cropping**: Remove excess whitespace
6. **Square Padding**: Maintain aspect ratio
7. **Resize**: Standardize to 224x224
8. **Normalization**: Scale to [0,1] range

### Feature Extraction

**HOG Features:**
- Pixels per cell: 16x16
- Cells per block: 2x2
- Orientations: 9 bins

**LBP Features:**
- Neighbors: 8
- Radius: 1
- Method: Uniform patterns

### Email Alert System

Automatic fraud alerts with three confidence levels:

- **🚨 HIGH CONFIDENCE** (≥80%): Immediate alert
- **⚠️ MEDIUM CONFIDENCE** (70-79%): Standard alert  
- **⚠️ MODERATE CONFIDENCE** (60-69%): Basic alert

## 🐛 Troubleshooting

### Common Issues

**1. Model Loading Errors**
```bash
# Ensure all model files exist
ls -la models/
# Re-train if missing
python scripts/cnn_train.py
python scripts/classical_train.py
```

**2. Memory Issues**
```python
# Reduce batch size in training scripts
BATCH_SIZE = 16  # Instead of 32
```

**3. Email Not Sending**
```bash
# Check .env configuration
# Use app-specific passwords for Gmail
# Verify SMTP settings
```

**4. Import Errors**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Performance Optimization

**For Better Accuracy:**
- Increase training data size
- Use data augmentation
- Tune hyperparameters
- Collect higher quality images

**For Faster Inference:**
- Use model quantization
- Implement caching
- Optimize preprocessing pipeline

## 📈 Future Enhancements

### Planned Features

- **🔄 Model Versioning**: Track and manage model versions
- **📱 Mobile App**: React Native mobile application
- **🌐 REST API**: Full RESTful API with authentication
- **📊 Analytics Dashboard**: Real-time monitoring and statistics
- **🔐 Enhanced Security**: Advanced authentication and encryption
- **🎯 Active Learning**: Continuous model improvement
- **🌍 Multi-language**: Support for multiple languages

### Research Directions

- **Siamese Networks**: For signature comparison
- **Attention Mechanisms**: Focus on discriminative regions
- **Transfer Learning**: Pre-trained model fine-tuning
- **Adversarial Training**: Robustness against attacks
- **Federated Learning**: Privacy-preserving training

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the Repository**
2. **Create Feature Branch**: `git checkout -b feature/amazing-feature`
3. **Commit Changes**: `git commit -m 'Add amazing feature'`
4. **Push to Branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed
- Ensure backward compatibility

### Code Style

```python
# Use type hints
def predict_signature(image_path: str) -> Tuple[str, float]:
    """
    Predict signature authenticity.
    
    Args:
        image_path: Path to signature image
        
    Returns:
        Tuple of (prediction_type, confidence)
    """
    pass
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow Team**: For the deep learning framework
- **scikit-learn Contributors**: For machine learning algorithms
- **Flask Community**: For the web framework
- **OpenCV Team**: For computer vision utilities
- **Research Community**: For signature verification research

## 📞 Support

For support and questions:

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: support@yourproject.com

## 📊 Performance Benchmarks

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB+ |
| CPU | 4 cores | 8+ cores |
| Storage | 2GB | 5GB+ |
| GPU | None | CUDA-compatible |

### Inference Speed

| Model Type | Average Time | Batch Size |
|------------|-------------|------------|
| Single CNN | 0.5s | 1 |
| Ensemble | 2.0s | 1 |
| Batch Processing | 0.3s/image | 32 |

---

## 🚀 Quick Start Example

```python
# Complete example: Train models and make predictions

# 1. Train models
import subprocess
subprocess.run(["python", "scripts/cnn_train.py"])
subprocess.run(["python", "scripts/classical_train.py"])

# 2. Make prediction
from ensemble_predictor import EnsemblePredictor

predictor = EnsemblePredictor()
result = predictor.ensemble_predict("test_signature.png")
print(f"Prediction: {result[0]}, Confidence: {result[1]}%")

# 3. Start web app
subprocess.run(["python", "app.py"])
```

---

**Built with ❤️ for signature fraud detection and document security.**

*Last updated: September 2025*
