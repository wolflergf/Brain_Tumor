# Brain Tumor Classification System 🧠

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)](https://streamlit.io/)

A modernized deep learning system for classifying brain tumors from MRI images using Transfer Learning (EfficientNetB0) and advanced preprocessing techniques.

## ✨ Key Features

- **Advanced Model:** Leverages `EfficientNetB0` for superior feature extraction.
- **Enhanced Contrast:** Uses **CLAHE** (Contrast Limited Adaptive Histogram Equalization) to improve MRI visibility.
- **Robust Training:** Implements modern data augmentation and early stopping.
- **Interactive UI:** A professional Streamlit dashboard with real-time feedback and Plotly visualizations.
- **Reliability:** Comprehensive error handling and type hinting.

## 📁 Project Structure

```text
Brain_Tumor/
├── src/
│   ├── model_factory.py    # Architecture definition (EfficientNetB0)
│   ├── model_trainer.py    # Training pipeline and evaluation
│   ├── predictor.py        # Inference logic
│   └── preprocessing.py    # CLAHE and normalization
├── utils/                  # Helper scripts
├── app_v2.py               # Modern Streamlit interface
├── brain_tumor_model.keras # Trained model weights
└── requirements.txt        # Reproducible dependencies
```

## 🚀 Getting Started

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Training the Model

To train the model with the modernized pipeline:

```python
from src.model_trainer import BrainTumorTrainer

trainer = BrainTumorTrainer(dataset_dir="./brain_tumor_classification/Training")
trainer.train(epochs=20)
trainer.evaluate()
trainer.save_model("brain_tumor_model.keras")
```

### 3. Running the Web App

```bash
streamlit run app_v2.py
```

## 🧪 Methodology

- **Preprocessing:** MRI images are resized to 224x224, enhanced via CLAHE, and normalized to [0, 1].
- **Transfer Learning:** The base `EfficientNetB0` (pre-trained on ImageNet) is frozen, with a custom GlobalAveragePooling and Dense head for the 4 target classes.
- **Metrics:** Performance is tracked via Accuracy, Precision, Recall, and Confusion Matrices.

## ⚠️ Disclaimer

This tool is intended for research and educational purposes. It is **not** a medical device. Always consult a qualified medical professional for clinical diagnosis.

---
*Developed as part of an AI reliability enhancement initiative.*
