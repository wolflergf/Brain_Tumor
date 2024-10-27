# Brain Tumor Classification System 🧠

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)](https://streamlit.io/)

A deep learning-based system for classifying brain tumors from MRI images. This project includes both the model training pipeline and a web interface for real-time classification.

## Features

- Classification of four types of brain conditions:
  - Glioma
  - Meningioma
  - Pituitary
  - No Tumor
- Interactive web interface built with Streamlit
- Real-time image processing and classification
- Detailed probability breakdown for each classification
- Data augmentation for improved model robustness
- Comprehensive evaluation metrics

## Project Structure

```
brain_tumor_classification/
├── Training/                 # Training dataset directory
├── train_model.py           # Model training script
├── app.py                   # Streamlit web application
├── brain_tumor_model.keras  # Trained model file
└── requirements.txt         # Project dependencies
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/wolflergf/Brain_Tumor.git
cd Brain_Tumor
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model on your own dataset:

1. Organize your dataset in the following structure:

```
Training/
├── glioma/
├── meningioma/
├── pituitary/
└── no_tumor/
```

2. Run the training script:

```bash
python train_model.py
```

The script will:

- Apply data augmentation
- Train the CNN model
- Generate performance metrics
- Save the trained model

### Running the Web Application

To start the web interface:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Model Architecture

The CNN architecture consists of:

- 2 Convolutional layers with ReLU activation
- 2 MaxPooling layers
- Flatten layer
- Dense layer with 128 units
- Dropout layer (0.2)
- Output layer with softmax activation

## Performance Metrics

The model is evaluated using:

- Accuracy
- Loss
- Classification report
- Training/validation curves

## Dependencies

- TensorFlow 2.x
- Streamlit
- NumPy
- Pillow
- scikit-learn
- matplotlib

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is intended to be used as a support system only and should not be used as the sole basis for medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## Author

- [wolflergf](https://github.com/wolflergf)

## Acknowledgments

- Dataset providers
- TensorFlow and Streamlit communities
- Contributors and testers
