# UBC-OCEAN Image Classification using Keras and JAX



## Overview

This project focuses on developing a machine learning model for classifying images using the UBC-OCEAN dataset. The model is built using the Keras library with JAX backend for enhanced performance. The project involves data preprocessing, model configuration, training, and evaluation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sannidhayj20/Ovarian-Cancer-Subtype-Classification-UBC-Ocean-.git
   cd ubc-ocean-image-classification
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Adjust the configuration parameters in the `Config` class as needed:

```python
class Config:
    is_submission = False
    
    # Reproducibility
    SEED = 42
    
    # Training
    train_csv_path = "/path/to/train.csv"
    train_thumbnail_paths = "/path/to/train_thumbnails"
    batch_size = 8
    learning_rate = 1e-3
    epochs = 2
    
    # Inference
    test_csv_path = "/path/to/test.csv"
    test_thumbnail_paths = "/path/to/test_thumbnails"
```

## Usage

1. Set up the environment variable for the backend:
   ```python
   import os
   os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow", "torch"
   ```

2. Run the training script:
   ```python
   python train.py
   ```

3. Evaluate the model:
   ```python
   python evaluate.py
   ```

## Results

The results of the model training and evaluation will be saved in the `results` directory. You can visualize the performance using various metrics and plots provided in the notebook.

## Acknowledgments

This project was inspired by the UBC-OCEAN dataset and utilizes various machine learning libraries including Keras, TensorFlow, and JAX.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


