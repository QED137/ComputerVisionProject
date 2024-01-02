import os
from tensorflow.keras.models import load_model
import numpy as np
import os
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize


def save_model(model, model_path):
    """
    Save a Keras model to a specified path.

    Args:
    model: Trained Keras model.
    model_path: Path where the model should be saved.
    """
    model.save(model_path)


def load_or_train_model(model_path, model_generation_func, compile_func, train_func, warmup_func, x_train, y_train,
                        x_test, y_test, epochs):
    """
    Load a model from a path if it exists, otherwise create, compile, train, and warmup it.
    """
    if os.path.exists(model_path):
        model = load_model(model_path)
        # Perform warmup after loading the model
        warmup_func(model, x_train, y_train, x_test, y_test)
    else:
        model = model_generation_func()
        model = compile_func(model)
        train_func(model, x_train, y_train, x_test, y_test, epochs)
        # Perform warmup after training
        warmup_func(model, x_train, y_train, x_test, y_test)
        save_model(model, model_path)

    return model


def save_predictions(predictions, filename):
    """Save model predictions to a file."""
    np.save(filename, predictions)

def load_predictions(filename):
    """Load model predictions from a file."""
    return np.load(filename)

def compute_and_save_predictions(model, x_data, filename):
    """
    Compute predictions using the model if not already saved.
    Save or load predictions as necessary.
    """
    if os.path.exists(filename):
        print(f"Loading predictions from {filename}")
        predictions = load_predictions(filename)
    else:
        print(f"Generating and saving predictions to {filename}")
        predictions = model.predict(x_data)
        save_predictions(predictions, filename)
    return predictions




