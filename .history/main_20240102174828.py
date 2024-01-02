import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import custom modules and functions
from dataloader import load_data
from model import generate_model, compile_model
from eval import train_model, warmup, evaluate_model
from test import generate_confusion_matrix
from resnet_Model import create_and_compile_resnet50_model
from performance_auc import compute_and_save_predictions
from preprocessor import custom_augmentation_generator

# Constants
NUM_EPOCH = 5
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Paths for model saving
cnn_model_path = 'cnn_model.h5'
resnet_model_path = 'resnet_model.h5'

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = load_data()

# -------------------------
# Baseline CNN Model
# -------------------------

# Load or Train CNN Model
model_cnn = load_or_train_model(
    cnn_model_path,
    lambda: generate_model(x_train.shape[1:]),
    compile_model,
    train_model,
    warmup,
    x_train, y_train, x_test, y_test,
    NUM_EPOCH
)

# Train CNN Model using custom augmentation generator
model_cnn.fit(
    custom_augmentation_generator(x_train, y_train, batch_size=32),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // 32,
    epochs=NUM_EPOCH
)

# Evaluate CNN Model
evaluate_model(model_cnn, x_test, y_test)

# Save or Load CNN Model Predictions
y_pred_cnn = compute_and_save_predictions(model_cnn, x_test, 'y_pred_cnn.npy')

# Generate Confusion Matrix and ROC AUC for CNN Model
generate_confusion_matrix(y_test, y_pred_cnn, class_names)

# -------------------------
# ResNet50 Model
# -------------------------
# Uncomment the following block to train and evaluate the ResNet50 model.

'''
# Load or Train ResNet50 Model
model_resnet50, train_X, train_Y, valid_X, valid_Y = create_and_compile_resnet50_model()

# Train ResNet50 Model
train_model(model_resnet50, train_X, train_Y, valid_X, valid_Y, NUM_EPOCH)

# Evaluate ResNet50 Model
evaluate_model(model_resnet50, valid_X, valid_Y)
'''

#  ROC curves, AUC computation, etc., can be run as well
