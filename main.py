import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from dataloader import load_data, create_data_augmentation
from model import generate_model, compile_model
from eval import train_model, warmup, evaluate_model
from test import generate_confusion_matrix, plot_predictions
from resnet_Model import create_and_compile_resnet50_model
from performance_auc import compute_roc_auc, compute_mean_auc,plot_roc_curves
from model_manager import load_or_train_model, compute_and_save_predictions
# main.py
from preprocessor import custom_augmentation_generator
from model import generate_model

# Constants
NUM_EPOCH = 5
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#define models path
cnn_model_path = 'cnn_model.h5'
resnet_model_path = 'resnet_model.h5'



# Load data
(x_train, y_train), (x_test, y_test) = load_data()

# Load or Train CNN Model
model = load_or_train_model(
    cnn_model_path,
    lambda: generate_model(x_train.shape[1:]),
    compile_model,
    train_model,  # You will replace this with custom training using the generator
    warmup,
    x_train, y_train, x_test, y_test,
    NUM_EPOCH
)

# Using the custom augmentation generator in training
model.fit(custom_augmentation_generator(x_train, y_train, batch_size=32),
          validation_data=(x_test, y_test),
          steps_per_epoch=len(x_train) // 32,
          epochs=NUM_EPOCH)

evaluate_model(model, x_test, y_test)


# Save or Load Predictions
y_pred_cnn_filename = 'y_pred_cnn.npy'
y_pred_cnn = compute_and_save_predictions(model, x_test, y_pred_cnn_filename)

# Generate Confusion Matrix and Compute ROC AUC for CNN Model
generate_confusion_matrix(y_test, y_pred_cnn, class_names)
#fpr, tpr, roc_auc = compute_roc_auc(y_test, y_pred_cnn, num_classes=10)
#plot_roc_curves(fpr, tpr, roc_auc, num_classes=10, class_names=class_names)
# Plot ROC curves
#plot_roc_curves(fpr, tpr, roc_auc, num_classes=10, class_names=class_names)



#calling resnt50 model

'''model, train_X, train_Y, valid_X, valid_Y = create_and_compile_resnet50_model()


#warmup(model, train_X, train_Y, valid_X, valid_Y)
history = train_model(model, train_X, train_Y, valid_X, valid_Y,NUM_EPOCH)
evaluate_model(model, valid_X, valid_Y)'''

