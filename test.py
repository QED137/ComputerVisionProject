
import matplotlib.pyplot as plt
# metrics.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize


def generate_confusion_matrix(y_test, y_pred, class_names):
    """
    Generates and plots a normalized confusion matrix.

    Args:
    y_test: True labels.
    y_pred: Predicted labels or probabilities.
    class_names: List of class names for labeling the axes.

    Returns:
    None
    """
    # Binarize the labels if they are not already
    y_test_binarized = label_binarize(y_test, classes=range(len(class_names)))

    # Convert probabilities to predicted class labels
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test_binarized, axis=1)

    # Generate the confusion matrix and normalize it
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plotting the normalized confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.show()


def plot_predictions(model, x_test, y_test, class_names):
    # Predictions on a subset of the test data
    num_images = 10
    random_indices = np.random.choice(range(len(x_test)), num_images)
    test_images = x_test[random_indices]
    test_labels = y_test[random_indices]
    predictions = model.predict(test_images)

    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(2, num_images, i+1)
        plt.imshow(test_images[i])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        predicted_label = np.argmax(predictions[i])
        true_label = np.argmax(test_labels[i])
        color = 'green' if predicted_label == true_label else 'red'

        plt.xlabel(f"Pred: {class_names[predicted_label]}\nTrue: {class_names[true_label]}", color=color)
    plt.show()
