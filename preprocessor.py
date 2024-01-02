# preprocessor.py
#prerpocessing data to avoid overfitting

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import random

def create_data_augmentation():
    return ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2]
    )

def random_blur(image):
    if random.random() < 0.5:
        return cv2.GaussianBlur(image, (5, 5), 0)
    return image

def add_random_noise(image):
    if random.random() < 0.5:
        noise_factor = 0.2
        noise = np.random.randn(*image.shape) * noise_factor
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image, 0, 255)  # Ensure pixel values are valid
        return noisy_image
    return image

def custom_augmentation_generator(x, y, batch_size=32):
    augmentation = create_data_augmentation()
    gen = augmentation.flow(x, y, batch_size=batch_size)
    for imgs, labels in gen:
        augmented_imgs = [add_random_noise(random_blur(img)) for img in imgs]
        yield np.array(augmented_imgs), labels
