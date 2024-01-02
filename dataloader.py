# data_loader.py
import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

def load_data():
  result = tfds.load('cifar10', batch_size = -1)
  (x_train, y_train) = result['train']['image'],result['train']['label']
  (x_test, y_test) = result['test']['image'],result['test']['label']

  x_train = x_train.numpy().astype('float32') / 256
  x_test = x_test.numpy().astype('float32') / 256

  # Convert class vectors to binary class matrices.
  y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
  return ((x_train, y_train), (x_test, y_test))

def create_data_augmentation():
    return ImageDataGenerator(

            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images horizontally
            zoom_range=0.1  # randomly zoom into images

    )
