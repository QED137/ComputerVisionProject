# Resent50
'''
Feature Extraction is performed by ResNet50 pretrained on imagenet weights.
Input size is 224 x 224.
'''

# resnet50_model.py
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten, UpSampling2D, Input

def load_and_preprocess_cifar10():
    # Load CIFAR-10 data
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # Preprocess images for ResNet50
    train_images = preprocess_image_input(train_images)
    test_images = preprocess_image_input(test_images)

    # Convert labels to one-hot encoding
    num_classes = 10
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

    return (train_images, train_labels), (test_images, test_labels)

def preprocess_image_input(input_images):
    input_images = input_images.astype('float32')
    output_images = tf.keras.applications.resnet50.preprocess_input(input_images)
    return output_images


def feature_extractor(inputs):
    # Define the ResNet50 feature extractor
    feature_extractor = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')(inputs)
    return feature_extractor


def classifier(inputs):
    # Define the classifier layers
    x = GlobalAveragePooling2D()(inputs)
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(10, activation="softmax", name="classification")(x)
    return x


def final_model(inputs):
    # Upsample and connect the feature extractor and classifier
    resize = UpSampling2D(size=(7, 7))(inputs)
    resnet_feature_extractor = feature_extractor(resize)
    classification_output = classifier(resnet_feature_extractor)
    return classification_output


def generate_resnet50_transfer_model(input_shape, num_classes=10, learning_rate=0.001):
    # Generate the complete model
    inputs = Input(shape=input_shape)
    classification_output = final_model(inputs)
    model = Model(inputs=inputs, outputs=classification_output)

    # Compile the model
    opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def create_and_compile_resnet50_model():
    # Load and preprocess data
    (train_X, train_Y), (valid_X, valid_Y) = load_and_preprocess_cifar10()

    # Generate and compile the model
    model = generate_resnet50_transfer_model(train_X.shape[1:], num_classes=10, learning_rate=0.001)

    return model, train_X, train_Y, valid_X, valid_Y
