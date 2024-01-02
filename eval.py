# train_eval.py
import tensorflow as tf
def train_model(model, x_train, y_train, x_test, y_test, epochs):
  history=model.fit(x_train, y_train, batch_size=220, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)
  return history

def warmup(model, x_train, y_train, x_test, y_test):
  # Warm up the JIT, we do not wish to measure the compilation time.
  initial_weights = model.get_weights()
  train_model(model, x_train, y_train, x_test, y_test, epochs=1)
  model.set_weights(initial_weights)

def evaluate_model(model, x_test, y_test):
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])