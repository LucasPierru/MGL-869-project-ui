import tensorflow as tf
import numpy as np
import cifar10 as Cifar10
from tensorflow.keras.applications.efficientnet import preprocess_input
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
# Parameters
IMG_SIZE = 224
BATCH_SIZE = 32

class Tests(Cifar10.Cifar10):
  def __init__(self):
      super().__init__()
      self.model = None
      self.robustness_ds = None
      self.monitor_ds = None
      self.interpreter = None

  def preprocess(image, label):
      image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
      image = preprocess_input(image)  # EfficientNet-specific preprocessing
      return image, label

  def transform_dataset_size(self, x, y):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(self.preprocess).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

  def compile_model(self):
    self.model = tf.keras.models.load_model("../models/transfer_learning.keras", compile=False)
  # Run inference on test dataset and time it
  def measure_inference_time(self, dataset, num_batches=10):
    total_time = 0
    total_images = 0

    for i, (images, labels) in enumerate(dataset.take(num_batches)):
        start_time = time.perf_counter()
        _ = self.model.predict(images, verbose=0)
        end_time = time.perf_counter()

        batch_time = end_time - start_time
        total_time += batch_time
        total_images += images.shape[0]

        print(f"Batch {i+1}: {images.shape[0]} images in {batch_time:.4f} seconds")

    avg_time_per_image = total_time / total_images
    print(f"\nTotal images: {total_images}")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Average time per image: {avg_time_per_image*1000:.6f} ms")
    return avg_time_per_image
  
  def get_robustness_dataset(self):
    # Load the robustness dataset
    self.robustness_ds = self.transform_dataset_size(self.x_test[:1000], self.y_test[:1000])

  def add_noise(img, label, mean=32, stddev=5):
    noise = tf.random.normal(shape=tf.shape(img), mean=mean, stddev=stddev, dtype=img.dtype)
    noisy_img = tf.add(img, noise)
    return noisy_img, label

  def rotate_img_90(img, label):
    return tf.image.rot90(img), label

  def brighten_img(img, label, delta=50):
    return tf.image.adjust_brightness(img, delta), label

  def contrast_img(img, label, contrast_factor=-35):
    return tf.image.adjust_brightness(img, contrast_factor), label
  
  def add_gaussian_noise_to_dataset(self):
    return self.robustness_ds.map(self.add_noise)
  
  def rotate_90(self):
    return self.robustness_ds.map(lambda x, y: (self.rotate_img_90(x,y)))

  def adjust_brightness(self, delta=50):
    return self.robustness_ds.map(lambda x, y: (self.brighten_img(x,y, delta)))

  def adjust_contrast(self, contrast_factor=-35):
    return self.robustness_ds.map(lambda x, y: self.contrast_img(x, y, contrast_factor))
  
  def robustness_test_suite(self):
    variants = {
        'original': self.robustness_ds,
        'gaussian_noise': self.add_gaussian_noise_to_dataset(),
        "brightness": self.adjust_brightness(),
        "contrast": self.adjust_contrast(),
        "rotated_90_degrees": self.rotate_90(),
    }

    results = {}
    for name, ds in variants.items():
        loss, acc = self.model.evaluate(ds, verbose=0)
        results[name] = acc
        print(f"{name}: {acc:.4f}")

    return results

  def create_tf_lite_model(self):
    converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,         # TensorFlow Lite built-in ops
        tf.lite.OpsSet.SELECT_TF_OPS            # Enable TF ops fallback (larger binary)
    ]
    tflite_model = converter.convert()

    with open("model/model_quant.tflite", "wb") as f:
        f.write(tflite_model)
  
  def get_interpreter(self):
    self.interpreter = tf.lite.Interpreter(model_path="model/model_quant.tflite", num_threads=1)
    self.self.interpreter.allocate_tensors()

    input_details = self.interpreter.get_input_details()
    output_details = self.interpreter.get_output_details()

    # For convenience
    input_index = input_details[0]['index']
    output_index = output_details[0]['index']
    return input_index, output_index
  
  def measure_inference_time_lite(self):
    correct = 0
    total = len(self.x_test[:100])
    input_index, output_index = self.get_interpreter()
    start = time.time()

    for i in range(total):
        input_image = self.preprocess(self.x_test[i])
        input_image = np.expand_dims(input_image, axis=0)

        self.interpreter.set_tensor(input_index, input_image)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(output_index)
        predicted = np.argmax(output[0])

        if predicted == self.y_test[i]:
            correct += 1
    end = time.time()

    print(f"Accuracy: {correct / total:.4f}")
    print(f"Total inference time: {end - start:.2f} seconds")
    print(f"Average per image: {(end - start) / total:.4f} seconds")

  def get_monitor_dataset(self):
    # Load the robustness dataset
    self.monitor_ds = self.transform_dataset_size(self.x_test, self.y_test)
  
  # 1. Compute entropy (uncertainty)
  def compute_prediction_entropy(predictions):
      return np.array([entropy(p) for p in predictions])

  # 2. Plot distribution of entropy
  def plot_entropy_distribution(entropies):
      plt.figure(figsize=(6, 4))
      sns.histplot(entropies, bins=30, kde=True)
      plt.title("Prediction Entropy Distribution")
      plt.xlabel("Entropy")
      plt.ylabel("Frequency")
      plt.grid(True)
      plt.show()

  # 3. Confidence distribution
  def plot_confidence_distribution(predictions):
      confidences = np.max(predictions, axis=1)
      plt.figure(figsize=(6, 4))
      sns.histplot(confidences, bins=30, kde=True)
      plt.title("Prediction Confidence Distribution")
      plt.xlabel("Confidence (Max Softmax Score)")
      plt.ylabel("Frequency")
      plt.grid(True)
      plt.show()
  
  def monitor_batch(self, num_batches=5):
    all_preds = []
    all_labels = []

    for images, labels in self.monitor_ds.take(num_batches):
        preds = self.model.predict(images)
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Plot confidence
    self.plot_confidence_distribution(all_preds)

    # Plot entropy
    pred_entropy = self.compute_prediction_entropy(all_preds)
    self.plot_entropy_distribution(pred_entropy)

    # Optional: Save to CSV for analysis
    # np.savetxt("monitoring_entropy.csv", pred_entropy, delimiter=",")

    return all_preds, all_labels, pred_entropy

if __name__ == "__main__":
  tests = Tests()
  tests.compile_model()
  tests.get_robustness_dataset()
  tests.robustness_test_suite()
  tests.measure_inference_time_lite()
  tests.monitor_batch(val_ds, num_batches=5)