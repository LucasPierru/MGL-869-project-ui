import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.datasets import cifar10

class KMeansClassifier:
    def __init__(self, n_clusters=10, random_state=42, n_init=10):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.cluster_labels = ["airplane","automobile","bird","cat","deer",
                               "dog","frog","horse","ship","truck"]
        self.model = KMeans(n_clusters=self.n_clusters,
                            random_state=self.random_state,
                            n_init=self.n_init)

    def load_and_preprocess(self):
        # Charge et normalise CIFAR-10, aplatis les images
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        self.y_train = y_train.flatten()
        self.y_test  = y_test.flatten()

        x_train = x_train.astype('float32') / 255.0
        x_test  = x_test.astype('float32') / 255.0

        n_samples, h, w, c = x_train.shape
        self.x_train_flat = x_train.reshape(n_samples, h*w*c)
        self.x_test_flat  = x_test.reshape(x_test.shape[0], h*w*c)

    def fit(self):
        start = time.time()
        self.model.fit(self.x_train_flat)
        end = time.time()
        print(f"Training time: {end - start:.2f} seconds")

    def predict(self, X):
        # renvoie les étiquettes de clusters converties en classes
        cluster_ids = self.model.predict(X)
        return np.array([self.cluster_labels[i] for i in cluster_ids])

    def time_single_prediction(self, n_iter=20):
        # mesure le temps moyen pour classer 1 image
        start = time.perf_counter()
        for _ in range(n_iter):
            _ = self.predict(self.x_test_flat[:1])
        end = time.perf_counter()
        avg_ms = (end - start) / n_iter * 1000
        print(f"Temps moyen pour classer 1 image : {avg_ms:.3f} ms "
              f"({n_iter} exécutions, total = {(end - start):.3f} s)")

    def evaluate(self):
        # obtient y_pred comme indices de cluster → classes
        y_pred = self.predict(self.x_test_flat)
        # convertit classes en indices pour metrics
        y_pred_idx = np.array([self.cluster_labels.index(c) for c in y_pred])
        
        acc  = accuracy_score(self.y_test, y_pred_idx)
        prec = precision_score(self.y_test, y_pred_idx, average='macro', zero_division=0)
        rec  = recall_score(self.y_test, y_pred_idx, average='macro', zero_division=0)
        f1   = f1_score(self.y_test, y_pred_idx, average='macro', zero_division=0)
        
        print(f"Accuracy       : {acc * 100:.2f}%")
        print(f"Precision (M)  : {prec * 100:.2f}%")
        print(f"Recall    (M)  : {rec * 100:.2f}%")
        print(f"F1-score  (M)  : {f1 * 100:.2f}%")

# Exemple d’utilisation
if __name__ == "__main__":
    clf = KMeansClassifier()
    clf.load_and_preprocess()
    clf.fit()
    clf.time_single_prediction()
    clf.evaluate()
