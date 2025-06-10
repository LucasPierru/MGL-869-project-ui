import cifar10 as Cifar10
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time


class KMeans(Cifar10.Cifar10):

    def __init__(self):
        super().__init__()
        self.x_train_flat = None
        self.x_test_flat = None

    def normalize_data(self):

        self.y_train = self.y_train.flatten()
        self.y_test = self.y_test.flatten()

        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_test = self.x_test.astype('float32') / 255.0

        # Aplatir chaque image (32×32×3 → 3072)
        n_samples, h, w, c = self.x_train.shape
        self.x_train_flat = self.x_train.reshape(n_samples, h * w * c)
        self.x_test_flat = self.x_test.reshape(self.x_test.shape[0], h * w * c)
    
    def train_kmeans(self):
        self.model = KMeans(n_clusters=10, random_state=42, n_init=10)
        start_time = time.time()
        self.model.fit(self.x_train_flat)
        end_time = time.time()
        print(f"Training time: {end_time - start_time:.2f} seconds")

    def time_prediction(self):
        t0 = time.perf_counter()
        for _ in range(20):
            cluster_id = self.model.predict(self.x_test_flat)[0]
            print("")
            _ = self.class_names[cluster_id]
        t1 = time.perf_counter()

        avg_time = (t1 - t0) / 20
        print(f"Temps moyen pour classer 1 image : {avg_time * 1000:.3f} ms "
            f"({20} exécutions, total = {(t1 - t0):.3f} s)")
        
    def compute_purity(self):
        labels_cluster = self.model.labels_  # vecteur de taille 50000

        # 3. Calculer la pureté
        N = len(self.y_train)  # 50000

        purety_sum = 0
        for cluster_id in range(self.num_classes):
            # Masque des index appartenant au cluster
            mask = (labels_cluster == cluster_id)
            if not np.any(mask):
                continue
            # Comptage des vraies classes dans ce cluster
            true_labels_in_cluster = self.y_train[mask]
            count_per_class = np.bincount(true_labels_in_cluster, minlength=10)
            m_j = np.max(count_per_class)
            purety_sum += m_j

        purity = purety_sum / N
        print(f"Purity (train) = {purity * 100:.2f}%")

    def compute_performance(self):
        # Supposons que vous ayez déjà :
        # y_test : array des labels réels (taille N)
        # y_pred : array des labels prédits par KMeans (taille N)

        # 1. Accuracy
        acc = accuracy_score(self.y_test, self.y_pred)
        print(f"Accuracy : {acc * 100:.2f}%")

        # 2. Précision (average='macro' pour moyenne non pondérée sur les 10 classes)
        prec = precision_score(self.y_test, self.y_pred, average='macro', zero_division=0)
        print(f"Precision (macro) : {prec * 100:.2f}%")

        # 3. Rappel (recall)
        rec = recall_score(self.y_test, self.y_pred, average='macro', zero_division=0)
        print(f"Recall (macro)    : {rec * 100:.2f}%")

        # 4. F1-score
        f1 = f1_score(self.y_test, self.y_pred, average='macro', zero_division=0)
        print(f"F1-score (macro)  : {f1 * 100:.2f}%")

if __name__ == "__main__":
    kmeans = KMeans()
    kmeans.load_data_set()
    kmeans.normalize_data()
    kmeans.train_kmeans()
    kmeans.time_prediction()

    # Prédiction sur l'ensemble de test
    kmeans.y_pred = kmeans.model.predict(kmeans.x_test_flat)
    
    # Calcul de la pureté
    kmeans.compute_purity()

    # Calcul des performances
    kmeans.compute_performance()
