from random import sample
from sklearn.datasets._samples_generator import make_blobs
import numpy as np
import matplotlib.pyplot as plt

class KMeansClassifier:
    def __init__(self, k=3) -> None:
        self.k = k
        self.new_centroids = []
        self.centroids = None
        self.iterations = 0

    def distance(self, p1 : np.array, p2 : np.array):
        return np.sqrt(np.sum((p1 - p2)**2, axis=0))
    
    def multi_distance(self, p, ps):
        distances = []
        for point in ps:
            distances.append(self.distance(point, p))
        return np.argmin(distances)
    
    def delta(self, points1, points2):
        distances = []
        for i in range(len(points1)):
            distances.append(self.distance(points1[i], points2[i]))
        return sum(distances) / len(points1)

    def train(self, data_points, threshold=0.0001) -> None:
        data_points = np.array(data_points, dtype=float)

        for i in range(self.k):
            self.new_centroids.append(data_points[i * len(data_points)//self.k])
        
        self.new_centroids = np.array(self.new_centroids)
        self.centroids = np.zeros(shape=self.new_centroids.shape)

        while self.delta(self.new_centroids, self.centroids) > threshold:
            self.centroids = self.new_centroids.copy()
            self.new_centroids = []
            self.data_clusters = {i : [] for i in range(self.k)}

            for data_point in data_points:
                self.data_clusters[self.multi_distance(data_point, self.centroids)].append(data_point)
            
            for i in range(self.k):
                self.new_centroids.append(np.mean(self.data_clusters[i], axis=0))
            
            self.new_centroids = np.array(self.new_centroids, dtype=float)
            self.iterations += 1

    def fit(self, datapoints) -> None:
        try : 
            if not self.centroids: raise Exception('Model is not trained on any data!')
        except :
            pass

        for datapoint in datapoints:
            cluster = self.multi_distance(datapoint, self.centroids)
            self.data_clusters[cluster].append(datapoint)

        for i in range(self.k):
            self.centroids[i] = np.mean(self.data_clusters[i], axis=0)
    
    def info(self) -> None:
        print('-'*32)
        print("        -: Clustering Info :-")
        print("Total Iterations : ", self.iterations)
        print('-'*16)

        print("Cluster Sizes :- ")
        for i in range(self.k):
            print(f'Cluster {i+1} : {len(self.data_clusters[i])}')
        print('-'*16)

        print("Centroids :-")
        for i in range(self.k):
            print(f'Centroid {i+1} : {self.centroids[i]}')
        print('-'*32)
    
    def show_as_graph(self) -> None:
        try:
            if not self.centroids: raise Exception('Model is not trained on any data!')
        except ValueError:
            pass

        colors = ['r', 'g', 'b', 'y', 'c', 'm']

        fig, ax = plt.subplots()

        fig.set_facecolor((1, 0, 0, 0.5))

        for i in range(self.k):
            X_data = [item[0] for item in self.data_clusters[i]]
            Y_data = [item[1] for item in self.data_clusters[i]]
            ax.scatter(X_data, Y_data, c=colors[i%6])
        
        cenX = [item[0] for item in self.centroids]
        cenY = [item[1] for item in self.centroids]

        ax.scatter(cenX, cenY, c='k', marker='x')
        ax.grid(True)
        ax.set_title('K-Means-Clustering')
        plt.show()

if __name__ == '__main__':
    sample_size = 400
    k = 3

    X, y = make_blobs(n_samples=sample_size, centers=k, center_box=(-20, 20))

    classifier = KMeansClassifier(k=k)
    classifier.train(X[:300])

    classifier.info()
    classifier.show_as_graph()

    classifier.fit(X[300:])

    classifier.info()
    classifier.show_as_graph()
