# Nathan Hancock KMeans Program

import sys
import random
# import matplotlib.pyplot as plt
import numpy as np


class FileInput:

    data_matrix = []
    k = None

    @classmethod
    def run(cls, k_val, file_name):
        FileInput.k = k_val

        with open(file_name, "r") as rf:
            for cnt, line in enumerate(rf):
                '''converts data points to int values'''
                data_pt = list(map(int, (line.strip('\n').split("\t"))))
                FileInput.data_matrix.append(data_pt)


class KMeans:

    def __init__(self, k, matrix, max_iter=150):
        self.k = k
        self.matrix = matrix
        self.max_iter = max_iter
        self.centroids = None
        self.data_class = np.zeros(matrix.shape[0])

    def get_centroids(self):

        """centroids stored in numpy array"""
        centroids = np.zeros((self.k, 2))
        '''list of random values'''
        random_list = []
        for i in range(self.k):
            """random val for array point for centroid"""
            random_val = random.randint(0, self.matrix.shape[0]-1)
            while random_val in random_list:
                random_val = random.randint(0, self.matrix.shape[0])
            random_list.append(random_val)
            centroid = self.matrix[random_val]
            centroids[i] = centroid
        self.centroids = centroids

    '''returns 1d array containing distances from input centroid'''
    @staticmethod
    def get_distances(input_centroid, points):
        return np.apply_along_axis(np.linalg.norm, 1, points - input_centroid)

    def k_means(self):
        num_row = self.matrix.shape[0]
        centroid_distances = np.zeros([num_row, self.k])

        for x in range(self.max_iter):
            for i, c in enumerate(self.centroids):
                centroid_distances[:, i] = self.get_distances(c, self.matrix)
            self.data_class = np.argmin(centroid_distances, axis=1)
            for c in range(self.k):
                self.centroids[c] = np.mean(self.matrix[self.data_class == c], 0)
        with open('output.txt', 'w') as wf:
            for idx, val in enumerate(self.matrix):
                new_array = np.append(val, self.data_class[idx])
                for i in new_array:
                    wf.write(str(i) + '\t')
                wf.write('\n')

        """
            OPTIONAL CODE TO OUTPUT GRAPH 
            SIMPLY CHANGE group_colors and
            centroid_colors TO MATCH NUMBER 
            OF CLUSTERS REQUIRED AND 
            UNCOMMENT THE MATPLOTLIB LIBRARY
        """
        """
        group_colors = ['skyblue', 'coral', 'lightgreen', 'violet']
        centroid_colors = ['blue', 'darkred', 'green', 'purple']
        colors = [group_colors[j] for j in self.data_class]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(self.matrix[:, 0], self.matrix[:, 1], color=colors)
        ax.scatter(self.matrix[:, 0], self.matrix[:, 1], color=colors, alpha=0.5)
        ax.scatter(self.centroids[:, 0], self.centroids[:, 1], color=centroid_colors, marker='o', lw=2)
        ax.set_xlabel('$x_val$')
        ax.set_ylabel('$y_val$')
        plt.show()
        """


if __name__ == "__main__":

    FileInput.run(sys.argv[1], sys.argv[2])
    np_matrix = np.array(FileInput.data_matrix)
    kTest = KMeans(int(sys.argv[1]), np_matrix)
    kTest.get_centroids()
    kTest.k_means()

