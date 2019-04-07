import pandas as pd
import numpy as np


class PCAReduction:
    ## Class Variables ##
    X_i = X_f = n_k = k_vec = None

    ## Constructors ##
    def __init__(self, final_dim):
        self.n_k = final_dim

    ## Methods ##
    def parse_data(self, file_path, sep):
        df = pd.read_csv(file_path, sep=sep)
        self.X_i = df.values

        assert(len(self.X_i[0]) >= self.n_k)

    def gen_data(self):
        np.random.seed(1)
        self.X_i = np.random.multivariate_normal(np.array([0, 0, 0]), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), 10)

        assert(len(self.X_i[0]) >= self.n_k)

    def _standardize(self):
        X_m = np.array([[np.mean(X_n)] for X_n in self.X_i.T])
        X_s = np.array([[np.std(X_n)] for X_n in self.X_i.T])
        self.X_i = self.X_i.T - X_m
        self.X_i /= X_s

    def _calc_k_vec(self):
        X_cov = np.cov(self.X_i)
        val, vec = np.linalg.eig(X_cov)

        # sort eigen
        eigen_pairs = [(np.abs(val[i]), vec[:, i]) for i in range(len(val))]
        eigen_pairs.sort(key=lambda pair: pair[0], reverse=True)

        # choose top k vectors
        self.k_vec = np.array([eigen_pairs[k][1] for k in range(self.n_k)])

    def reduce(self):
        self._standardize()
        self._calc_k_vec()
        self.X_f = self.k_vec.dot(self.X_i)


def main():
    pca = PCAReduction(final_dim=2)
    pca.gen_data()
    pca.reduce()


if __name__ == "__main__":
    main()

