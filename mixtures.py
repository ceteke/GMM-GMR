import numpy as np
from utils import gaussian

class GMR:
    def __init__(self, gmm):
        self.gmm = gmm
        self.n_components = self.gmm.means_.shape[0]

    def xi_s_k(self, xi_t, k):
        cov_k = self.gmm.covariances_[k]
        mu_k = self.gmm.means_[k]

        mu_t_k = mu_k[0:1].reshape(-1, 1)
        mu_s_k = mu_k[1:].reshape(-1, 1)
        cov_t_k = cov_k[0:1, 0:1]
        cov_ts_k = cov_k[0:1, 1:]
        cov_st_k = cov_k[1:, 0:1]
        cov_s_k = cov_k[1:, 1:]

        return mu_s_k + cov_st_k.dot(np.linalg.inv(cov_t_k)).dot(xi_t - mu_t_k)

    def get_denom(self, xi_t):
        probs = 0.0
        for k in range(self.n_components):
            mu_t_k = self.gmm.means_[k][0]
            var_t_k = self.gmm.covariances_[k][0,0]
            probs += gaussian(xi_t, mu_t_k, var_t_k)
        return probs

    def estimate(self, xi_t):
        result = 0.0
        for k in range(self.n_components):
            xi_s_k_head = self.xi_s_k(xi_t, k)
            mu_t_k = self.gmm.means_[k][0]
            var_t_k = self.gmm.covariances_[k][0,0]
            beta_k = gaussian(xi_t, mu_t_k, var_t_k) / self.get_denom(xi_t)
            result += xi_s_k_head * beta_k
        return result