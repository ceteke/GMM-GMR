import numpy as np
from utils import align_trajectories, gaussian
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


class GMM_GMR(object):
    """
    Implementation of GMM-GMR based imitation
    """

    def __init__(self, trajectories, n_components):
        '''
        :param trajectories: Trajectories obtained from demonstrations. If this not
        an aligned numpy array (i.e list of non-aligned trajectories) trajectories are
        aligned
        :param n_components: Number of PCA components
        '''
        if isinstance(trajectories, list):
            self.trajectories = np.array(align_trajectories(trajectories))

        self.T = self.trajectories.shape[1]
        self.N = self.trajectories.shape[0]
        self.D = self.trajectories.shape[2]

        self.pca = PCA(n_components)

    def fit(self):

        trajectories_latent = self.pca.fit_transform(self.trajectories.reshape(-1,self.D))
        print "Explained variance: {}%".format(np.sum(self.pca.explained_variance_ratio_))

        temporal = np.array([range(self.T)] * self.N).reshape(-1, 1)
        spatio_temporal = np.concatenate((temporal, trajectories_latent), axis=1)

        components = [5, 10, 15, 20, 25, 30, 35, 40, 45]
        bics = []

        for c in components:
            gmm = GaussianMixture(n_components=c)
            gmm.fit(spatio_temporal)
            bics.append(gmm.bic(spatio_temporal))

        c = components[np.argmin(bics)]
        print "Selected n mixtures: {}".format(c)

        self.gmm = GaussianMixture(n_components=c)
        self.gmm.fit(spatio_temporal)

        print "Is GMM converged: ", self.gmm.converged_

        self.gmr = GMR(self.gmm)

        self.centers = self.gmm.means_
        self.centers_temporal = self.centers[:, 0]
        self.centers_spatial_latent = self.centers[:, 1:]
        self.centers_spatial = self.pca.inverse_transform(self.centers_spatial_latent)

    def generate_trajectory(self, interval=0.1):
        times = np.arange(min(self.centers_temporal), max(self.centers_temporal) + interval,
                          interval)
        trj = []

        for t in times:
            trj.append(self.gmr.estimate(t))

        trj = np.squeeze(np.array(trj))
        trj = self.pca.inverse_transform(trj)

        return times, trj

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
        cov_st_k = cov_k[1:, 0:1]

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