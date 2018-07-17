import pickle, numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from utils import align_trajectories, process_motion_data
from mixtures import GMR


state_files, perception_files = [], []

for i in [1,2,3,4,5,6,7]:
    state_files.append('sim_demos/{}/robot_states.pk'.format(i))
    perception_files.append('sim_demos/{}/pcae.pk'.format(i))

N = len(state_files)
ee_poses = []
latent_size = 3

for i, sf in enumerate(state_files):
    per = pickle.load(open(perception_files[i], 'rb'))
    rob = pickle.load(open(sf, 'rb'))

    ee_poses.append(np.array(process_motion_data(rob, per)))

ee_poses = np.array(align_trajectories(ee_poses))
print "Data shape:", ee_poses.shape

pca = PCA(n_components=latent_size)
ee_poses_latent = pca.fit_transform(ee_poses.reshape(-1,6))
print "Exp var.", np.sum(pca.explained_variance_ratio_)

T = ee_poses.shape[1]

temporal = np.array([range(T)] * N).reshape(-1, 1)
spatio_temporal = np.concatenate((temporal, ee_poses_latent), axis=1)

components = [5,10,15,20,25,30,35]
bics = []

for c in components:
    gmm = GaussianMixture(n_components=c)
    gmm.fit(spatio_temporal)
    bics.append(gmm.bic(spatio_temporal))

c = components[np.argmin(bics)]
print "Selected n component: {}".format(c)

gmm = GaussianMixture(n_components=c)
gmm.fit(spatio_temporal)

print "Is GMM converged: ", gmm.converged_

centers = gmm.means_
temporal_mu = centers[:,0]
spatial_mu = centers[:,1:]

centers_cartesian = pca.inverse_transform(spatial_mu)

f, axarr = plt.subplots(3, 1)

for i in range(N):
    for j in range(3):
        axarr[j].plot(ee_poses[i,:,j], label=i, linestyle=':')

for j in range(3):
    axarr[j].scatter(temporal_mu, centers_cartesian[:,j], label='centers')

interval = 0.1
times = np.arange(min(temporal_mu), max(temporal_mu)+interval, interval)

trj = []

gmr = GMR(gmm)
for t in times:
    trj.append(gmr.estimate(t))
trj = np.squeeze(np.array(trj))
trj = pca.inverse_transform(trj)

for j in range(3):
    axarr[j].plot(times, trj[:,j], label='estimated')

plt.legend()
plt.show()