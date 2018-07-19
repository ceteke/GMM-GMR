import pickle
import matplotlib.pyplot as plt
from gmm_gmr.mixtures import GMM_GMR

ee_poses = pickle.load(open('data/example_data.pk', 'rb'))

gmm_gmr = GMM_GMR(ee_poses, 3)
gmm_gmr.fit()

f, axarr = plt.subplots(3, 1)

for i in range(len(ee_poses)):
    for j in range(3):
        axarr[j].plot(gmm_gmr.trajectories[i,:,j], linestyle=':')

for j in range(3):
    axarr[j].scatter(gmm_gmr.centers_temporal, gmm_gmr.centers_spatial[:,j], label='centers')

times, trj = gmm_gmr.generate_trajectory(0.1)
for j in range(3):
    axarr[j].plot(times, trj[:,j], label='estimated')

plt.legend()
plt.show()