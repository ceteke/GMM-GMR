# GMM-GMR

This repository contains the implementation of [GMM-GMR](https://ieeexplore.ieee.org/document/4126276/) based imitation learning
from multiple trajectories. Python 2.7 is used for implementation so the
code can be ROS compatible.

## Installation

```sh
$ git clone & cd gmm_gmr
$ python setup.py install
```

## Usage

```python
from gmm_gmr.mixtures import GMM_GMR # Import the GMM_GMR class

... # Load your data and do stuff

gmm_gmr = GMM_GMR(trajectories, n_components)
gmm_gmr.fit()

t, generated_trajectory = gmm_gmr.generate_trajectory(interval)
```

### GMM_GMR(trajectories, n_components)

* ```trajectories```: Trajectories used for training. If these are not
aligned i.e ```list``` of trajectories, Dynamic Time Warping is used to align the trajectories. 
If you have aligned trajectories this should be ```np.ndarray``` with shape ```(N,T,D)``` where
```N``` is the number of trajectories, ```T``` is the time steps and ```D``` is the
dimension. **Note:** If you are using end-effector poses, it is recommended
that you convert Quaternions to Euler angles.
* ```n_components```: Number of principle components. This should be less than ```D```.

### GMM_GMR.fit()
This function trains PCA and GMM. Returns ```None```.

**Warning:** The explained variance should be greater than 95% so you can
generate nice looking trajectories.

### GMM_GMR.generate_trajectory(interval)
This function applies GMR on the time steps that are generated from minimum 
time step to maximum time step with increases of ```interval```. Then the trajectory is
converted from latent space to the original space.

Returns a tuple ```(time, trajectory``` time is the time steps and trajectory is 
the generated trajectory.