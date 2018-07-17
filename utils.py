from dtw import dtw
import numpy as np
import math


def align_trajectories(data):
    ls = np.argmax([d.shape[0] for d in data])  # select longest as basis

    data_warp = []

    for j, d in enumerate(data):
        dist, cost, acc, path = dtw(data[ls], d,
                                    dist=lambda x, y: np.linalg.norm(x - y, ord=1))

        data_warp += [d[path[1]][:data[ls].shape[0]]]

    return data_warp

def find_nearest_idx(array, value):
    array = np.array(array)
    return (np.abs(array-value)).argmin()

def process_motion_data(robot_states, perception_states, remove_still=True):
    """
    :param robot_states: array of tuples (time, state)
    :param perception_states: array of tuples (time, state)
    :return: original robot states, processed robot states, original perception states, processed perception states,
    durations
    """

    robot_times = [r[0] for r in robot_states][1:]  # FIXME: First time is weird
    robot_states = [r[4] for r in robot_states][1:]

    perception_times = [p[0] for p in perception_states]
    perception_states = [p[1] for p in perception_states]

    new_robot_states = []
    new_perception_states = []
    new_perception_times = []
    durations = []
    prev_time = robot_times[0]
    added_idxs = []

    for i, tp in enumerate(perception_times):
        robot_idx = find_nearest_idx(robot_times, tp)
        if robot_idx in added_idxs: continue  # Sometimes robot times comes delayed, ignore those robot states

        added_idxs.append(robot_idx)
        new_robot_states.append(np.array(robot_states[robot_idx]))
        new_perception_states.append(np.array(perception_states[i]))
        new_perception_times.append(perception_times[i])
        durations.append(robot_times[robot_idx] - prev_time)
        prev_time = robot_times[robot_idx]

    assert len(new_robot_states) == len(new_perception_states)

    # Remove waiting points at start and end
    if remove_still:
        i = 1
        while i < len(new_robot_states) - 1:
            diff_start = np.linalg.norm(new_robot_states[0] - new_robot_states[i])
            diff_end = np.linalg.norm(new_robot_states[-1] - new_robot_states[i])
            if diff_end < 1e-2 or diff_start < 1e-2:
                del new_robot_states[i]
                del durations[i]
                del new_perception_states[i]
                del new_perception_times[i]
            else:
                i += 1

    assert len(new_robot_states) > 0, "All points are removed!!"

    new_robot_states = np.array(new_robot_states)
    # Convert quaternion to euler
    euler_robot_states = np.zeros((new_robot_states.shape[0], 6))
    for i, rs in enumerate(new_robot_states):
        euler_robot_states[i,:3] = new_robot_states[i,:3]
        euler_robot_states[i,3:] = quaternion_to_euler_angle(*new_robot_states[i,3:])

    return euler_robot_states


def quaternion_to_euler_angle(x, y, z, w):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z

def gaussian(x, mu, var):
    exponent = -((x-mu)**2)/(2*var)
    return (1/math.sqrt((2*math.pi*var))) * math.exp(exponent)

