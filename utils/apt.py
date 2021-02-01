# ORGANIZATION

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.io as sio
from scipy import interpolate
from dtw import *


def build_subject_sessions(subject, files):
    """
        for a given subject, build a list of time-ordered list of session data
    """
    session_params = {
        'subject': subject,
        'hand': "ND",
        'direction': "CCW",
    }
    files = [
        f for f in files
        if all([p in f.name.split('.') for p in session_params.values()]) and
        ("Test" == f.name.split('.')[-2] or "Train" == f.name.split('.')[-3])
    ]
    sessions = [Session(f, verbose=False) for f in files]
    sessions = sorted(sessions, key=lambda s: s.order)
    return sessions


def combine_trials_from_sessions(sessions, target=None):
    # combine all trials from
    trials = []
    for session in sessions:
        session_trials = session.combine_trials(target=target)
        trials.extend(session_trials)
    return trials


def combine_blocks_from_sessions(sessions, target):
    block_lists = []
    for session in sessions:
        block_lists.extend(session.get_blocks(target=target))
    return block_lists


# find the median length trajectory in a list of trajectories
def find_median_length_trial(trials):
    lens = np.array([max(t.polar_trajectory.shape) for t in trials])
    lens_sorted_indices = np.argsort(lens)
    median_idx = lens_sorted_indices[len(lens) // 2]  # rough median index
    return trials[median_idx]


def stack_curves(sessions, target, reward=None, type=None):
    """
        stack same-shape, interpolated curves with attributes

        event_type:: "training" or "Demo"
        event_target:: 1 or 2 (meaning right or left)
        reward:: None, True, False (None gives True and False)

    """
    curve_matrix = []  #np.empty(shape=(0, 0, n_timepoints))
    for session in sessions:
        trials = session.combine_trials(target)
        for i, trial in enumerate(trials):
            if type == "interpolated":
                curve = trial.interpolated_polar_trajectory
            else:
                curve = trial.warped_polar_trajectory
            # filter for required trial types
            # if reward is specfied, filter for it
            if reward != None:
                if trial.reward == reward:
                    curve_matrix.append(curve)
            # otherwise take rewarded and unrewarded trials
            else:
                curve_matrix.append(curve)
    return np.stack(curve_matrix)


def stack_derivatives(sessions, target, reward=None):
    """
        stack same-shape, interpolated curves with attributes
        event_target:: 1 or 2 (meaning right or left)
        reward:: None, True, False (None gives True and False)

    """
    curve_matrix = []  #np.empty(shape=(0, 0, n_timepoints))
    for session in sessions:
        trials = session.combine_trials(target)
        for trial in trials:
            curve = trial.interpolated_polar_derivative
            # filter for required trial types
            # if reward is specfied, filter for it
            if reward != None:
                if trial.reward == reward:
                    curve_matrix.append(curve)
            # otherwise take rewarded and unrewarded trials
            else:
                curve_matrix.append(curve)
    return np.stack(curve_matrix)


def stack_reward(sessions, target):
    stacked_reward = []
    for session in sessions:
        trials = session.combine_trials(target)
        for t in trials:
            stacked_reward.append(t.reward)
    return np.array(stacked_reward).reshape(-1, 1)


def compute_interpolation(trajectory, precomputed_u=None):
    x = trajectory[0]
    y = trajectory[1]
    if precomputed_u is None:
        tckx, local_u = interpolate.splprep([np.arange(x.shape[0]), x],
                                            s=0,
                                            k=1)
        tcky, _ = interpolate.splprep([np.arange(y.shape[0]), y],
                                      u=local_u,
                                      s=0,
                                      k=1)
        return tckx, tcky, local_u
    else:
        tckx, _ = interpolate.splprep([np.arange(x.shape[0]), x],
                                      u=precomputed_u,
                                      s=0,
                                      k=1)
        tcky, _ = interpolate.splprep([np.arange(y.shape[0]), y],
                                      u=precomputed_u,
                                      s=0,
                                      k=1)
        return tckx, tcky, precomputed_u


def apply_interpolation(trajectory, npts=200, interp=None, precomputed_u=None):
    """
        pass in trajectory, get a fresh interpolation from scratch (no u out)
        pass in trajectory + interp, get samples of existing interp out
        pass in trajectory + precomputed_u, get a new interp of trajectory at precomputed_u
    """
    if interp is None:
        interp = compute_interpolation(trajectory, precomputed_u)
    _, tnew = interpolate.splev(np.linspace(0, 1, npts), interp[0], der=0)

    _, rnew = interpolate.splev(np.linspace(0, 1, npts), interp[1], der=0)
    return np.vstack([tnew, rnew])


def apply_time_warping(query, reference):
    rt = reference[0]
    rr = reference[1]
    qt = query[0]
    qr = query[1]
    #     sp = "symmetric2"
    #     sp = "asymmetricP2"
    sp = rabinerJuangStepPattern(5, slope_weighting='d', smoothed=False)
    alignment = dtw(qt, rt, keep_internals=False, step_pattern=sp)
    wq = warp(alignment, index_reference=False)
    new_theta = np.hstack([qt[wq], qt[wq][-1]])
    new_radius = np.hstack([qr[wq], qr[wq][-1]])
    return np.vstack([new_theta, new_radius])


def warp_and_interp_stack(trajectory_list):
    # warped
    warped_trajectories = []
    median_trajectory = find_median_length_trajectory(trajectory_list)
    for trajectory in trajectory_list:
        warped_trajectories.append(
            apply_time_warping(trajectory, median_trajectory))
    warped_trajectories = np.stack(warped_trajectories)
    # warp + interp
    # use mean of interpolated as reference
    warped_interped_trajectories = []
    for trajectory in warped_trajectories:
        warped_interped_trajectories.append(apply_interpolation(trajectory))
    warped_interped_trajectories = np.stack(warped_interped_trajectories)
    return warped_interped_trajectories


def interp_and_warp_stack(trajectory_list):
    # interped
    interped_trajectories = []
    for trajectory in trajectory_list:
        interped_trajectories.append(apply_interpolation(trajectory))
    interped_trajectories = np.stack(interped_trajectories)
    # interp + warp
    # use mean of interpolated as reference
    interped_mean = np.mean(interped_trajectories, axis=0)
    interped_warped_trajectories = []
    for trajectory in interped_trajectories:
        interped_warped_trajectories.append(
            apply_time_warping(trajectory, interped_mean))
    interped_warped_trajectories = np.stack(interped_warped_trajectories)
    return interped_warped_trajectories


def interp_stack(trajectory_list):
    # interped
    interped_trajectories = []
    for trajectory in trajectory_list:
        interped_trajectories.append(apply_interpolation(trajectory))
    interped_trajectories = np.stack(interped_trajectories)
    return interped_trajectories


class Trial():
    def __init__(self, trajectory, reward, MT, target, direction, requiredMT,
                 inMT, KP):
        self.trajectory = trajectory
        self.reward = reward
        self.MT = MT
        self.target = target
        self.requiredMT = requiredMT
        self.inMT = inMT
        self.direction = direction
        self.KP = KP

        self.whitened_trajectory = None
        self.whitened_derivative = None
        self.whitened_trajectory_difference = None
        self.whitened_derivative_difference = None
        self.difference_reward = None

        self.flip_trajectory()
        self.convert_to_polar()
        self.compute_derivative()
        self.interpolate_trajectory()
        self.interpolate_derivative()

    def flip_trajectory(self):
        if self.direction == "CCW":
            if self.target == 1:  # right
                self.aligned_trajectory = np.stack(
                    [-self.trajectory[0], self.trajectory[1]])
            elif self.target == 2:  # left
                self.aligned_trajectory = np.stack(
                    [self.trajectory[0], -self.trajectory[1]])
            else:
                print(self.target)
        elif self.direction == "CW":
            if self.target == 1:  # right
                self.aligned_trajectory = np.stack(
                    [-self.trajectory[0], -self.trajectory[1]])
            elif self.target == 2:  # left
                self.aligned_trajectory = np.stack(
                    [self.trajectory[0], self.trajectory[1]])
            else:
                print(self.target)

    def convert_to_polar(self):
        points = []
        for p in self.aligned_trajectory.T:
            r, theta = cartesian_to_polar(p)
            points.append([theta, r])
        self.polar_trajectory = np.vstack(points).T

    def is_outlier(self,
                   max_length=250,
                   max_radius=500,
                   theta_limit=4,
                   max_theta_derivative=0.3):
        if not np.any(self.polar_trajectory[0] > theta_limit)\
            and not np.any(self.polar_trajectory[0] < -theta_limit)\
                and not self.polar_trajectory.shape[1] > max_length\
                    and not np.any(self.polar_trajectory[1] > max_radius)\
                        and not np.any(self.polar_derivative[0] > max_theta_derivative):
            return False
        else:
            return True

    # compute derivative
    def compute_derivative(self):
        self.polar_derivative = np.vstack([
            np.diff(self.polar_trajectory[0], 1),
            np.diff(self.polar_trajectory[1], 1)
        ])

    # interpolate
    def compute_interpolation(self, trajectory, precomputed_u=None):
        x = trajectory[0]
        y = trajectory[1]
        if precomputed_u is None:
            tckx, local_u = interpolate.splprep([np.arange(x.shape[0]), x],
                                                s=0,
                                                k=1)
            tcky, _ = interpolate.splprep([np.arange(y.shape[0]), y],
                                          u=local_u,
                                          s=0,
                                          k=1)
            return tckx, tcky, local_u
        else:
            tckx, _ = interpolate.splprep([np.arange(x.shape[0]), x],
                                          u=precomputed_u,
                                          s=0,
                                          k=1)
            tcky, _ = interpolate.splprep([np.arange(y.shape[0]), y],
                                          u=precomputed_u,
                                          s=0,
                                          k=1)
            return tckx, tcky, precomputed_u

    def apply_interpolation(self,
                            trajectory,
                            npts=200,
                            interp=None,
                            precomputed_u=None):
        """
            pass in trajectory, get a fresh interpolation from scratch (no u out)
            pass in trajectory + interp, get samples of existing interp out
            pass in trajectory + precomputed_u, get a new interp of trajectory at precomputed_u
        """
        if interp is None:
            interp = self.compute_interpolation(trajectory, precomputed_u)
        _, tnew = interpolate.splev(np.linspace(0, 1, npts), interp[0], der=0)

        _, rnew = interpolate.splev(np.linspace(0, 1, npts), interp[1], der=0)
        return np.vstack([tnew, rnew])

    def interpolate_trajectory(self):
        interp = compute_interpolation(self.polar_trajectory)
        self.interpolation_knots = interp[2][:-1]  # computed u
        self.interpolated_polar_trajectory = apply_interpolation(
            self.polar_trajectory)

    def interpolate_derivative(self):
        self.interpolated_polar_derivative = apply_interpolation(
            self.polar_derivative, precomputed_u=self.interpolation_knots)

    def apply_time_warping(self, reference):
        qt = self.interpolated_polar_trajectory[0]
        qr = self.interpolated_polar_trajectory[1]
        qdt = self.interpolated_polar_derivative[0]
        qdr = self.interpolated_polar_derivative[1]
        #     sp = "symmetric2"
        #     sp = "asymmetricP2"
        # compute the warping alignment function
        sp = rabinerJuangStepPattern(5, slope_weighting='d', smoothed=True)
        alignment = dtw(qt, reference, keep_internals=False, step_pattern=sp)
        # get warping indices
        wq = warp(alignment, index_reference=False)
        self.warped_polar_trajectory = np.stack(
            [np.hstack([qt[wq], qt[wq][-1]]),
             np.hstack([qr[wq], qr[wq][-1]])])
        self.warped_polar_derivative = np.stack([
            np.hstack([qdt[wq], qdt[wq][-1]]),
            np.hstack([qdr[wq], qdr[wq][-1]])
        ])


class Block():
    def __init__(self, number, target):
        self.number = number
        self.target = target
        self.trials = []

    # stacking trials
    def combine_trials(self):
        trials = []
        for trial in self.trials:
            trials.append(trial)
        return trials


class Session():
    def __init__(self, file, verbose=False):
        self.verbose = verbose
        self.direction = file.name.split(".")[2]
        if file.name.split(".")[-2] == "Test":
            self.session_type = "Test"
        elif file.name.split(".")[-3] == "Train":
            self.session_type = "Train"
        else:
            raise ValueError("Unknown session type")

        self.parse_filename(file)

        # compile data from files
        self.load_session_from_mat(file)

        # blocks of Trial objects
        self.build_blocks()

    def parse_filename(self, file):
        # S305.ND.CCW.Day3.Train.1.mat
        self.name = file.name
        parts = self.name.split('.')
        self.subject = parts[0]
        self.hand = parts[1]
        self.direction = parts[2]
        self.day = parts[3]
        self.type = parts[4]
        if self.type == "Train":
            self.order = int(self.day[-1] + parts[5])
        else:
            self.order = int(self.day[-1] + "0")

    def load_session_from_mat(self, file):
        """
        input:: a session's mat file,
        return:: a list of left-starts and a list of right-starts

        """
        mat_data = sio.loadmat(file)
        data_array = mat_data['DATA']

        # data attributes / keys
        # if self.verbose:
        # for name in mat_data['DATA'].dtype.names:
        #     print(name, mat_data['DATA'][name].shape)

        data_array_x = data_array["x"]
        data_array_y = data_array["y"]

        # cursor positions
        self.KPs = []
        for d in data_array["KP"]:
            # remove zero padding from the ends of trials
            nz = d[0][np.nonzero(d[0])]  # numpy arrays
            inside_tube = []
            if nz.shape[0] == 0:
                self.KPs.append(None)
            else:
                for nzn in nz:
                    if nzn == 1:
                        inside_tube.append(True)
                    elif nzn == -1:
                        inside_tube.append(False)
                self.KPs.append(np.array(inside_tube).reshape(-1))

        # cursor positions
        self.trajectories = []
        for d in zip(data_array_x, data_array_y):
            # remove zero padding from the ends of trials
            nzx = d[0][0][np.nonzero(d[0][0])]  # numpy arrays
            nzy = d[1][0][np.nonzero(d[1][0])]
            if nzx.shape != nzy.shape:
                min_l = np.min([nzx.shape, nzy.shape])
                nzx = nzx[:min_l]
                nzy = nzy[:min_l]
            self.trajectories.append(
                np.stack([nzx.reshape(-1), nzy.reshape(-1)]))

        # Demo or Training
        event_types = []
        for et in data_array["event_type"]:
            event_types.append(et[0][0])  # list of strings
        self.event_types = np.array(event_types)  # np array of trials
        if self.verbose:
            print("event_type")
            print(self.event_types[:5])

        # Which Block: 1,2,3
        block_nums = []
        for bn in data_array["block_num"]:
            block_nums.append(bn[0][0])
        self.block_nums = np.array(block_nums)
        if self.verbose:
            print("block_num")
            print(self.block_nums[:5])

        # 1 or 2 == right or left
        event_targets = []
        for target in data_array["event_target"]:
            event_targets.append(target[0][0])
        self.event_targets = np.array(event_targets)
        if self.verbose:
            print("event_target")
            print(self.event_targets[:5])

        # min, max
        requiredMT = []
        for rMT in data_array["required_MT"]:
            requiredMT.append(rMT[0][0])
        self.requiredMT = np.array(requiredMT)
        if self.verbose:
            print("required_MT")
            print(self.requiredMT[:10])

        # None, True, False
        rewarded_list = []
        for success in data_array["Success"]:
            try:
                s = success[0][0]
                if s == 1:
                    rewarded_list.append(True)
                else:
                    rewarded_list.append(False)
            except:
                rewarded_list.append(None)
        self.rewarded = np.array(rewarded_list)
        if self.verbose:
            print("in tube")
            print(self.rewarded[:10])

        inMT = []
        for inMTbool in data_array["inMT"]:
            try:
                s = inMTbool[0][0]
                if s == 1:
                    inMT.append(True)
                else:
                    inMT.append(False)
            except:
                inMT.append(None)
        self.inMT = np.array(inMT)
        if self.verbose:
            print("in time limit")
            print(self.inMT[:10])

        MT = []
        for movement_time in data_array["MT"]:
            try:
                MT.append(movement_time[0][0][0] * 1000)
            except:
                MT.append(None)
        self.MT = np.array(MT)
        if self.verbose:
            print("movement time")
            print(self.MT[:10])

    def build_blocks(self):
        """
            only training trials
            no outlier trials
        """
        self.right_blocks = []
        self.left_blocks = []
        self.outlier_trials = []
        for block_number in range(1, len(np.unique(self.block_nums)) + 1):
            right_block = Block(block_number, target=1)
            left_block = Block(block_number, target=2)
            for i in range(len(self.trajectories)):
                if self.block_nums[i] == block_number\
                    and self.event_types[i] == "training":
                    trial = Trial(trajectory=self.trajectories[i],
                                  reward=self.rewarded[i],
                                  MT=self.MT[i],
                                  direction=self.direction,
                                  target=self.event_targets[i],
                                  requiredMT=self.requiredMT[i],
                                  inMT=self.inMT[i],
                                  KP=self.KPs[i])
                    if not trial.is_outlier():
                        if trial.target == 1:
                            right_block.trials.append(trial)
                        elif trial.target == 2:
                            left_block.trials.append(trial)
                    else:
                        self.outlier_trials.append(trial)
            self.right_blocks.append(right_block)
            self.left_blocks.append(left_block)

    # stacking trials
    def combine_trials(self, target):
        # combine all trials from blocks in session
        #
        trials = []
        if target == 1:
            for block in self.right_blocks:
                trials.extend(block.combine_trials())
        elif target == 2:
            for block in self.left_blocks:
                trials.extend(block.combine_trials())
        else:
            raise ValueError("Incorrect value for target; must be 1 or 2.")
        return trials

    def get_blocks(self, target):
        if target == 1:
            return self.right_blocks
        elif target == 2:
            return self.left_blocks
        else:
            raise ValueError("Incorrect value for target; must be 1 or 2.")
