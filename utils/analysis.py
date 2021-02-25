import matplotlib.pyplot as plt
import matplotlib.lines as mlines
plt.rcParams['figure.figsize'] = [9, 6]
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
import numpy as np
import scipy.signal
import scipy.ndimage
import seaborn as sns
import pandas as pd
from pathlib import Path
import seaborn as sns


def preprocess_emg(recording,
                   nch=32,
                   start=20,
                   end=-1,
                   lowcutoff=5,
                   highcutoff=20):
    data_subset = recording[:nch, :]
    for i, t in enumerate(data_subset):
        data_subset[i, :] = lowpass(rectify(
            highpass(data_subset[i, :], cutoff=highcutoff)),
                                    cutoff=lowcutoff)
        shortened = highpass(data_subset[:, start - 1:end], cutoff=0.1)
    return shortened


def highpass(sig, cutoff=50):
    b, a = scipy.signal.butter(2, cutoff, 'highpass', analog=False, fs=2000)
    return scipy.signal.filtfilt(b, a, sig, axis=-1)


def lowpass(sig, cutoff=500):
    b, a = scipy.signal.butter(2, cutoff, 'lowpass', analog=False, fs=2000)
    return scipy.signal.filtfilt(b, a, sig, axis=-1)


def notch(sig, freq=50):
    """
        Not really sure how to tune this effectively... 
    """
    b, a = scipy.signal.iirnotch(freq, 30, fs=2000)
    return scipy.signal.filtfilt(b, a, sig, axis=-1)


def moving_average(a, window_length=100):
    """
        boxcar window average
    """
    return scipy.ndimage.convolve1d(
        a, np.ones((window_length)), axis=1, mode='nearest') / window_length


def blur(a, sigma=50):
    """
        gaussian convolution
    """
    return scipy.ndimage.gaussian_filter1d(a,
                                           sigma=sigma,
                                           axis=1,
                                           mode='nearest')


def rectify(a):
    return np.abs(a)


def demean(a):
    assert a.shape[0] < a.shape[1]
    return a - np.mean(a, axis=1).reshape(-1, 1)


def preprocess(data, sigma=40):
    data_mv = data * 0.0002861
    data_mv = demean(data_mv)
    output = rectify(data_mv)
    output = blur(output, sigma)
    return output


def get_dropped_samples(counter):
    if len(counter.shape) > 1:
        counter = counter[0]
    drops = (counter[1:] - counter[:-1]) - 1
    drops[np.where(drops == -(2**16))] = 0
    for idx in np.where(drops != 0):
        print(f"Dropped {drops[idx]} samples at {idx}")
    return drops


def fill_time_array(dataset):
    dataset['time'] = np.arange(0,
                                len(dataset.time) / dataset.sampling_rate,
                                1 / dataset.sampling_rate)
