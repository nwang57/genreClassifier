import os
import sys
import numpy as np
import scipy

from scipy.fftpack import fft

from utils import DATA_DIR
from utils import show_feature, load_source, stft, plot_stft

def zero_crossing_rate(wavedata, window_size, sample_rate):
    num_windows = int(np.ceil(len(wavedata) / window_size))

    timestamps = (np.arange(0, num_windows - 1) * (window_size / float(sample_rate)))

    zcr = []

    for i in range(0, num_windows-1):
        start = i * window_size
        end = np.min([(start + window_size - 1), (len(wavedata))])

        zc = 0.5 * np.mean(np.abs(np.diff(np.sign(wavedata[start:end]))))
        zcr.append(zc)

    return np.asarray(zcr), np.asarray(timestamps)

def root_mean_square(wavedata, window_size, sample_rate):
    #original wavedata is an array of int16 [-32768:32767], square value may exceed the range
    wavedata = np.int32(wavedata)
    num_windows = int(np.ceil(len(wavedata)/window_size))

    timestamps = timestamps = (np.arange(0, num_windows - 1) * (window_size / float(sample_rate)))

    rms = []

    for i in range(0, num_windows-1):

        start = i * window_size
        end = np.min([(start + window_size - 1), len(wavedata)])

        rms_seg = np.sqrt(np.mean(wavedata[start:end]**2))
        rms.append(rms_seg)
    return np.asarray(rms), np.asarray(timestamps)


#spectual analysis
def spectual_centroid(wavedata, window_size, sample_rate):
    magnitude_spectrum = stft(wavedata, window_size)

    timebins, freqbins = np.shape(magnitude_spectrum)
    timestamps = np.arange(0,timebins-1) * (timebins / float(sample_rate))

    sc = []
    for t in range(timebins-1):
        power_spectrum = np.abs(magnitude_spectrum[t])**2
        sc_t = np.sum(power_spectrum * np.arange(1, freqbins+1)) / np.sum(power_spectrum)
        sc.append(sc_t)

    sc = np.asarray(sc)
    sc = np.nan_to_num(sc)

    return sc, np.asarray(timestamps)

if __name__ == "__main__":
    source = load_source("blues")

    zcr, ts = zero_crossing_rate(source['blues'][99]['wavedata'], 1024, source['blues'][99]['sample_rate'])
    source['blues'][99]['zcr'] = zcr

    rms, ts = root_mean_square(source['blues'][99]['wavedata'], 1024, source['blues'][99]['sample_rate'])
    source['blues'][99]['rms'] = rms
    #show_feature(source['blues'][99], "rms", ts, "blue_1024_rms")
    plot_stft(source['blues'][99]['wavedata'], source['blues'][99]['sample_rate'])

