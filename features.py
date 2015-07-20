import os
import sys
import numpy as np
import scipy

from scipy.fftpack import fft

from utils import DATA_DIR
from utils import show_feature, load_source, stft, plot_stft

#For example, a measurement with a high zero-crossing rate, i.e., the number of samples per second that cross the zero reference line, indicates that it is noisy.
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

#The RMS method takes the square of the instantaneous voltage before averaging, then takes the square root of the average.
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
#
#center of gravity (balancing point of the spectrum)
#It determines the frequency area around which most of the signal energy concentrates
#gives an indication of how “dark” or “bright” a sound is

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


# the frequency below which some fraction, k (typically 0.85, 0.9 or 0.95 percentile), of the cumulative spectral power resides
# measure of the skewness of the spectral shape
# indication of how much energy is in the lower frequencies
# It is used to distinguish voiced from unvoiced speech or music

def spectral_rolloff(wavedata, window_size, sample_rate, k=0.85):
    magnitude_spectrum = stft(wavedata, window_size)
    power_spectrum = np.abs(magnitude_spectrum) ** 2
    timebins, freqbins = np.shape(magnitude_spectrum)

    timestamps = (np.arange(0,timebins - 1) * (timebins / float(sample_rate)))

    sr = []

    spectral_sum = np.sum(power_spectrum, axis=1)
    for t in range(timebins-1):
        sr_t = np.where(np.cumsum(power_spectrum[t,:]) >= k * spectral_sum[t])[0][0]
        sr.append(sr_t)

    sr = np.asarray(sr).astype(float)
    sr = (sr / freqbins) * (sample_rate / 2.0)
    return sr, np.asarray(timestamps)


    # squared differences in frequency distribution of two successive time frames
    # measures the rate of local change in the spectrum

def spectral_flux(wavedata, window_size, sample_rate):
    magnitude_spectrum = stft(wavedata, window_size)
    timebins, freqbins = np.shape(magnitude_spectrum)

    timestamps = (np.arange(0,timebins - 1) * (timebins / float(sample_rate)))

    sf = np.sqrt(np.sum(np.diff(np.abs(magnitude_spectrum))**2, axis=1)) / freqbins

    return sf[1:], np.asarray(timestamps)

if __name__ == "__main__":
    source = load_source("blues")

    # zcr, ts = zero_crossing_rate(source['blues'][99]['wavedata'], 512, source['blues'][99]['sample_rate'])
    # source['blues'][99]['zcr'] = zcr

    # rms, ts = root_mean_square(source['blues'][99]['wavedata'], 512, source['blues'][99]['sample_rate'])
    # source['blues'][99]['rms'] = rms

    # sc, ts = spectual_centroid(source['blues'][99]['wavedata'], 512, source['blues'][99]['sample_rate'])
    # source['blues'][99]['sc'] = sc

    # sr, ts = spectral_rolloff(source['blues'][99]['wavedata'], 512, source['blues'][99]['sample_rate'])
    # source['blues'][99]['sr'] = sr

    sf, ts = spectral_flux(source['blues'][99]['wavedata'], 512, source['blues'][99]['sample_rate'])
    source['blues'][99]['sf'] = sf
    show_feature(source['blues'][99], "sf", ts, "blue_512_sf")
    #plot_stft(source['blues'][99]['wavedata'], source['blues'][99]['sample_rate'], 512)

