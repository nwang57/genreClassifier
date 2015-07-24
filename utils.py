import os
import sys
import timeit
import numpy as np
import sunau
import wave
import scipy
import scipy.io.wavfile
import matplotlib.pyplot as plt

from pydub import AudioSegment



DATA_DIR = "/home/nick/Desktop/zone/python_fun/projects/genre/data/"
FT_DIR = "/home/nick/Desktop/zone/python_fun/projects/genre/feature/"
TEST_FILE = "/home/nick/Desktop/zone/python_fun/projects/genre/data/blues/blues.00099.au.wav"
GENRE_DICT = {
              "blues"     : 1,
              "classical" : 2,
              "country"   : 3,
              "disco"     : 4,
              "hiphop"    : 5,
              "jazz"      : 6,
              "metal"     : 7,
              "pop"       : 8,
              "reggae"    : 9,
              "rock"      : 0
             }

def convert_dataset_to_wav():
    """
        Converts all files of the GTZAN dataset
        to the WAV (uncompressed) format.
        using customized Python Audio Tools
        http://audiotools.sourceforge.net/index.html
    """
    pass
    # start = timeit.default_timer()
    # rootdir = DATA_DIR
    # for subdir, dirs, files in os.walk(rootdir):
    #     for file in files:
    #         path = subdir+'/'+file
    #         if path.endswith("au"):
    #             song = AudioSegment.from_file(path,"mp3")
    #             song = song[:30000]
    #             song.export(path[:-2]+"wav",format='wav')

    # for subdir, dirs, files in os.walk(rootdir):
    #     for file in files:
    #         path = subdir+'/'+file
    #         if not path.endswith("wav"):
    #             os.remove(path)

    # stop = timeit.default_timer()
    # print("Conversion time = ", (stop - start))

def plot_time_domain(file):
    rate, X = scipy.io.wavfile.read(file)
    timp = len(X) / float(rate)
    t = np.linspace(0,timp,len(X))
    plt.plot(t,X)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.savefig(os.path.join('.', "confusion_matrix.png"), bbox_inches="tight")

#plot the timedomain features against original wave data
def show_feature(music, feature, timestamps, name):
    feature_data = music[feature]
    feature_x = timestamps * 100.0
    feature_y = (feature_data / np.max(feature_data))


    wave_x = ((np.arange(0, music['wavedata'].shape[0]).astype(np.float)) / music["sample_rate"]) * 100.0
    wave_y = (music['wavedata'] / np.max(music['wavedata']) / 2) - 0.5

    plt.plot(wave_x, wave_y, color = 'lightgrey')

    plt.plot(feature_x, feature_y, color = 'r')
    plt.savefig(os.path.join('.', "%s.png" % name), bbox_inches="tight")

def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0,1,freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    newspec = np.complex128(np.zeros([timebins, len(scale)]))

    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:, scale[i]:], axis=1)
        else:
            newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)

    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])

    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]

    return newspec, freqs

#short time fourier transform
def stft(data, window_size, overlap=0.5, window=np.hanning):
    data = np.int32(data)
    win = window(window_size)
    hop_size = int(window_size - np.floor(overlap * window_size))

    samples = np.append(np.zeros(np.floor(window_size/2.0)), data)
    #number of windows
    cols = np.ceil( (len(samples) - window_size) / float(hop_size)) + 1

    samples = np.append(samples, np.zeros(window_size))

    windows = np.lib.stride_tricks.as_strided(samples, shape=(cols, window_size), strides=(samples.strides[0]*hop_size, samples.strides[0])).copy()
    windows *= win

    return np.fft.rfft(windows)

def plot_stft(samples, sample_rate, binsize=1024):
    fourier = stft(samples, binsize)
    sshow, freq = logscale_spec(fourier, factor=1.0, sr=sample_rate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6)

    timebins, freqbins = np.shape(ims)
    fig, ax = plt.subplots(1,1,sharey=True, figsize=(15, 3.5))

    cax = ax.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap="jet", interpolation="none")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (hz)")
    ax.set_xlim([0, timebins-1])
    ax.set_ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    ax.set_xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/sample_rate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    ax.set_yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])
    plt.savefig(os.path.join('.', "stft.png"), bbox_inches="tight")

#load all the music file into source dict, or specific genre
def load_source(genre = None):
    source = {}

    #read the data into source
    start = timeit.default_timer()
    if genre:
        if genre in GENRE_DICT.keys():
            rootdir = DATA_DIR + "%s/" % genre
            source[genre] = []
        else:
            raise ValueError("could not find gengre %s in %r" % (genre, GENRE_LIST))
    else:
        rootdir = DATA_DIR
        for gen in GENRE_DICT.keys():
            source[gen] = []

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if subdir[-1] == '/':
                gen = subdir.split('/')[-2]
                path = subdir + file
            else:
                gen = subdir.split('/')[-1]
                path = subdir + '/' + file
            music = {'name' : file}
            music['sample_rate'], music['wavedata'] = scipy.io.wavfile.read(path)
            source[gen].append(music)
    end = timeit.default_timer()
    print("load all music takes ", (end - start))
    return(source)



if __name__ == "__main__":
    #convert_dataset_to_wav()
    plot_time_domain(TEST_FILE)

