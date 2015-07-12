import os
import sys
import timeit
import numpy as np
import sunau
import wave
import matplotlib.pyplot as plt


DATA_DIR = "/home/nick/Desktop/zone/python_fun/projects/genre/data/"

def au2wav(in_file, out_file):
    au = sunau.open(in_file, 'r')
    wav = wave.open(out_file, 'w')

    wav.setnchannels(au.getnchannels())
    wav.setsampwidth(au.getsampwidth())
    wav.setframerate(au.getframerate())

    wav.writeframes(au.readframes(au.getnframes()))

    wav.close()
    au.close()

def convert_dataset_to_wav():
    """
        Converts all files of the GTZAN dataset
        to the WAV (uncompressed) format.
    """
    start = timeit.default_timer()
    rootdir = DATA_DIR
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            path = subdir+'/'+file
            if path.endswith("au"):
                out_path = path[:-2] + "wav"
                au2wav(path, out_path)

    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            path = subdir+'/'+file
            if not path.endswith("wav"):
                os.remove(path)

    stop = timeit.default_timer()
    print("Conversion time = ", (stop - start))

if __name__ == "__main__":
    #convert_dataset_to_wav()
    a=np.array([1,4,5])
    plt.plot(a)
    plt.ylabel('some numbers')
    plt.show()
    #plt.savefig(os.path.join('.', "confusion_matrix.png"), bbox_inches="tight")

