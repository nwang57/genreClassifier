#genre classification
==============================



###1. features

musical surface features:

    1. zero_crossing_rate (mean, std)
    2. root_mean_square (mean, std)
    3. spectual_centroid (mean, std)
    4. spectral_rolloff (mean, std)
    5. spectral_flux (mean, std)

###2. classification

KNN with 8 nearest neighbors and weights on distance achieves 0.4179 test accuracy (5 fold CV)
window_size = 512 => freq resolution = 22050/512 = 43 Hz

    "rock"      : 0
    "blues"     : 1,
    "classical" : 2,
    "country"   : 3,
    "disco"     : 4,
    "hiphop"    : 5,
    "jazz"      : 6,
    "metal"     : 7,
    "pop"       : 8,
    "reggae"    : 9,

        0  1  2  3  4  5  6  7  8  9
    0 [27  0  0  5 15  3  7 17 14 12]
    1 [ 0 73  1 19  2  3  1  1  0  0]
    2 [ 0  4 88  8  0  0  0  0  0  0]
    3 [10 17  3 41  6  1  4  5  4  9]
    4 [16  8  0  4 30  4  5 11 14  8]
    5 [ 2 15  1  6 16 38  1  9  6  6]
    6 [12  5  8  4  5  2 23  7 13 21]
    7 [13  2  0  2 10  6  6 26 20 15]
    8 [17  0  0  4  7  3  7 14 34 14]
    9 [16  0  0  4  8  2  9  8 15 38]


metal, pop , rock and reggae mess up together

window_size = 2048 => freq resolution = 10.7 Hz
k = 16
accuracy = 0.437

    [[48  1  0  7  3  2 13 11  5 10]
    [ 0 72  2 17  4  5  0  0  0  0]
    [ 0  3 88  9  0  0  0  0  0  0]
    [19 17  3 39  6  0  6  3  2  5]
    [11  8  0  4 27  3  7  9 17 14]
    [14 18  1  5 10 40  1  2  4  5]
    [25  4  8  8  2  1 20  6  6 20]
    [21  3  0  1  8  0  1 18 37 11]
    [ 3  0  0  2 11  0  4 21 45 14]
    [ 5  0  0  0  5  0 10 18 22 40]]


###Results:

scikits.talkbox only works for python2.X, so need to create a new virtual environment to compute the mfcc features and store them into the .npy file then switch to python3 for classification.

##SVM with linear kernal
For C: 0.010000, train_score=0.511500, test_score=0.463000
()
For C: 1.000000, train_score=0.496750, test_score=0.468000
()
For C: 5.000000, train_score=0.496750, test_score=0.460000
()
For C: 10.000000, train_score=0.493500, test_score=0.463000
()
#after normalized
For C: 1000000.000000, train_score=0.674500, test_score=0.476000

##Random Forest
For n_tree: 500.000000, train_score=0.999250, test_score=0.506000


    [[61  0  0  2  3  0 10 10  6  8]
    [ 0 90  1  6  0  3  0  0  0  0]
    [ 0  4 92  4  0  0  0  0  0  0]
    [17 12  1 43  2  2 10  4  4  5]
    [17  1  0  3 29  4  4 11 20 11]
    [13  9  1  4  8 49  5  3  4  4]
    [28  2  6  7  2  2 30  2  3 18]
    [24  0  0  0  2  0  1 21 44  8]
    [ 9  0  0  0  7  1  4 22 46 11]
    [ 5  0  0  1  3  1 10 14 21 45]]



##Neural Network

need `pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip --user`

input: 23
hidden: 200
output: 10

('train score: ', 0.4025)
('test score: ', 0.325)



TODO
1. derive avg time between spikes

