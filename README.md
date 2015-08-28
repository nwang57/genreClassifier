#genre classification
==============================



###1. features

musical surface features:

    1. zero_crossing_rate (mean, std)
    2. root_mean_square (mean, std)
    3. spectual_centroid (mean, std)
    4. spectral_rolloff (mean, std)
    5. spectral_flux (mean, std)

[4.83105943e-01,   1.84310733e+04,   1.24440400e+02,   9.00677659e+03,   7.94717232e+03,   2.77841531e-02,   4.49269394e+02,   7.41860025e+00,   4.18480207e+02,   4.06216013e+02]
[7.90827566e-02,   4.34688333e+03,   1.57038219e+01,   1.08481008e+03,   2.39471207e+03,   4.53308749e-02,   1.98990870e+03,   1.01553804e+01,   9.57025983e+02,   1.26527589e+03]
[7.77188590e-02,   1.39098817e+03,   1.70829439e+01,   1.08835124e+03,   7.86361410e+02,   2.96035048e-02,   7.83475502e+02,   6.49370947e+00,   5.08584650e+02,   4.71072649e+02]
[2.46012995e-01,   1.04139634e+04,   6.04769071e+01,   4.35541927e+03,   4.79234636e+03,   3.68710289e-02,   1.13152848e+03,   8.42975050e+00,   6.90911633e+02,   7.86895641e+02]
[3.87325963e-01,   1.46280435e+04,   9.72035814e+01,   7.06493468e+03,   6.43438790e+03,   4.15238878e-02,   9.38244313e+02,   1.07006811e+01,   8.15919922e+02,   6.63249954e+02]
[2.32178816e-01,   1.01488648e+04,   5.55646449e+01,   4.02808082e+03,   4.79594947e+03,   6.63507203e-02,   2.35324535e+03,   1.63761152e+01,   1.34154998e+03,   1.46177247e+03]
[4.12836607e-01,   1.62220613e+04,   1.06573290e+02,   7.71836447e+03,   7.07211771e+03,   3.84887018e-02,   5.98884286e+02,   8.77042654e+00,   5.20141043e+02,   4.91233260e+02]
[4.79226371e-01,   1.79608841e+04,   1.23118557e+02,   8.91401789e+03,   7.72826768e+03,   2.52572928e-02,   4.18417003e+02,   7.17436373e+00,   4.01023738e+02,   3.81576788e+02]
[4.99011171e-01,   1.89062184e+04,   1.28813098e+02,   9.31239340e+03,   8.12395226e+03,   2.42741962e-02,   3.96582308e+02,   6.94055221e+00,   3.68046444e+02,   3.71607870e+02]
[4.91632821e-01,   1.88563064e+04,   1.27501301e+02,   9.24380527e+03,   8.09656730e+03,   3.63207609e-02,   4.69776601e+02,   8.82770524e+00,   4.75494623e+02,   4.09178053e+02]


###2. classification

KNN with 8 nearest neighbors and weights on distance achieves 0.4179 test accuracy (5 fold CV)
window_size = 512 => freq resolution = 22050/512 = 43 Hz
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


###Note:

scikits.talkbox only works for python2.X, so need to create a new virtual environment to compute the mfcc features and store them into the .npy file then switch to python3 for classification.

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

###Random Forest
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
1. debug nan and inf
`>>> np.sum(np.isinf(X),axis=0)`
`array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])`
`>>> np.sum(np.isnan(X),axis=0)`
`array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])`

2. derive avg time between spikes

