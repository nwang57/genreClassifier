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
