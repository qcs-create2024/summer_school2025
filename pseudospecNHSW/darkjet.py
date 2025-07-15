import numpy as np

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

darkjet32 = np.array([0,        0,   0.6250,
        0,        0,   0.7499,
        0,        0,   0.8749,
        0,        0,   0.9997,
        0,   0.1249,   0.9994,
        0,   0.2497,   0.9989,
        0,   0.3742,   0.9978,
        0,   0.4979,   0.9958,
        0,   0.6201,   0.9921,
        0,   0.7387,   0.9849,
        0,   0.8498,   0.9713,
        0,   0.9453,   0.9453,
   0.1121,   0.8966,   0.7845,
   0.2021,   0.8083,   0.6062,
   0.2505,   0.6680,   0.4175,
   0.2625,   0.5249,   0.2625,
   0.3281,   0.5249,   0.1968,
   0.5010,   0.6680,   0.1670,
   0.7073,   0.8083,   0.1010,
   0.8966,   0.8966,        0,
   0.9453,   0.8272,        0,
   0.9713,   0.7284,        0,
   0.9849,   0.6156,        0,
   0.9921,   0.4960,        0,
   0.9958,   0.3734,        0,
   0.9978,   0.2495,        0,
   0.9989,   0.1249,        0,
   0.9994,        0,        0,
   0.8747,        0,        0,
   0.7499,        0,        0,
   0.6249,        0,        0,
   0.5000,        0,        0
], dtype=np.float64).reshape((32, 3))
cmap_list = []
for row in range(32):
    cmap_list.append(darkjet32[row, :])

darkjet = LinearSegmentedColormap.from_list('darkjet', cmap_list, 256)
mpl.colormaps.register(darkjet)
