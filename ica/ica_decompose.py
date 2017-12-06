#=============================================================================
#
# Author: lujunliang Arron - j_ackson123@163.com
#
# QQ : 2817022753
#
# Last modified: 2017-12-06 12:30
#
# Filename: ica_decompose.py
#
# Description: 
#
#=============================================================================

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from scipy import signal

from sklearn.decomposition import FastICA, PCA

# #############################################################################
# Generate sample data
np.random.seed(0)
data = scio.loadmat('base_1.mat')
S1 = data['data'][:,1]
n_samples = S1.shape[0]
time = np.linspace(0, 8, n_samples)

# S2 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
S2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
# S_2 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal
# S2 += 0.2 * np.random.normal(size=S2.shape)  # Add noise

S = np.c_[S1, S2]


S /= S.std(axis=0)  # Standardize data

# Mix data
A = np.c_[np.ones([2, 1]), np.random.rand(2, 2)]  # Mixing matrix
X = np.dot(S, A)  # Generate observations

# Compute ICA
ica = FastICA(n_components=2)
S_ica = ica.fit_transform(X)

# #############################################################################
# Plot results

plt.figure()
# plot source signal
ax = plt.subplot(4, 1, 1)
ax.set_title('Source 1')
ax.plot(S1)
ax = plt.subplot(4, 1, 2)
ax.set_title('Source 1')
ax.plot(S2)
# plot mixing signal
ax = plt.subplot(4, 1, 3)
ax.set_title('Observations')
ax.plot(X)
# decomponent
ax = plt.subplot(4, 1, 4)
ax.set_title('ICA Result')
ax.plot(-1*S_ica)

plt.subplots_adjust(0.12, 0.09, 0.9, 0.9, 0.2, 0.81)
plt.show()