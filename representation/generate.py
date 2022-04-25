import numpy as np
from matplotlib import pyplot as plt

skeleton_data = np.load("a1_s1_t1_skeleton.npy")
t = np.reshape(skeleton_data, (skeleton_data.shape[0], skeleton_data.shape[2], 3))
print("Old shape %s, New shape %s"%(skeleton_data.shape, t.shape))
# Normalization
t -= np.min(t)
t /= np.max(t)
# Displaying
plt.imshow(t, interpolation='nearest')
plt.show()
