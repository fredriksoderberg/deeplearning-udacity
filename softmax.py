"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    sum = 0
    y = np.copy(x)
    for i in np.nditer(y):
        sum += np.exp(i)
    for j in np.nditer(y, op_flags=['readwrite']):
        j[...] = np.exp(j) / sum
    return y
    
print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
