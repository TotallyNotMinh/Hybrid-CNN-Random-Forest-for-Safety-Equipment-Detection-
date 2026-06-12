import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

# ---------------------------------
# Binary image containing boundary
# 1 = boundary
# 0 = background
# ---------------------------------

A = np.array([
    [0,0,0,0,0,0,0],
    [0,1,1,1,1,1,0],
    [0,1,0,0,0,1,0],
    [0,1,0,0,0,1,0],
    [0,1,0,0,0,1,0],
    [0,1,1,1,1,1,0],
    [0,0,0,0,0,0,0]
], dtype=bool)

# complement
Ac = ~A

# seed point inside region
X = np.zeros_like(A, dtype=bool)
X[3,3] = True

# 4-connected structuring element
B = np.array([
    [0,1,0],
    [1,1,1],
    [0,1,0]
], dtype=bool)

steps = [X.copy()]

while True:
    X_new = binary_dilation(X, structure=B) & Ac

    steps.append(X_new.copy())

    if np.array_equal(X_new, X):
        break

    X = X_new

# ---------------------------------
# Display all iterations
# ---------------------------------

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

im = ax.imshow(steps[0], cmap='gray')
ax.axis('off')

def update(frame):
    im.set_data(steps[frame])
    ax.set_title(f"Iteration {frame}")
    return [im]

ani = FuncAnimation(
    fig,
    update,
    frames=len(steps),
    interval=800,
    blit=True
)

plt.show()