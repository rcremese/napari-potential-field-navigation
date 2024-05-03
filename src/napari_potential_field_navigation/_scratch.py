"""Physarum simulation example.

See https://sagejenson.com/physarum for the details."""

import numpy as np
from pathlib import Path

path = Path("paper_solutions/test_APF.npz").resolve(strict=True)

with np.load(path) as data:
    print(data.keys())
    print(data["method"])
    print(data["image"].shape)
    print(data["goal"])
    print(data["init_positions"])
    print(data["scalar_field"].shape)


def calculate_curvature(x, y):
    # Calculate first and second derivatives using finite differences
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    # Calculate curvature using the formula: curvature = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    numerator = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2)
    denominator = (dx_dt**2 + dy_dt**2) ** (3 / 2)
    curvature = numerator / denominator

    return curvature


# Example usage
t = np.linspace(0, 2 * np.pi, 100)  # Time array
x = np.sin(t)  # X coordinates of the trajectory
y = np.cos(t)  # Y coordinates of the trajectory

curvature = calculate_curvature(x, y)

# Plot the curvature
import matplotlib.pyplot as plt

plt.plot(x, y)
plt.plot(t, curvature)
plt.xlabel("Time")
plt.ylabel("Curvature")
plt.title("Curvature of the trajectory")
plt.show()
