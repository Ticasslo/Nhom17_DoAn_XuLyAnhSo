import numpy as np
import matplotlib.pyplot as plt

def heart_wave(k=11):
    x = np.linspace(-np.sqrt(3.3), np.sqrt(3.3), 8000)
    a = np.abs(x) ** (2 / 3)
    b = 0.9 * np.sqrt(np.clip(3.3 - x**2, 0, None))
    fx = a + b * np.sin(k * np.pi * x)
    return x, fx

x, fx = heart_wave()

plt.figure(figsize=(7, 7))
ax = plt.gca()
ax.plot(x, fx, color="orange", linewidth=1.8)
ax.set_xlim(-2, 2)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect("equal")
ax.grid(True, linestyle="--", alpha=0.5)

plt.show()