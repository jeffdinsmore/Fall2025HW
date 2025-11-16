import numpy as np
import matplotlib.pyplot as plt

# ----------Problem 2d---------------
# Coil 1
# --- Parameters ---
mu0 = 4 * np.pi * 1e-7     # H/m
N1  = 400                  # turns
g   = 0.0005               # m (air gap length)
w   = 0.01                 # m (translator width)
d   = 0.04                 # m (depth into page)

# Position from 0 to just under 10 mm (avoid division by zero at x = 0 and x = w)
x = np.linspace(1e-6, 0.01 * 0.999, 300)  # m
x_mm = x * 1000                           # for plotting

# --- Reluctances for coil 1 self-flux path ---

alpha = g / (mu0 * d)

# Left gap (above coil 1): depends on (w - x)
Rg1 = alpha / (w - x)

# Center gap: constant width w
Rgc = alpha / w

# Right gap (above coil 2): depends on x
Rg2 = alpha / x

# Parallel combo of center and right gaps
R_parallel = (Rgc * Rg2) / (Rgc + Rg2)

# Total reluctance seen by coil 1
R_total = Rg1 + R_parallel

# --- Self-inductance of coil 1 ---
L1 = N1**2 / R_total   # H

# --- Plot L1 vs position ---
plt.figure()
plt.plot(x_mm, L1)
plt.xlabel("Translator position x_L (mm)")
plt.ylabel("Self-inductance L1 (H)")
plt.title("Self-Inductance of Coil 1 vs Position")
plt.grid(True)
plt.tight_layout()

# --- Plot total reluctance vs position ---
plt.figure()
plt.plot(x_mm, R_total)
plt.xlabel("Translator position x_L (mm)")
plt.ylabel("Total reluctance R_total (A·turns/Wb)")
plt.title("Total Reluctance Seen by Coil 1 vs Position")
plt.yscale("log")  # log scale to show the steep rise near 10 mm
plt.grid(True)
plt.tight_layout()

plt.show()


# Coil 2



# ----------Problem 2e---------------
# Range of x_L from 0 to 10 mm
x_mm = np.linspace(0, 10, 200)     # mm
x_m = x_mm / 1000                  # convert mm → meters

# Force function: F = -80425 * x_L  (x_L in meters)
F = -80425 * x_m

# Plot
plt.figure()
plt.plot(x_mm, F)
plt.xlabel("x_L (mm)")
plt.ylabel("Force F (N)")
plt.title("Force vs. x_L for F = -80425·x_L")
plt.grid(True)

plt.show()
