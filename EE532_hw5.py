import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# 1) Define angle grid over one electrical period (0–360°)
# --------------------------------------------------------------------
N = 2000  # number of points (fine resolution)
theta_deg = np.linspace(0, 360, N, endpoint=False)
theta_rad = np.deg2rad(theta_deg)

# --------------------------------------------------------------------
# 2) Build TURNS functions n_a, n_b, n_c from your dot/X positions
#    Phase a: dot at 0°, X at 180°  -> 1 from 0–180, 0 from 180–360
#    Phase b: dot at 120°, X at 300° -> 1 from 120–300, 0 elsewhere
#    Phase c: X at 60°, dot at 240° -> -1 from 60–240, 0 elsewhere
# --------------------------------------------------------------------

# Helper: rectangular window 1 on [start, end), 0 elsewhere (in degrees)
def rect_deg(theta, start, end):
    """Return 1 where start <= theta < end (mod 360), else 0."""
    # handles wrap-around if end < start
    if start < end:
        return np.where((theta >= start) & (theta < end), 1.0, 0.0)
    else:
        return np.where((theta >= start) | (theta < end), 1.0, 0.0)

# Phase a turns
n_a = rect_deg(theta_deg, 0, 180)   # 1 from 0–180, 0 from 180–360

# Phase b turns (shifted by 120°)
n_b = rect_deg(theta_deg, 120, 300)

# Phase c turns (negative between 60–240)
n_c = -rect_deg(theta_deg, 60, 240)

# --------------------------------------------------------------------
# 3) Winding functions w = n - average(n)  (remove DC component)
# --------------------------------------------------------------------
w_a = n_a - np.mean(n_a)
w_b = n_b - np.mean(n_b)
w_c = n_c - np.mean(n_c)

print("Phase A average (should be ~0):", np.mean(w_a))
print("Phase B average (should be ~0):", np.mean(w_b))
print("Phase C average (should be ~0):", np.mean(w_c))

# --------------------------------------------------------------------
# 4) Compute the fundamental (1st harmonic) using a discrete Fourier series
#    w(theta) ≈ c1 e^{jθ} + c1* e^{-jθ} = 2 Re{ c1 e^{jθ} }
#    with c1 = (1/N) ∑ w(θ_k) e^{-j θ_k}
# --------------------------------------------------------------------

def fundamental_component(w, theta_rad):
    Npts = len(theta_rad)
    # complex Fourier coefficient for n = 1
    c1 = np.sum(w * np.exp(-1j * theta_rad)) / Npts
    # reconstruct fundamental as real function
    w1 = 2 * np.real(c1 * np.exp(1j * theta_rad))
    amp = 2 * np.abs(c1)  # amplitude of the fundamental
    return w1, amp, c1

w1_a, amp_a, c1_a = fundamental_component(w_a, theta_rad)
w1_b, amp_b, c1_b = fundamental_component(w_b, theta_rad)
w1_c, amp_c, c1_c = fundamental_component(w_c, theta_rad)

print("Fundamental amplitude |W1_a| ≈", amp_a, " (theoretical 2/pi ≈", 2/np.pi, ")")
print("Fundamental amplitude |W1_b| ≈", amp_b)
print("Fundamental amplitude |W1_c| ≈", amp_c)

# --------------------------------------------------------------------
# 5) Plot ONLY the fundamental of each phase
# --------------------------------------------------------------------
plt.figure(figsize=(10, 6))

plt.plot(theta_deg, w1_a, label="Phase A fundamental")
plt.plot(theta_deg, w1_b, label="Phase B fundamental")
plt.plot(theta_deg, w1_c, label="Phase C fundamental")

plt.xlabel("Electrical angle θ (degrees)")
plt.ylabel("Winding function fundamental w₁(θ)")
plt.title("Fundamental (1st harmonic) of stator phase winding functions")
plt.grid(True)
plt.legend()
plt.xlim(0, 360)

plt.tight_layout()
plt.show()
