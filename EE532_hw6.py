
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ----------Problem 2d---------------
# Coil 1
# --- Parameters ---
Pin = 125 * .746           # kW
Pinpu = Pin/Pin
R1  = 0.046                  # in pu
R2  = 0.023               #
X1   = 0.08                 #
X2   = 0.08                 #
Xm  = 2.3
Rm = 1e9
VL = 460
V1 = 1
VLpu = VL/VL
Vppu = 1
p = 4
wsync = 4 * np.pi * 60 / p
wsyncpu = wsync / wsync
we_pu = 1
s_vec = np.linspace(0.00849, .00852, 500)  # m

Zth = 1j*Xm * (R1 + 1j*X1) / (R1 + 1j*(X1 + Xm))
print(f"Zth = {Zth}\n")
Vth = Xm / (R1 + 1j*X1 + 1j*Xm) * Vppu
print(f"Vth = {Vth}\n")

s = R2 / np.abs(Zth)
print(f"s = {s}")

s2 = R2 / (X1 + X2)
print(f"S_Approximate = {s2}")
#Rth = Zth.real
Xth = Zth.imag
#Xth = X1
Rth = R1 * (Xm / (X1 + Xm))**2
Xe = Xth + X2

s3 = R2 / np.sqrt(Rth**2 + (Xth + X2)**2)
print(f"S_Exact = {s3}")

Tmax = np.abs(Vth)**2 / (2 * wsyncpu * (Rth + np.sqrt(Rth**2 + Xe**2)))
Tmax_appr = V1**2 / (2 * (X1 + X2))*we_pu
print(f"Torque max = {Tmax}")
print(f"Torque approximate pu = {Tmax_appr}")
Zseries = Rth + 1j*(Xth + X2)

Vphase = 460/np.sqrt(3)
Vphasepu = Vphase/VL
f = 60
P = 4
I1r = Pin / (3 * Vphase)
#Rth = R1 * (Xm/(X1+Xm))**2
#I2 = Xm / np.sqrt(R1**2+(X1+Xm)**2) / np.sqrt((Rth + R2/s)**2 + (X1 + X2)**2)

#Pag = 3 * I2**2 * R2/s
# Air-gap power function in PU
def Pag(s):
    #Z2 = (R2 / s) + 1j*X2
    #I2 = Vth / (Zth + Z2)
    Rth = R1 * (Xm/(X1+Xm))**2
    # Corrected line: Use s1 instead of global s
    I2 = Xm / np.sqrt(R1**2+(X1+Xm)**2) / np.sqrt((Rth + R2/s)**2 + (X1 + X2)**2)
    return 3 * (np.abs(I2)**2) * (R2 / s)

# Compute Pag over the slip vector (for plotting)
Pag_vals = np.array([Pag(si) for si in s_vec])


# Solve for s where Pag = 1 pu
def root_fn(s):
    return Pag(s) - 1

s_initial_guess = 0.03   # reasonable full-load slip guess
s_solution = fsolve(root_fn, s_initial_guess)[0]

print("Slip where Pag = 1 pu:")
print(f"s = {s_solution:.12f}")

# --- Plot Power air gap vs slip ---
"""plt.figure()
plt.plot(s_vec, Pag_vals, label="Power vs slip")
plt.xlabel("Slip")
plt.ylabel("Power air gap (pu)")
plt.title("Power Air Gap vs Slip")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()"""

# %%
