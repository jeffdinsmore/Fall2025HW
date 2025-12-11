

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ---------------------------
# Motor parameters
# ---------------------------
R1 = 0.1
R2 = 0.15
L1 = 0.447e-3
L2 = 0.637e-3
Lm = 9.5e-3
Rm = np.inf
p = 4   # poles

# Frequencies to investigate
frequencies = [50, 60, 70, 80]

# Rated point for V/Hz control
VLL_rated = 220.0   # line-to-line voltage at 60 Hz
f_rated   = 60.0
V_per_Hz  = VLL_rated / f_rated    # ≈ 3.67 V/Hz

print("V/Hz ratio =", V_per_Hz)

# Slip range
s = np.linspace(0.001, 1.0, 2000)

# Storage for mechanical speed plotting
speed_curves_constant_VLL = {}

speed_curves_variable_VLL = {}

# --------------------------------------------------------
#  PLOT 1: TORQUE vs SLIP using constant V/Hz
# --------------------------------------------------------
plt.figure(figsize=(10,6))

def calculate(f):
    # Voltage scaled by V/Hz rule
    # Phase voltage
    V_phase1 = VLL_rated / np.sqrt(3)

    VLL = V_per_Hz * f
    V_phase2 = VLL / np.sqrt(3)

    w = 2*np.pi*f

    # Reactances
    X1 = w*L1
    X2 = w*L2
    Xm = w*Lm

    # Impedances
    Z1 = R1 + 1j*X1
    Zm = 1j*Xm
    Zth = Zm*Z1 / (Z1 + Zm)
    Vth1= V_phase1 * (Zm / (Z1 + Zm))
    Vth2 = V_phase2 * (Zm / (Z1 + Zm))

    # Synchronous speed
    omega_sync = 4*np.pi*f / p     # rad/s
    n_sync_rpm = 120*f / p         # rpm

    # Rotor impedance as function of slip
    Z2 = (R2/s) + 1j*X2

    # Rotor current
    I2_1 = Vth1 / (Zth + Z2)
    I2_2 = Vth2 / (Zth + Z2)

    # Torque expression
    P_gap1 = 3*(np.abs(I2_1)**2)*(R2/s)
    P_gap2 = 3*(np.abs(I2_2)**2)*(R2/s)
    T = P_gap1 / omega_sync
    T2 = P_gap2 / omega_sync
    n_mech = (1 - s) * n_sync_rpm
    return T, T2, n_mech, VLL

for f in frequencies:
    
    T, T2, n_mech, VLL= calculate(f)
    # Save torque-speed curve for second plot
    speed_curves_constant_VLL[f] = (n_mech, T)
    speed_curves_variable_VLL[f] = (n_mech, T2)

    # Plot torque vs slip
    plt.plot(s, T2, label=f"{f} Hz  (VLL={VLL:.1f}V)")

plt.xlabel("Slip (s)")
plt.ylabel("Torque (N·m)")
plt.title("Torque vs Slip with Constant V/Hz Control")
plt.grid(True)
plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------------
#  PLOT 2: TORQUE vs MECHANICAL SPEED (rpm)
# --------------------------------------------------------
plt.figure(figsize=(10,6))

for f in frequencies:
    n_mech, T = speed_curves_constant_VLL[f]
    plt.plot(n_mech, T2, label=f"{f} Hz")

plt.xlabel("Mechanical Speed (rpm)")
plt.ylabel("Torque (N·m)")
plt.title("Torque vs Mechanical Speed with Constant V/Hz Control")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%
