
"""
    Equations used are from the lecture notes and the book "Electric
    Machinery and Power System Fundamentals" by Stephen J. Chapman

    Created: 12/03/2025

    Author: Jeff Dinsmore

    Results:
    s_max_appr = 0.14375

    s_max_exact = 0.14035

    Tmax_exact pu= 2.257 pu
    Tmax_appr pu = 3.125 pu

    Rated slip where Pag = 1 pu:
    s = 0.028258486997

    Tbase = 508.70 Nm, Sbase = 95887.04 VA, wsync = 188.50 rad/s, 
    Ibase = 208.45 A, Pconv_actual = 93250.00 W

    I_start_pu_appr = 6.186409 pu, Tstart_pu_appr = 0.898437 pu,
    I_actual_appr = 1289.6 A, T_actual_appr = 457.0 Nm

    I_start_pu_exact = 5.638941 pu, T_pu_exact = 0.731346 pu,
    I_actual_exact = 1175.4 A, T_actual_exact = 372.0 Nm

    Stator Losses - I1_pu = 1.212701 pu, P_scl_pu = 0.067650 pu, P_scl_actual = 6308.3 W
    Rotor Losses - I2_pu = 1.108867 pu, P_rcl_pu = 0.028281 pu, P_rcl_actual = 2637.2 W

    Efficiency = 91.25%
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# ----------HW 6 Problem 1---------------

# --- Parameters ---
Pout = 125 * .746           # kW
Pout_pu = Pout/Pout
R1  = 0.046                  # in pu
R2  = 0.023               #
X1   = 0.08                 #
X2   = 0.08                 #
Xm  = 2.3
Rm = 1e9
VLL = 460
V_base = VLL
V1 = 1
VLpu = VLL/V_base
Vphase_pu = 1
p = 4
wsync = 4 * np.pi * 60 / p
wsyncpu = wsync / wsync
we_pu = 1

# Used to find rated slip
s_vec = np.linspace(0.000000001, 1, 500)
# Used to calculate Torque and current
s3_vec = np.linspace(0.000000001, 1, 400)

# Zthevenin exact value
Zth = 1j*Xm * (R1 + 1j*X1) / (R1 + 1j*(X1 + Xm))

# Vthevenin exact value
Vth = Xm / (R1 + 1j*X1 + 1j*Xm) * Vphase_pu

# slip approximated equation
s2 = R2 / (X1 + X2)
print(f"s_max_appr = {s2:.5f}\n")

# Xthevenin value from Zth imaginary
Xth = Zth.imag
# Rthevnin
Rth = R1 * (Xm / (X1 + Xm))**2
Rth_chat = Xm**2*R1/(R1**2 + (X1+Xm)**2)
print(f"Rth from book = {Rth}, Rth from Chat = {Rth_chat}")
# Xe is a shortcut to the torque max equation as to not write out everything
Xe = Xth + X2

# Max torque slip value
s3 = R2 / np.sqrt(Rth**2 + (Xth + X2)**2)
print(f"s_max_exact = {s3:.5f}\n")

# Torque max exact value in pu
Tmax = np.abs(Vth)**2 / (2 * wsyncpu * (Rth + np.sqrt(Rth**2 + Xe**2)))

# Torque max approximate value in pu
Tmax_appr = V1**2 / (2 * (X1 + X2))*we_pu
print(f"Tmax_exact pu= {Tmax:.3f} pu")
print(f"Tmax_appr pu = {Tmax_appr:.3f} pu\n")

# Air-gap power function in PU
def Pag(s):
    # Rthevenin
    Rth = R1 * (Xm/(X1+Xm))**2
    # Rotor current
    I2 = Xm / np.sqrt(R1**2+(X1+Xm)**2) / np.sqrt((Rth + R2/s)**2 + (X1 + X2)**2)
    return (np.abs(I2)**2) * (R2 / s)

# Solve for s where Pag = 1 pu
def root_fn(s):
    return Pag(s) - 1

# reasonable full-load slip guess
s_initial_guess = 0.03   
# Solution for s_rated
s_solution = fsolve(root_fn, s_initial_guess)[0]

print("Rated slip where Pag = 1 pu:")
print(f"s = {s_solution:.12f}")

#------------------- HW 6 - Problem 1b ----------------------------------
def Tind(s):
    # Rthevenin value in pu
    Rth = R1 * (Xm/(X1+Xm))**2
    # Rotor current exact value in pu
    I2_pu = np.abs(Vth / np.sqrt((Rth + R2/s)**2 + (Xth + X2)**2))
    # Air gap power in pu
    Pag_exact_pu = I2_pu**2 * R2 / s
    # Torque induced exact value in pu
    Tind_pu = Pag_exact_pu / wsyncpu
    return Tind_pu, I2_pu

# Compute Torque and Rotor Current over the slip vector (for plotting)
Tind_vals, I2_vals = Tind(s3_vec)

# ------- For computing Ibase and Tbase ----------
# Rated Rotor current exact value in pu
I2_pu = np.abs(Vth / np.sqrt((Rth + R2/s_solution)**2 + (Xth + X2)**2))
# Rated air gap power in pu
Pag_exact_pu = I2_pu**2 * R2 / s_solution
# Rated Pconvert power in pu
Pconv_pu = (1-s_solution) * Pag_exact_pu
# Pconvert actual value when not counting for mechanical losses
Pconv_actual = Pout
# Sbase apparent power calculated from actual power / power in pu
S_base = Pconv_actual * 1000 / Pconv_pu
# Real power base
P_base = Pout * 1000 
# Torque base to find actual torque
T_base = S_base / wsync        
# Torque induced in pu
Tind_pu = Pag_exact_pu / wsyncpu
# Current base derived from S_base
I_base = S_base / V_base
print(f"\nTbase = {T_base:.2f} Nm, Sbase = {S_base:.2f} VA, wsync = {wsync:.2f} rad/s, \nIbase = {I_base:.2f} A, Pconv_actual = {Pconv_actual*1000:.2f} W")

# Function to find actual torque induced values as a function of s
def Tind_actual(s):
    # Rthevenin
    Rth = R1 * (Xm/(X1+Xm))**2
    # Rotor Current from S. J. Chapman Book
    I2_pu = np.abs(Vth / np.sqrt((Rth + R2/s)**2 + (Xth + X2)**2))
    # Air Gap power without the 3 in the equation to get p.u.
    Pag_exact_pu = I2_pu**2 * R2 / s
    # Torque induced over a range of s
    Tind_value = Pag_exact_pu / wsyncpu
    return Tind_value * T_base, I2_pu * I_base

Tind_actual_vals, I2_actual_vals = Tind_actual(s3_vec)



# --- Plot Current and torque in pu vs slip ---
plt.figure()
plt.plot(s3_vec, Tind_vals, label="Torque vs slip")
plt.plot(s3_vec, I2_vals, label="Rotor Current vs slip")
plt.xlabel("Slip")
plt.ylabel("Torque Induced & Rotor Current (pu)")
plt.title("Torque Induced & Rotor Current vs Slip")
plt.legend()
plt.grid(True)
plt.tight_layout()

# --- Plot current and torque actual values vs slip ---
plt.figure()
plt.plot(s3_vec, Tind_actual_vals, label="Torque vs slip")
plt.plot(s3_vec, I2_actual_vals, label="Rotor Current vs slip")
plt.xlabel("Slip")
plt.ylabel("Torque Induced (Nm) & Rotor Current (A)")
plt.title("Torque Induced & Rotor Current vs Slip")
plt.legend()
plt.grid(True)
plt.tight_layout()
#------------------- end Problem 1b -----------------------------------


#----------------------- Problem 1c -----------------------------------
slip = 1

# Starting current and torque exact values in pu
I_start_exact_pu = np.abs(Vth / np.sqrt((Rth + R2/slip)**2 + (Xth + X2)**2))
T_start_exact_pu = I_start_exact_pu**2 * R2 / slip

# Starting current and torque exact actual values
I_actual_exact = I_start_exact_pu * I_base
T_actual_exact = T_start_exact_pu * T_base

# Starting current and torque approximate values in pu
I_start_appr_pu = np.abs(V1 / (R2 / slip + 1j * (X1 + X2)))
#T_start_appr_pu = I_start_appr_pu**2 * R2 / slip
T_start_appr_pu = V1**2 * R2 / (X1 + X2)**2

# Starting current and torque approximate actual values
I_actual_appr = I_start_appr_pu * I_base
T_actual_appr = T_start_appr_pu * T_base

print(f"\nI_start_pu_appr = {I_start_appr_pu:.6f} pu, Tstart_pu_appr = {T_start_appr_pu:.6f} pu,\nI_actual_appr = {I_actual_appr:.1f} A, T_actual_appr = {T_actual_appr:.1f} Nm\n")
print(f"I_start_pu_exact = {I_start_exact_pu:.6f} pu, T_pu_exact = {T_start_exact_pu:.6f} pu,\nI_actual_exact = {I_actual_exact:.1f} A, T_actual_exact = {T_actual_exact:.1f} Nm")

#------------------- end Problem 1c -----------------------------------


#----------------------- Problem 1d -----------------------------------
# ------- Stator losses, NOT including core losses
Z_tot = R1 + 1j * X1 + (1 / (1j*Xm) + 1 / (1j*X2 + R2/s_solution))**(-1)
I1 = np.abs(VLpu / Z_tot)
P_scl_pu = I1**2 * R1
P_scl_actual = P_scl_pu * P_base

# ------- Rotor losses, NOT including friction and windage losses
I2 = np.abs(Vth / np.sqrt((Rth + R2/s_solution)**2 + (Xth + X2)**2))
P_rcl_pu = I2**2 * R2
P_rcl_actual = P_rcl_pu * P_base

print(f"\nStator Losses - I1_pu = {I1:.6f} pu, P_scl_pu = {P_scl_pu:.6f} pu, P_scl_actual = {P_scl_actual:.1f} W")
print(f"Rotor Losses - I2_pu = {I2:.6f} pu, P_rcl_pu = {P_rcl_pu:.6f} pu, P_rcl_actual = {P_rcl_actual:.1f} W\n")

# ------- Calculate efficiency
Pin = Pout * 1000 + P_scl_actual + P_rcl_actual
Eff = Pout * 1000 / Pin * 100

print(f"Efficiency = {Eff:.2f}%")
#------------------- end Problem 1d -----------------------------------

plt.show()

# %%
