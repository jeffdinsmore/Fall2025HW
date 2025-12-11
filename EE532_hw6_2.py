"""
    Equations used are from the lecture notes and the book "Electric
    Machinery and Power System Fundamentals" by Stephen J. Chapman

    Created: 12/03/2025

    Author: Jeff Dinsmore

    Results:
    Voltage Hertz ratio = 4.400 for 50Hz

    Voltage Hertz ratio = 3.667 for 60Hz

    Voltage Hertz ratio = 3.143 for 70Hz

    Voltage Hertz ratio = 2.750 for 80Hz

"""


# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from matplotlib.ticker import MultipleLocator

def torque_slip_curve(
    V_LL = 220,              # V
    p = 4,
    f_list = [50, 60, 70, 80],    # Hz
    R1 = 0.1,                # Ohms
    R2 = 0.15,
    L1 = 0.447,              # mH
    L2 = 0.637,
    Lm = 9.5,
    Rm = np.inf,
    s_min=0.00001,
    s_max=1.0,
    num_points=1000,
):
    plt.figure(figsize=(10,6))

    torque_data = {}   # store curves for second plot

    for f in f_list:
        
        # Phase voltage
        V_phase = V_LL / np.sqrt(3)

        # Angular frequency
        w = 2 * np.pi * f

        # Reactances
        X1 = w * L1
        X2 = w * L2
        Xm = w * Lm

        # Stator impedance
        Z1 = R1 + 1j * X1

        # Magnetizing branch impedance (with Rm if finite)
        if np.isinf(Rm):
            Zm = 1j * Xm
        else:
            Zm = 1 / (1 / (1j * Xm) + 1 / Rm)

        # Thevenin equivalent seen from air gap
        Z_th = Zm * Z1 / (Z1 + Zm)
        V_th = V_phase * (Zm / (Z1 + Zm))

        # Synchronous mechanical speed (rad/s)
        # nsync (rpm) = 120 f / p  ->  omega_sync = 2π nsync / 60
        # Simplifies to: omega_sync = 4π f / p
        omega_sync = 4 * np.pi * f / p

        # Slip vector (avoid s = 0)
        s = np.linspace(s_min, s_max, num_points)

        # Rotor branch impedance as a function of slip
        Z2_s = (R2 / s) + 1j * X2

        # Rotor current from Thevenin source
        I2 = V_th / (Z_th + Z2_s)

        # Electromagnetic torque:
        # Air-gap power Pag = 3 |I2|^2 (R2 / s)
        # Mechanical power (neglecting losses) Pmech = Pag * (1 - s)
        # Torque T = Pmech / omega_sync
        P_gap = 3 * (np.abs(I2) ** 2) * (R2 / s)
        T = P_gap / omega_sync  # N·m

        # save for second plot
        n_sync = 120 * f / p
        n_mech = (1 - s) * n_sync
        torque_data[f] = (n_mech, T)

        # Plot ON SAME AXES
        plt.plot(s, T, label=f"{f} Hz")
    
    # Voltage / Hz ratio for all frequencies
    for f in f_list:
        print(f"Voltage Hertz ratio = {V_LL / f:.2f} for {f}Hz\n")

    # Plot
    plt.xlabel("Slip, s")
    plt.ylabel("Torque, T [N·m]")
    plt.title("Torque–Slip Curve for 3-Phase Induction Motor")
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.xticks(np.arange(0, 1.1, 0.1))   # <--- sets ticks every 0.1
    plt.legend()
    plt.tight_layout()

        # --- PLOT 2: TORQUE VS MECHANICAL SPEED (RPM) ---
    plt.figure(figsize=(10,6))

    for f in f_list:
        n_mech, T = torque_data[f]
        plt.plot(n_mech, T, label=f"{f} Hz")

    plt.xlabel("Mechanical Speed (rpm)")
    plt.ylabel("Torque (N·m)")
    plt.title("Torque vs Mechanical Speed at Multiple Frequencies")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()

    return s, T


if __name__ == "__main__":
    # Your specific motor:
    # R1 = 0.1 Ω, R2 = 0.15 Ω
    # L1 = 0.447 mH, L2 = 0.637 mH, Lm = 9.5 mH
    torque_slip_curve(
        V_LL=220.0,
        f_list=[50, 60, 70, 80],
        p=4,
        R1=0.1,
        R2=0.15,
        L1=0.447e-3,
        L2=0.637e-3,
        Lm=9.5e-3,
        Rm=np.inf,
    )

# %%
