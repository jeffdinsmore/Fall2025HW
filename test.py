
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.figure(figsize=(10,6))


# --- core torque computation for a single VLL, f ---
def torque_vs_slip_single(VLL, f, p=4,
                          R1=0.1, R2=0.15,
                          L1=0.447e-3, L2=0.637e-3, Lm=9.5e-3,
                          Rm=np.inf,
                          s_min=0.001, s_max=1.0, num_points=2000):
    """
    Compute torque vs slip for one voltage + frequency.
    Returns (s, T) where s is slip array, T is torque array [N·m].
    """

    torque_data = {}   # store curves for second plot

    # slip vector
    s = np.linspace(s_min, s_max, num_points)

    # per-phase voltage (Y connection assumed)
    V_phase = VLL / np.sqrt(3)

    w = 2 * np.pi * f

    # Reactances
    X1 = w * L1
    X2 = w * L2
    Xm = w * Lm

    # Stator and magnetizing branch
    Z1 = R1 + 1j * X1
    Zm = 1j * Xm if np.isinf(Rm) else 1 / (1 / (1j * Xm) + 1 / Rm)

    # Thevenin equivalent
    Z_th = Zm * Z1 / (Z1 + Zm)
    V_th = V_phase * (Zm / (Z1 + Zm))

    # Synchronous mechanical speed (rad/s)
    omega_sync = 4 * np.pi * f / p

    # Rotor branch impedance vs slip
    Z2_s = (R2 / s) + 1j * X2

    # Rotor current
    I2 = V_th / (Z_th + Z2_s)

    # Mechanical power and torque
    P_mech = 3 * (np.abs(I2) ** 2) * (R2 / s)
    T = P_mech / omega_sync

    
    return s, T, torque_data, n_mech


def torque_slip_vf_control():
    # Motor + control settings
    p = 4
    R1 = 0.1
    R2 = 0.15
    L1 = 0.447e-3
    L2 = 0.637e-3
    Lm = 9.5e-3
    Rm = np.inf

    f_list = [50, 60, 70, 80]

    # Rated point: 220 V at 60 Hz
    rated_VLL = 220.0
    rated_f = 60.0
    V_per_Hz = rated_VLL / rated_f  # ≈ 3.67 V/Hz

    print(f"Using constant V/Hz control: V/f = {V_per_Hz:.3f} V/Hz")

    plt.figure(figsize=(10, 6))

    for f in f_list:
        # Scale voltage with frequency to keep V/f constant
        VLL_f = V_per_Hz * f

        s, T, torque_data, n_mech = torque_vs_slip_single(
            VLL=VLL_f,
            f=f,
            p=p,
            R1=R1, R2=R2,
            L1=L1, L2=L2, Lm=Lm,
            Rm=Rm,
            s_min=0.001,
            s_max=1.0,
            num_points=2000
        )

        plt.plot(s, T, label=f"{f} Hz, VLL≈{VLL_f:.1f} V")

    plt.xlabel("Slip, s")
    plt.ylabel("Torque [N·m]")
    plt.title("Torque–Slip with Constant V/Hz Control (Approx. Constant Torque Capability)")
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.legend()
    plt.tight_layout()

    for f in f_list:
        n_mech, T = torque_data[f]
        plt.plot(n_mech, T, label=f"{f} Hz")

      # --- PLOT 2: TORQUE VS MECHANICAL SPEED (RPM) ---
    plt.figure(figsize=(10,6))
    plt.xlabel("Mechanical Speed (rpm)")
    plt.ylabel("Torque (N·m)")
    plt.title("Torque vs Mechanical Speed at Multiple Frequencies")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


    plt.show()


if __name__ == "__main__":
    torque_slip_vf_control()

# %%
