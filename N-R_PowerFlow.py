"""
NR Power Flow: 3-bus system
Buses: 1=Slack, 2=PQ (load), 3=PV (gen)
Given: V1=1.0∠0, PD2=0.9, QD2=0.5, PG3=1.3, V3=1.01
Lines: Z12=j0.1, Z13=j0.25, Z23=j0.2
Epsilon=0.01, k=0
Flat start: σ1=0, σ2=0, V2=1

---Need to update this to handle n buses

Author: Jeff Dinsmore
"""

import numpy as np
import math
import cmath
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from pprint import pprint as pp

# --- tolerances / iteration ---
EPS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13]  # mismatch tolerance (pu)
#k = 0       # NR iteration counter
MAX_ITERS = 20


SLACK, PQ, PV = 0, 1, 2

def load_system_file(filename):
    """
    Parse the FiveBus_PQ-style file and build the buses dict.

    Returns:
        baseMVA
        buses: {bus_num: {...}}
        bus_name_to_num: {"Alan":1, "Betty":2, ...}
    """
    buses = {}
    bus_name_to_num = {}
    baseMVA = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue

            parts = line.split()

            # SYSTEM  FiveBus_PQ   100   0.05
            if parts[0].upper() == "SYSTEM":
                baseMVA = float(parts[2])

            # BUS  name  type  volts  Pgen  Qgen  Pload  Qload  Qcap
            elif parts[0].upper() == "BUS":
                _, name, btype, V, PG, QG, Pload, Qload, Qcap = parts

                bus_num = len(buses) + 1
                bus_name_to_num[name] = bus_num

                if btype.upper() == "SL":
                    btype_code = SLACK
                elif btype.upper() == "PQ":
                    btype_code = PQ
                elif btype.upper() == "PV":
                    btype_code = PV
                else:
                    raise ValueError(f"Unknown bus type {btype} for bus {name}")

                V = float(V)

                # convert MW/MVAr to per-unit on baseMVA
                PG = float(PG) / baseMVA
                QG = float(QG) / baseMVA
                PD = float(Pload) / baseMVA
                QD = float(Qload) / baseMVA
                # Qcap is ignored

                buses[bus_num] = {
                    "name": name,
                    "type": btype,
                    "V": V,
                    "δ": 0.0,
                    "PG": PG,
                    "QG": QG,
                    "PD": PD,
                    "QD": QD,
                }

    return baseMVA, buses, bus_name_to_num


# --- system data (per-unit) ---
baseMVA, buses, name_map = load_system_file("FiveBus_PQ")
#print(f"buses-----------------\n{buses}\n")


def load_line_data(filename, bus_name_to_num):
    """
    Parse LINE data from the system file and build the Z dict.

    Z[(i, j)] = R + jX   where i, j are *bus numbers* (1-based, ints)

    Assumes:
      - filename is like 'FiveBus_PQ'
      - bus_name_to_num maps bus names ('Alan') -> bus numbers (1, 2, ...)
      - Rse, Xse are already in per-unit
    """
    Z = {}

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue

            parts = line.split()

            if parts[0].upper() == "LINE":
                # LINE  from  to  Rse  Xse  Gsh  Bsh  Rating
                _, from_name, to_name, Rse, Xse, Gsh, Bsh, rating = parts

                i = bus_name_to_num[from_name]
                j = bus_name_to_num[to_name]

                R = float(Rse)
                X = float(Xse)

                # store only one direction; build_Ybus will handle symmetry
                if i > j:
                    i, j = j, i

                Z[(i, j)] = R + 1j * X

    return Z

# --- line impedances (pu) ---
Z = load_line_data("FiveBus_PQ", name_map)
#print(f"Z----------------------\n{Z1}\n")

# --- flat start values ---
delta = np.array([0.0, 0.0])   # σ2, σ3 (bus angles in radians excluding slack)
V = np.array([1.0, 1.01])      # V2, V3 magnitudes



# store symmetrically for convenience
#Z.update({(j, i): z for (i, j), z in list(Z.items())})

def degreeToRadians(degree):
    radians = np.pi * degree / 180
    return radians

# Y-Bus function ----------------------------------
def build_Ybus(Z, n=None):
    """
    Build an n x n Y-bus matrix from a dict of line impedances Z.

    Z keys are (i, j) bus-number pairs (1-based), e.g. (1,2), (2,3), etc.
    If n is not given, it is inferred from the largest bus number in Z.
    """
    # infer number of buses if not given
    if n is None:
        n = max(max(i, j) for (i, j) in Z.keys())

    Y = np.zeros((n, n), dtype=complex)

    # build Ybus from line data
    for (i, j), z in Z.items():
        if i == j:
            continue  # ignore self entries if any
        if i > j:
            continue  # handle each pair once (i < j)

        y = 1 / z
        i_idx = i - 1
        j_idx = j - 1

        # off-diagonal (mutual)
        Y[i_idx, j_idx] -= y
        Y[j_idx, i_idx] -= y

        # diagonal (self)
        Y[i_idx, i_idx] += y
        Y[j_idx, j_idx] += y

    return Y


#def tabulate_results(data):
 #   print(tabulate(data, headers='keys', tablefmt='grid'))

# build Y-bus -----------------------------------------------
Ybus = build_Ybus(Z)
#print(f"[INFO] Your Ybus matrix is: \n{Ybus}\n")
def display_Ybus(Ybus):
    rows = []
    for i in range(Ybus.shape[0]):
        row = []
        for j in range(Ybus.shape[1]):
            row.append(f"{Ybus[i, j].real:.4f} + j{Ybus[i, j].imag:.4f}")
        rows.append(row)

    print(tabulate(rows, tablefmt="grid"))

# extract diagonal elements ---------------------------------
Y_diag = np.diag(Ybus)
#print(f"[INFO] Your Y-diagonals are: {Y_diag}\n")

# Extract off-diagonal Y values (i ≠ j) ----------------------
off_diagonals = []
n = len(buses)
for i in range(n):
    for j in range(n):
        if i != j:
            y = Ybus[i, j]
            mag = abs(y)
            ang = np.angle(y)  # radians
            off_diagonals.append(((i+1, j+1), mag, ang))

# Convert to array for clarity (bus pair, magnitude, angle)
Y_off = np.array(off_diagonals, dtype=object)
#print(f"[INFO] Your Ybus off diagonal elements are: \n{Y_off}\n")

# convert to polar form (magnitude, angle in radians)
Y_polar_diag = np.array([(abs(y), np.angle(y)) for y in Y_diag])
#print(f"[INFO] Your Ybus polar diagonals are: \n{Y_polar_diag}\n")

Y_polar_off = np.array([(abs(y), np.angle(y)) for y in Y_diag])
#print(f"[INFO] Your Ybus polar off diagonals are: \n{Y_polar_off}\n")



# For mismatch matrix
class PowerVariables:
    """
    Holds all P and Q vectors needed for NR power flow.

    After computations, these attributes will be filled:
        P_spec : reduced P_spec (non-slack buses)
        Q_spec : reduced Q_spec (PQ buses)

        P_calc : reduced P_calc (non-slack buses)
        Q_calc : reduced Q_calc (PQ buses)

    Also stores the bus lists so functions know which
    array index corresponds to which bus in the system.
    """

    def __init__(self):
        # reduced arrays (start empty; will be set by methods)
        self.P_spec = None
        self.Q_spec = None
        self.P_calc = None
        self.Q_calc = None

        # store corresponding bus numbers
        self.non_slack_buses = None
        self.pq_buses = None
    # ---------------------------------------------------------------------------
    def build_spec_arrays(self, buses):
        """
        Build reduced P_spec and Q_spec arrays from the buses dict.

        - P_spec contains one entry per NON-SLACK bus
        - Q_spec contains one entry per PQ bus

        Stores results in:
            self.P_spec
            self.Q_spec
            self.non_slack_buses
            self.pq_buses
        """
        
        bus_nums = sorted(buses.keys())

        # identify bus types
        slack_bus = next(b for b in bus_nums if buses[b]["type"] == "SL")
        self.non_slack_buses = [b for b in bus_nums if b != slack_bus]
        self.pq_buses        = [b for b in bus_nums if buses[b]["type"] == "PQ"]

        # --- build reduced P_spec (PG − PD for non-slack buses) ---
        self.P_spec = []
        for b in self.non_slack_buses:
            data = buses[b]
            PG = data.get("PG", 0.0)
            PD = data.get("PD", 0.0)
            self.P_spec.append(PG - PD)
        self.P_spec = np.array(self.P_spec, dtype=float)

        # --- build reduced Q_spec (QG − QD for PQ buses only) ---
        self.Q_spec = []
        for b in self.pq_buses:
            data = buses[b]
            QG = data.get("QG", 0.0)
            QD = data.get("QD", 0.0)
            self.Q_spec.append(QG - QD)
        self.Q_spec = np.array(self.Q_spec, dtype=float)

    # Setting up calculations for mismatch matrix ----------------------------------------
    def build_calc_arrays(self, buses, Ybus):
        """
        Compute reduced P_calc and Q_calc from Ybus and current bus voltages.

        - P_calc matches self.P_spec  (non-slack buses, same order)
        - Q_calc matches self.Q_spec  (PQ buses, same order)

        Requires:
            self.non_slack_buses
            self.pq_buses

        If those are not set yet, this will call build_spec_arrays(buses).
        """
        # make sure bus lists exist
        if self.non_slack_buses is None or self.pq_buses is None:
            self.build_spec_arrays(buses)

        n = len(buses)

        # full complex voltage vector V (all buses)
        V = np.zeros(n, dtype=complex)
        
        for b, data in buses.items():
            V[b - 1] = data["V"] * np.exp(1j * data["δ"])

        G = Ybus.real
        B = Ybus.imag

        Vm = np.abs(V)
        Va = np.angle(V)

        P_full = np.zeros(n)
        Q_full = np.zeros(n)

        # standard power injection formulas
        for i in range(n):
            for k in range(n):
                theta = Va[i] - Va[k]
                VkVi  = Vm[i] * Vm[k]
                P_full[i] += VkVi * (G[i, k] * np.cos(theta) + B[i, k] * np.sin(theta))
                Q_full[i] += VkVi * (G[i, k] * np.sin(theta) - B[i, k] * np.cos(theta))

        # reduce to match spec arrays
        self.P_calc = np.array([P_full[b - 1] for b in self.non_slack_buses], dtype=float)
        self.Q_calc = np.array([Q_full[b - 1] for b in self.pq_buses],       dtype=float)

    # Build mismatch matrix ------------------------------------------------------------------------
    def build_mismatch_vector(self):
        """
        Build the NR mismatch vector Δ = [ΔP; ΔQ] using the
        reduced arrays stored in this object.

        Requires:
            self.P_spec, self.Q_spec
            self.P_calc, self.Q_calc

        Returns:
            mismatch : column vector (ΔP followed by ΔQ)
        """
        if self.P_spec is None or self.P_calc is None:
            raise ValueError("P_spec or P_calc not set. Call build_spec_arrays and build_calc_arrays first.")
        if self.Q_spec is None or self.Q_calc is None:
            raise ValueError("Q_spec or Q_calc not set. Call build_spec_arrays and build_calc_arrays first.")

        # ΔP for non-slack buses
        deltaP = self.P_spec - self.P_calc

        # ΔQ for PQ buses
        deltaQ = self.Q_spec - self.Q_calc

        # stack into a column vector [ΔP; ΔQ]
        mismatch = np.concatenate([deltaP, deltaQ]).reshape(-1, 1)
        return mismatch

    # Build unknown matrix -----------------------------------------------------------------------------------
    def build_unknown_state_vector(self, buses):
        """
        Build the NR state (unknown) vector x = [δ_non_slack; V_PQ].

        Uses:
            self.non_slack_buses
            self.pq_buses

        Reads δ and V from the buses dict.

        Returns:
            x : 1D numpy array (δ for non-slack, then V for PQ buses)
        """
        # ensure bus lists exist
        if self.non_slack_buses is None or self.pq_buses is None:
            self.build_spec_arrays(buses)

        delta_list = []
        for b in self.non_slack_buses:
            delta_list.append(buses[b]["δ"])

        V_list = []
        for b in self.pq_buses:
            V_list.append(buses[b]["V"])

        x = np.concatenate([np.array(delta_list, dtype=float),
                            np.array(V_list,    dtype=float)])

        # optional: store it on the object
        self.state_vector = x
        return x
    
    # Build J1 for the jacobian matrix ---------------------------------------------------------
    def build_J1(self, buses, Ybus):
        """
        Build J1 = dP/d(delta) for non-slack buses.

        Rows: non-slack buses
        Cols: non-slack buses

        Returns:
            J1 : numpy array (ns × ns)
        """
        ns = len(self.non_slack_buses)
        J1 = np.zeros((ns, ns), dtype=float)

        # Access Ybus terms
        G = Ybus.real
        B = Ybus.imag

        # Build complex voltage vector first
        # (needed for angles and |V|)
        n_buses = len(buses)
        V_complex = np.zeros(n_buses, dtype=complex)
        for b, data in buses.items():
            V_complex[b - 1] = data["V"] * np.exp(1j * data["δ"])

        Vm = np.abs(V_complex)
        Va = np.angle(V_complex)

        # Also need Q_calc values indexed by bus number
        # P_calc/Q_calc in your object are reduced vectors,
        # so map them back to full-bus numbering:
        Q_full = np.zeros(n_buses)
        for idx, b in enumerate(self.pq_buses):
            Q_full[b - 1] = self.Q_calc[idx]   # only PQ buses have Q_calc

        for idx_i, bus_i in enumerate(self.non_slack_buses):
            i = bus_i - 1
            Vi = Vm[i]
            Qi = Q_full[i]

            for idx_k, bus_k in enumerate(self.non_slack_buses):
                k = bus_k - 1
                Vk = Vm[k]

                if i == k:
                    # Diagonal
                    J1[idx_i, idx_k] = -Qi - B[i, i] * Vi * Vi
                else:
                    # Off-diagonal
                    theta = Va[i] - Va[k]
                    J1[idx_i, idx_k] = (
                        Vi * Vk * (G[i, k] * np.sin(theta) - B[i, k] * np.cos(theta))
                    )

        return J1
    
    # J2 of the Jacobian Matrix -------------------------------------------------------------------------
    def build_J2(self, buses, Ybus):
        """
        Build J2 = dP/dV for NR power flow.

        Rows  : non-slack buses
        Cols  : PQ buses

        Returns:
            J2 : numpy array (ns × npq)
        """
        ns  = len(self.non_slack_buses)
        npq = len(self.pq_buses)
        J2 = np.zeros((ns, npq), dtype=float)

        # Real/imag parts of Ybus
        G = Ybus.real
        B = Ybus.imag

        # Build complex voltages
        n_buses = len(buses)
        V_complex = np.zeros(n_buses, dtype=complex)
        for b, data in buses.items():
            V_complex[b - 1] = data["V"] * np.exp(1j * data["δ"])

        Vm = np.abs(V_complex)
        Va = np.angle(V_complex)

        # Need P_calc mapped into full bus indexing
        P_full = np.zeros(n_buses)
        for idx, b in enumerate(self.non_slack_buses):
            # P_calc is stored in same order as non-slack buses
            P_full[b - 1] = self.P_calc[idx]

        # Build J2
        for row_idx, bus_i in enumerate(self.non_slack_buses):
            i = bus_i - 1
            Vi = Vm[i]

            for col_idx, bus_k in enumerate(self.pq_buses):
                k = bus_k - 1
                Vk = Vm[k]

                theta = Va[i] - Va[k]

                if i == k:
                    # Diagonal only if that bus is PQ
                    P_i = P_full[i]
                    J2[row_idx, col_idx] = (P_i / Vi) + G[i, i] * Vi
                else:
                    # Off-diagonal
                    J2[row_idx, col_idx] = Vi * (
                        G[i, k] * np.cos(theta) + B[i, k] * np.sin(theta)
                    )

        return J2
    
    # J3 for the Jacobian Matrix ---------------------------------------------------------------
    def build_J3(self, buses, Ybus):
        """
        Build J3 = dQ/d(delta)

        Rows: PQ buses
        Cols: non-slack buses

        Returns:
            J3 : numpy array (npq × ns)
        """
        npq = len(self.pq_buses)
        ns  = len(self.non_slack_buses)
        J3 = np.zeros((npq, ns), dtype=float)

        G = Ybus.real
        B = Ybus.imag

        # Build complex voltage vector
        n_buses = len(buses)
        V_complex = np.zeros(n_buses, dtype=complex)
        for b, data in buses.items():
            V_complex[b - 1] = data["V"] * np.exp(1j * data["δ"])

        Vm = np.abs(V_complex)
        Va = np.angle(V_complex)

        # Need P_calc mapped to full bus indexing
        P_full = np.zeros(n_buses)
        for idx, b in enumerate(self.non_slack_buses):
            P_full[b - 1] = self.P_calc[idx]  # only non-slack buses have P_calc values

        # Build J3
        for row_idx, bus_i in enumerate(self.pq_buses):
            i = bus_i - 1
            Vi = Vm[i]

            for col_idx, bus_k in enumerate(self.non_slack_buses):
                k = bus_k - 1
                Vk = Vm[k]

                theta = Va[i] - Va[k]

                if i == k:
                    # Diagonal entry (PQ bus only)
                    P_i = P_full[i]
                    J3[row_idx, col_idx] = P_i - G[i, i] * Vi * Vi
                else:
                    # Off-diagonal
                    J3[row_idx, col_idx] = -Vi * Vk * (
                        G[i, k] * np.cos(theta) + B[i, k] * np.sin(theta)
                    )

        return J3
    
    # J4 for the Jacobian Matrix ---------------------------------------------------------------
    def build_J4(self, buses, Ybus):
        """
        Build J4 = dQ/dV for PQ buses.

        Rows: PQ buses
        Cols: PQ buses

        Returns:
            J4 : numpy array (npq × npq)
        """
        npq = len(self.pq_buses)
        J4 = np.zeros((npq, npq), dtype=float)

        G = Ybus.real
        B = Ybus.imag

        # Build complex voltage vector
        n_buses = len(buses)
        V_complex = np.zeros(n_buses, dtype=complex)
        for b, data in buses.items():
            V_complex[b - 1] = data["V"] * np.exp(1j * data["δ"])

        Vm = np.abs(V_complex)
        Va = np.angle(V_complex)

        # Need Q_calc in full bus indexing
        Q_full = np.zeros(n_buses)
        for idx, b in enumerate(self.pq_buses):
            Q_full[b - 1] = self.Q_calc[idx]

        for row_idx, bus_i in enumerate(self.pq_buses):
            i = bus_i - 1
            Vi = Vm[i]
            Qi = Q_full[i]

            for col_idx, bus_k in enumerate(self.pq_buses):
                k = bus_k - 1
                Vk = Vm[k]

                theta = Va[i] - Va[k]

                if i == k:
                    # Diagonal PQ bus
                    J4[row_idx, col_idx] = (Qi / Vi) - B[i, i] * Vi
                else:
                    # Off-diagonal
                    J4[row_idx, col_idx] = Vi * (
                        G[i, k] * np.sin(theta) - B[i, k] * np.cos(theta)
                    )

        return J4

    # Full Jacobian Matrix build ---------------------------------------------------------------
    def build_jacobian(self, buses, Ybus):
        """
        Build the full NR Jacobian:
        
            J = [ J1   J2
                  J3   J4 ]

        Returns:
            J : full Jacobian matrix (numpy array)
        """
        # Build the four blocks
        J1 = self.build_J1(buses, Ybus)
        J2 = self.build_J2(buses, Ybus)
        J3 = self.build_J3(buses, Ybus)
        J4 = self.build_J4(buses, Ybus)

        # Stack horizontally (J1 with J2, J3 with J4)
        top = np.hstack((J1, J2))
        bottom = np.hstack((J3, J4))

        # Stack vertically into the full Jacobian
        J = np.vstack((top, bottom))
        return J

    # Solve for unknown state matrix ---------------------------------------------------------------
    def solve_for_state_update(self, buses, Ybus):
        """
        Build mismatch and Jacobian, then solve for Δx.

        Δx = [Δδ_non_slack; ΔV_PQ]

        Returns:
            delta_x : 1D numpy array (same length as state vector)
        """
        # Make sure spec/calc are up to date
        self.build_spec_arrays(buses)
        self.build_calc_arrays(buses, Ybus)

        # Build mismatch vector (column)
        mismatch = self.build_mismatch_vector()   # shape (ns+npq, 1)

        # Build full Jacobian
        J = self.build_jacobian(buses, Ybus)

        # Solve J * Δx = mismatch  (flatten mismatch to 1D)
        delta_x = np.linalg.solve(J, mismatch.flatten())

        # Optional: store it
        self.delta_x = delta_x
        return delta_x
    
    # Apply the solved unknown state to new state matrix ---------------------------------------------------------------
    def apply_state_update(self, buses, delta_x):
        """
        Apply Δx to update bus angles and voltages.

        delta_x is ordered as:
            [Δδ for non-slack buses,
             ΔV for PQ buses]
        """
        ns = len(self.non_slack_buses)
        npq = len(self.pq_buses)

        d_delta = delta_x[:ns]
        d_V     = delta_x[ns:ns+npq]

        # Update angles for non-slack buses
        for idx, b in enumerate(self.non_slack_buses):
            buses[b]["δ"] += d_delta[idx]

        # Update voltages for PQ buses
        for idx, b in enumerate(self.pq_buses):
            buses[b]["V"] += d_V[idx]

#print(dir(PowerVariables))


pv = PowerVariables()      # create the object
"""pv.build_spec_arrays(buses)  # call the method
pv.build_calc_arrays(buses, Ybus)
mismatch = pv.build_mismatch_vector()
#pv.build_mismatch_vector()

print("\nP_spec =", pv.P_spec)
print("\nQ_spec =", pv.Q_spec)
print("\nnon-slack =", pv.non_slack_buses)
print("\npq =", pv.pq_buses)
print("\nP_calc =", pv.P_calc)
print("\nQ_calc =", pv.Q_calc)
print("\nMismatch Vectors", mismatch)
print("Initial max mismatch:", np.max(np.abs(mismatch)))
print("\nUnknown state", pv.build_unknown_state_vector(buses))
#print("\nFull Jacobian matrix", pv.build_jacobian(buses, Ybus))
display_Ybus(pv.build_jacobian(buses, Ybus))
"""

delta_x = pv.solve_for_state_update(buses, Ybus)
pv.apply_state_update(buses, delta_x)
print("\nUnknown state", pv.build_unknown_state_vector(buses))
print("\nΔx:", delta_x)

for tol in EPS:
    num = -1
    for k in range(MAX_ITERS):
        num+=1
        # 1–3: build spec, calc, mismatch
        pv.build_spec_arrays(buses)
        pv.build_calc_arrays(buses, Ybus)
        mismatch = pv.build_mismatch_vector()   # column vector

        # 4: convergence check
        max_mis = np.max(np.abs(mismatch))
        print(f"Iter {k}: max mismatch = {max_mis:.6e}")
        if max_mis < tol:
            print("Converged!", num, tol, max_mis)
            break

        # 5: build Jacobian
        J = pv.build_jacobian(buses, Ybus)

        # 6: solve for Δx
        delta_x = np.linalg.solve(J, mismatch.flatten())

        # 7: apply update to buses
        pv.apply_state_update(buses, delta_x)
        
    else:
        print("Did not converge within MAX_ITERS")

#print("\nJ1 Jacobian matrix", pv.build_J1(buses, Ybus))
#print("\nJ2 Jacobian matrix", pv.build_J2(buses, Ybus))
#print("\nJ3 Jacobian matrix", pv.build_J3(buses, Ybus))
#print("\nJ4 Jacobian matrix", pv.build_J4(buses, Ybus))

#display_Ybus(Ybus)
#build_mismatch_matrix(Ybus)
#calc_power_injections(Ybus, V)

#build_mismatch_vector(Ybus, V, P_spec, Q_spec, slack_bus, pv_buses, pq_buses)

#tabulate_results(buses)
#print(Ybus)
#print("Off diag", Y_off)
#print("Y polar", np.round(np.cos(np.pi/2),8))
#print("radians", degreeToRadians(90))
# optional: if you want to view angles in degrees
# Y_polar_deg = np.array([(abs(y), np.degrees(np.angle(y))) for y in Y_diag])