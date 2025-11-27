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
EPS = 0.01  # mismatch tolerance (pu)
#k = 0       # NR iteration counter

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
print(f"buses-----------------\n{buses}\n")


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

TL = 3      # Number of Transmission lines

# --- bus type enum ---
SLACK, PQ, PV = 0, 1, 2


n = len(buses)       # Number of busses

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


def tabulate_results(data):
    print(tabulate(data, headers='keys', tablefmt='grid'))

# build Y-bus -----------------------------------------------
Ybus = build_Ybus(Z)
print(f"[INFO] Your Ybus matrix is: \n{Ybus}\n")

# extract diagonal elements ---------------------------------
Y_diag = np.diag(Ybus)
print(f"[INFO] Your Y-diagonals are: {Y_diag}\n")

# Extract off-diagonal Y values (i ≠ j) ----------------------
off_diagonals = []
for i in range(n):
    for j in range(n):
        if i != j:
            y = Ybus[i, j]
            mag = abs(y)
            ang = np.angle(y)  # radians
            off_diagonals.append(((i+1, j+1), mag, ang))

# Convert to array for clarity (bus pair, magnitude, angle)
Y_off = np.array(off_diagonals, dtype=object)
print(f"[INFO] Your Ybus off diagonal elements are: \n{Y_off}\n")

# convert to polar form (magnitude, angle in radians)
Y_polar_diag = np.array([(abs(y), np.angle(y)) for y in Y_diag])
print(f"[INFO] Your Ybus polar diagonals are: \n{Y_polar_diag}\n")

Y_polar_off = np.array([(abs(y), np.angle(y)) for y in Y_diag])
print(f"[INFO] Your Ybus polar off diagonals are: \n{Y_polar_off}\n")



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


#print(dir(PowerVariables))

# Setting up calculations for mismatch matrix
def build_calc_arrays(Ybus):
    """
    Compute reduced P_calc and Q_calc from Ybus and current bus voltages.

    Uses:
      - buses (global dict)
      - build_spec_arrays() to get non-slack and PQ bus lists

    Returns:
      P_calc : array for non-slack buses (same order as P_spec)
      Q_calc : array for PQ buses       (same order as Q_spec)
    """
    # get bus sets and ensure P_spec/Q_spec are built (we reuse the mapping)
    non_slack_bus_nums, pq_bus_nums = build_spec_arrays()

    # full complex voltage vector
    V = np.zeros(n, dtype=complex)
    for b, data in buses.items():
        V[b - 1] = data["V"] * np.exp(1j * data["δ"])

    G = Ybus.real
    B = Ybus.imag
    Vm = np.abs(V)
    Va = np.angle(V)

    P_full = np.zeros(n)
    Q_full = np.zeros(n)

    for i in range(n):
        for k in range(n):
            theta = Va[i] - Va[k]
            VkVi  = Vm[i] * Vm[k]
            P_full[i] += VkVi * (G[i, k] * np.cos(theta) + B[i, k] * np.sin(theta))
            Q_full[i] += VkVi * (G[i, k] * np.sin(theta) - B[i, k] * np.cos(theta))

    # reduce to match spec arrays
    P_calc = np.array([P_full[b - 1] for b in non_slack_bus_nums], dtype=float)
    Q_calc = np.array([Q_full[b - 1] for b in pq_bus_nums],       dtype=float)
    print("P_calc (reduced) =", P_calc)
    print("Q_calc (reduced) =", Q_calc)
    return P_calc, Q_calc, non_slack_bus_nums, pq_bus_nums

#
#print("P_calc =", P_calc)
#print("Q_calc =", Q_calc)

# Build Mismatch matrix
def build_mismatch_matrix(P_spec, Q_spec, P_calc, Q_calc):
    """
    Build the mismatch matrix Δ = [ΔP; ΔQ].

    Inputs:
      P_spec : reduced P_spec array (non-slack buses)
      Q_spec : reduced Q_spec array (PQ buses)
      P_calc : reduced P_calc array (non-slack buses)
      Q_calc : reduced Q_calc array (PQ buses)

    Returns:
      mismatch : column vector [ΔP; ΔQ]
    """

    # ΔP for non-slack buses
    deltaP = P_spec - P_calc

    # ΔQ for PQ buses
    deltaQ = Q_spec - Q_calc

    # stack into a column vector
    mismatch = np.concatenate([deltaP, deltaQ]).reshape(-1, 1)
    return mismatch

pv = PowerVariables()      # create the object
pv.build_spec_arrays(buses)  # call the method
pv.build_calc_arrays(buses, Ybus)

print("P_spec =", pv.P_spec)
print("Q_spec =", pv.Q_spec)
print("non-slack =", pv.non_slack_buses)
print("pq =", pv.pq_buses)
print("P_calc =", pv.P_calc)
print("Q_calc =", pv.Q_calc)

#build_mismatch_matrix(Ybus)
#calc_power_injections(Ybus, V)

#build_mismatch_vector(Ybus, V, P_spec, Q_spec, slack_bus, pv_buses, pq_buses)

#tabulate_results(buses)
#print(Ybus)
#print("Off diag", Y_off)
#print("Y polar", np.round(np.cos(np.pi/2),8))
print("radians", degreeToRadians(90))
# optional: if you want to view angles in degrees
# Y_polar_deg = np.array([(abs(y), np.degrees(np.angle(y))) for y in Y_diag])