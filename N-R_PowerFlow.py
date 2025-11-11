"""
NR Power Flow: 3-bus system
Buses: 1=Slack, 2=PQ (load), 3=PV (gen)
Given: V1=1.0∠0, PD2=0.9, QD2=0.5, PG3=1.3, V3=1.01
Lines: Z12=j0.1, Z13=j0.25, Z23=j0.2
Epsilon=0.01, k=0
Flat start: σ1=0, σ2=0, V2=1
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
k = 0       # NR iteration counter
n = 3       # Number of busses
TL = 3      # Number of Transmission lines

# --- bus type enum ---
SLACK, PQ, PV = 0, 1, 2

# --- system data (per-unit) ---
buses = {
    1: {"type": SLACK, "V": 1.0,  "delta": 0.0,    "PG": 0.0, "PD": 0.0, "QG": 0.0, "QD": 0.0},
    2: {"type": PQ,    "V": 1.0,  "delta": 0.0,    "PG": 0.0, "PD": 0.9, "QG": 0.0, "QD": 0.5},
    3: {"type": PV,    "V": 1.01, "delta": 0.0,    "PG": 1.3, "PD": 0.0, "QG": 0.0, "QD": 0.0},
}

# --- flat start values ---
sigma = np.array([0.0, 0.0])   # σ2, σ3 (bus angles in radians excluding slack)
V = np.array([1.0, 1.01])      # V2, V3 magnitudes

# --- line impedances (pu) ---
Z = {
    (1, 2): 1j * 0.1,
    (1, 3): 1j * 0.25,
    (2, 3): 1j * 0.2,
}

# store symmetrically for convenience
Z.update({(j, i): z for (i, j), z in list(Z.items())})

def degreeToRadians(degree):
    radians = np.pi * degree / 180
    return radians

# Y-Bus function
def build_Ybus(Z):
    """Return nxn Y-bus matrix for the given line impedances Z."""
    Y = np.zeros((n, n), dtype=complex)

    # mutual admittances (off-diagonal)
    Y[0, 1] = Y[1, 0] = -1 / Z[(1, 2)]
    Y[0, 2] = Y[2, 0] = -1 / Z[(1, 3)]
    Y[1, 2] = Y[2, 1] = -1 / Z[(2, 3)]

    # self admittances (diagonal)
    Y[0, 0] = 1 / Z[(1, 2)] + 1 / Z[(1, 3)]
    Y[1, 1] = 1 / Z[(1, 2)] + 1 / Z[(2, 3)]
    Y[2, 2] = 1 / Z[(1, 3)] + 1 / Z[(2, 3)]

    return Y

def tabulate_results(data):
    print(tabulate(data, headers='keys', tablefmt='grid'))

# build Y-bus
Ybus = build_Ybus(Z)

# extract diagonal elements
Y_diag = np.diag(Ybus)

# Extract off-diagonal Y values (i ≠ j)
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

# convert to polar form (magnitude, angle in radians)
Y_polar_diag = np.array([(abs(y), np.angle(y)) for y in Y_diag])
Y_polar_off = np.array([(abs(y), np.angle(y)) for y in Y_diag])

tabulate_results(buses)
print(Ybus)
print("Off diag", Y_off)
print("Y polar", np.round(np.cos(np.pi/2),8))
print("radians", degreeToRadians(90))
# optional: if you want to view angles in degrees
# Y_polar_deg = np.array([(abs(y), np.degrees(np.angle(y))) for y in Y_diag])