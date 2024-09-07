import numpy as np
from math import e, exp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


p1 = 1
p3 = 20
p4 = 10
p5 = 0.6
p6 = -5
x2_start = -1.9
x2_end = 4.5
step = 0.1
signs = 3

def calculate_x1(x2, p1, p4, p5, p6):
    x1 = (p1 * x2 + p5 * (x2 - p6)) / (p1 * p4)
    return x1

def calculate_p2(x2, x1, p1, p3):
    p2 = (p1 * x1) / ((1 - x1) * exp(x2 / (1 + x2 / p3)))
    return p2

x2_values = []
x1_values = []
p2_values = []
eigenvalues_list = []

for x2 in np.arange(x2_start, x2_end + step, step):
    x2_values.append(x2)

    x1 = calculate_x1(x2, p1, p4, p5, p6)
    x1_values.append(x1)

    p2 = calculate_p2(x2, x1, p1, p3)
    p2_values.append(p2)

    left_top = -p1 - p2 * e ** (x2 / (1 + x2 / p3))
    left_btm = -p2 * p4 * e ** (x2 / (1 + x2 / p3))
    right_top = p2 * (1 - x1) * (e ** ((p3 * x2)/(p3 + x2)) * p3**2) / (p3 + x2)**2
    right_btm = -p1 + p2 * p4 * (1 - x1) * (e ** ((p3 * x2)/(p3 + x2)) * p3**2) / (p3 + x2)**2 - p5
    A = np.array([[left_top, right_top],
                  [left_btm, right_btm],
                  ])

    print(f"left_top: {left_top}, left_btm: {left_btm}, right_top: {right_top}, right_btm: {right_btm}")
    eigenvalues = np.linalg.eigvals(A)

    print(f"x2 = {x2:.{signs}f}, x1 = {x1:.{signs}f}, p2 = {p2:.{signs}f} | "
          f"Eigenvalues: [{eigenvalues[0]:.{signs}f}, {eigenvalues[1]:.{signs}f}]")
    eigenvalues = np.linalg.eigvals(A)
    eigenvalues_list.append(eigenvalues)

stable_points = []
unstable_points = []

for x2, x1, p2, eigenvalues in zip(x2_values, x1_values, p2_values, eigenvalues_list):
    if all(eig_real < 0 for eig_real in eigenvalues.real):
        stable_points.append((p2, x2))
    else:
        unstable_points.append((p2, x2))

stable_p2, stable_x2 = zip(*stable_points)
unstable_p2, unstable_x2 = zip(*unstable_points)

plt.show()
