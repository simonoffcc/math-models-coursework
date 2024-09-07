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

def calculate_p2_var2(x2, x1, p1):
    p2 = (p1 * x1) / ((1 - x1) * exp(x2))
    return p2

x2_values = []
x1_values = []
p2_values = []
eigenvalues_list = []
bifurcation_points = []

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

    eigenvalues = np.linalg.eigvals(A)
    eigenvalues_list.append(eigenvalues)

    if len(eigenvalues_list) > 1:
        previous_eigenvalues = eigenvalues_list[-2]
        if any((e.real < 0) != (pe.real < 0) for e, pe in zip(eigenvalues, previous_eigenvalues)):
            bifurcation_points.append((p2, x2, x1))

stable_points = []
unstable_points = []

for x2, x1, p2, eigenvalues in zip(x2_values, x1_values, p2_values, eigenvalues_list):
    if all(eig_real < 0 for eig_real in eigenvalues.real):
        stable_points.append((p2, x2, x1))
    else:
        unstable_points.append((p2, x2, x1))

if stable_points:
    stable_p2, stable_x2, stable_x1 = zip(*stable_points)
else:
    stable_p2, stable_x2, stable_x1 = [], [], []

if unstable_points:
    unstable_p2, unstable_x2, unstable_x1 = zip(*unstable_points)
else:
    unstable_p2, unstable_x2, unstable_x1 = [], [], []

# if bifurcation_points:
#     bifurcation_p2, bifurcation_x2, bifurcation_x1 = zip(*bifurcation_points)
# else:
#     bifurcation_p2, bifurcation_x2, bifurcation_x1 = [], [], []

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot x2 vs p2
ax1.plot(p2_values, x2_values, label='x2(p2)', color='blue')
ax1.scatter(stable_p2, stable_x2, color='green', label='Устойчивые точки', s=50)
ax1.scatter(unstable_p2, unstable_x2, color='red', label='Неустойчивые точки', s=50)
# ax1.scatter(bifurcation_p2, bifurcation_x2, color='orange', label='Точки бифуркации', s=100, marker='x')
ax1.set_xlabel('p2', fontsize=14)
ax1.set_ylabel('x2', fontsize=14)
ax1.set_title('x2(p2)', fontsize=16)
ax1.legend(fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot x1 vs p2
ax2.plot(p2_values, x1_values, label='x1(p2)', color='purple')
ax2.scatter(stable_p2, stable_x1, color='green', label='Устойчивые точки', s=50)
ax2.scatter(unstable_p2, unstable_x1, color='red', label='Неустойчивые точки', s=50)
# ax2.scatter(bifurcation_p2, bifurcation_x1, color='orange', label='Точки бифуркации', s=100, marker='x')
ax2.set_xlabel('p2', fontsize=14)
ax2.set_ylabel('x1', fontsize=14)
ax2.set_title('x1(p2)', fontsize=16)
ax2.legend(fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
