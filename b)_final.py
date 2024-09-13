import numpy as np
from math import exp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Входные данные
p1 = 0.5
p4 = 8  # 8, 10, 12, 14
p5 = 0.8
p6 = 0
x2_start = -1.9
x2_end = 4.5
step = 0.1
signs = 3

# Выраженные функции
def calculate_x1(x2, p1, p4, p5, p6):
    x1 = (p1 * x2 + p5 * (x2 - p6)) / (p1 * p4)
    return x1

def calculate_p2(x2, x1, p1):
    p2 = (p1 * x1) / ((1 - x1) * exp(x2))
    return p2

# Проверка равновесия
def check_equilibrium(x2, x1, p2, p1, p4, p5):
    dx1_dt = p1 * x2 + p5 * (x2 - p6) - p1 * p4 * x1
    dp2_dt = (p1 * x1) - ((1 - x1) * p2 * exp(x2))
    return np.isclose(dx1_dt, 0, atol=1e-5) and np.isclose(dp2_dt, 0, atol=1e-5)

# Массивы для записи точек
x2_values = []
x1_values = []
p2_values = []
eigenvalues_list = []
bifurcation_points = []

# Общий цикл
for x2 in np.arange(x2_start, x2_end + step, step):
    x2_values.append(x2)

    x1 = calculate_x1(x2, p1, p4, p5, p6)
    x1_values.append(x1)

    p2 = calculate_p2(x2, x1, p1)
    p2_values.append(p2)

    left_top = -p1 - p2 * exp(x2)
    left_btm = -p2 * p4 * exp(x2)
    right_top = p2 * (1 - x1) * exp(x2)
    right_btm = -p1 + p2 * p4 * (1 - x1) * exp(x2) - p5
    A = np.array([[left_top, right_top],
                  [left_btm, right_btm]])
    eigenvalues = np.linalg.eigvals(A)
    eigenvalues_list.append(eigenvalues)

    if len(eigenvalues_list) > 1:
        previous_eigenvalues = eigenvalues_list[-2]
        if any((e.real < 0) != (pe.real < 0) for e, pe in zip(eigenvalues, previous_eigenvalues)):
            bifurcation_points.append((p2, x2, x1))

# Массивы для записи устойчивых и неустойчивых точек
stable_points = []
unstable_points = []

for x2, x1, p2, eigenvalues in zip(x2_values, x1_values, p2_values, eigenvalues_list):
    if all(eig_real < 0 for eig_real in eigenvalues.real):
        stable_points.append((p2, x2, x1))
    else:
        unstable_points.append((p2, x2, x1))

# Запись бифуркаций
bifurcation_indices = []
for i in range(1, len(eigenvalues_list)):
    prev_stable = all(eig.real < 0 for eig in eigenvalues_list[i-1])
    curr_stable = all(eig.real < 0 for eig in eigenvalues_list[i])
    if prev_stable != curr_stable:
        bifurcation_indices.append(i)

# Вывод точек бифуркаций для каждого графика в консоль
print("Бифуркация для x2(p2)")
for idx in bifurcation_indices:
    if idx < len(eigenvalues_list):
        print(f"Переход в точках номер {idx} - {idx+1}:")
        print(f"p2 = {p2_values[idx-1]:.{signs}f}, x2 = {x2_values[idx-1]:.{signs}f}, собств. знач. = {eigenvalues_list[idx-1]}")
        print(f" p2 = {p2_values[idx]:.{signs}f}, x2 = {x2_values[idx]:.{signs}f}, собств. знач. = {eigenvalues_list[idx]}")
        print("\n")

print("Бифуркация для x1(p2):")
for idx in bifurcation_indices:
    if idx < len(eigenvalues_list):
        print(f"Переход в точках номер {idx} - {idx+1}:")
        print(f"p2 = {p2_values[idx-1]:.{signs}f}, x1 = {x1_values[idx-1]:.{signs}f}, собств. знач. = {eigenvalues_list[idx-1]}")
        print(f"p2 = {p2_values[idx]:.{signs}f}, x1 = {x1_values[idx]:.{signs}f}, собств. знач. = {eigenvalues_list[idx]}")
        print("\n")

# Подготовка точек для отрисовки на графике
if stable_points:
    stable_p2, stable_x2, stable_x1 = zip(*stable_points)
else:
    stable_p2, stable_x2, stable_x1 = [], [], []

if unstable_points:
    unstable_p2, unstable_x2, unstable_x1 = zip(*unstable_points)
else:
    unstable_p2, unstable_x2, unstable_x1 = [], [], []

# Проверка равновесия для устойчивых точек
print("Проверка равновесия для устойчивых точек:")
for p2, x2, x1 in stable_points:
    is_equilibrium = check_equilibrium(x2, x1, p2, p1, p4, p5)
    print(f"p2 = {p2:.{signs}f}, x2 = {x2:.{signs}f}, x1 = {x1:.{signs}f} -> Равновесие: {is_equilibrium}")

# Проверка равновесия для неустойчивых точек
print("Проверка равновесия для неустойчивых точек:")
for p2, x2, x1 in unstable_points:
    is_equilibrium = check_equilibrium(x2, x1, p2, p1, p4, p5)
    print(f"p2 = {p2:.{signs}f}, x2 = {x2:.{signs}f}, x1 = {x1:.{signs}f} -> Равновесие: {is_equilibrium}")

# Рисовка графиков
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(p2_values, x2_values, label='x2(p2)', color='blue')
ax1.scatter(stable_p2, stable_x2, color='green', label='Устойчивые точки', s=50)
ax1.scatter(unstable_p2, unstable_x2, color='red', label='Неустойчивые точки', s=50)
ax1.set_xlabel('p2', fontsize=14)
ax1.set_ylabel('x2', fontsize=14)
ax1.set_title('x2(p2)', fontsize=16)
ax1.legend(fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)

ax2.plot(p2_values, x1_values, label='x1(p2)', color='purple')
ax2.scatter(stable_p2, stable_x1, color='green', label='Устойчивые точки', s=50)
ax2.scatter(unstable_p2, unstable_x1, color='red', label='Неустойчивые точки', s=50)
ax2.set_xlabel('p2', fontsize=14)
ax2.set_ylabel('x1', fontsize=14)
ax2.set_title('x1(p2)', fontsize=16)
ax2.legend(fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
