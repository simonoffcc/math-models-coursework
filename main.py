import numpy as np
from math import exp

p1 = 0.5
# p3 = 20
p4 = 14  # 8, 10, 12, 14
p5 = 0.8
p6 = 0
x2_start = -1.9
x2_end = 4.5
step = 0.1
signs = 3


# Собственные значения для проверки: -0.846 и -0.365

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

# Open the file in write mode
with open(f'B) p4={p4}.txt', 'w') as f:
    # Write the header
    f.write("x2|x1|p2|eigv1|eigv2\n")

    for x2 in np.arange(x2_start, x2_end + step, step):
        x2_values.append(x2)

        x1 = calculate_x1(x2, p1, p4, p5, p6)
        x1_values.append(x1)

        # p2 = calculate_p2(x2, x1, p1, p3)
        p2 = calculate_p2_var2(x2, x1, p1)
        p2_values.append(p2)

        # left_top = -p1 - p2 * e ** (x2 / (1 + x2 / p3))
        # left_btm = -p2 * p4 * e ** (x2 / (1 + x2 / p3))
        # right_top = p2 * (1 - x1) * (e ** ((p3 * x2) / (p3 + x2)) * p3 ** 2) / (p3 + x2) ** 2
        # right_btm = -p1 + p2 * p4 * (1 - x1) * (e ** ((p3 * x2) / (p3 + x2)) * p3 ** 2) / (p3 + x2) ** 2 - p5

        left_top = -p1 - p2 * exp(x2)
        left_btm = -p2 * p4 * exp(x2)
        right_top = p2 * (1 - x1) * exp(x2)
        right_btm = -p1 + p2 * p4 * (1 - x1) * exp(x2) - p5
        A = np.array([[left_top, right_top],
                      [left_btm, right_btm]])
        eigenvalues = np.linalg.eigvals(A)

        # Format the output string
        output_string = (f"{x2:.{signs}f}|{x1:.{signs}f}|{p2:.{signs}f}"
                         f"|{eigenvalues[0]:.{signs}f}|{eigenvalues[1]:.{signs}f}\n")

        # Write the formatted string to the file
        f.write(output_string)
