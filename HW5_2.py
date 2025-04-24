import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 精確解
def u1_exact(t):
    return 2 * np.exp(-3*t) - np.exp(-39*t) + (1/3)*np.cos(t)

def u2_exact(t):
    return -np.exp(-3*t) + 2 * np.exp(-39*t) - (1/3)*np.cos(t)

# 微分方程系統
def f_system(t, u):
    u1, u2 = u
    du1 = 9*u1 + 24*u2 + 5*np.cos(t) - (1/3)*np.sin(t)
    du2 = -24*u1 - 52*u2 - 9*np.cos(t) + (1/3)*np.sin(t)
    return np.array([du1, du2])

# Runge-Kutta 4階法
def rk4_system(f, u0, t0, t_end, h):
    t_values = [t0]
    u_values = [u0]
    t, u = t0, u0
    while t < t_end:
        k1 = h * f(t, u)
        k2 = h * f(t + h/2, u + k1/2)
        k3 = h * f(t + h/2, u + k2/2)
        k4 = h * f(t + h, u + k3)
        u = u + (k1 + 2*k2 + 2*k3 + k4) / 6
        t = round(t + h, 10)
        t_values.append(t)
        u_values.append(u)
    return np.array(t_values), np.array(u_values)

# 初始條件與步長
t0, t_end = 0.0, 1.0
u0 = np.array([4/3, 2/3])

# 執行 RK4，兩種步長
t_rk1, u_rk1 = rk4_system(f_system, u0, t0, t_end, 0.05)
t_rk2, u_rk2 = rk4_system(f_system, u0, t0, t_end, 0.1)

# 精確解
u1_true = u1_exact(t_rk1)
u2_true = u2_exact(t_rk1)

# 畫圖
plt.figure(figsize=(12, 6))
plt.plot(t_rk1, u_rk1[:, 0], 'b.-', label="u1 RK4 h=0.05")
plt.plot(t_rk1, u_rk1[:, 1], 'g.-', label="u2 RK4 h=0.05")
plt.plot(t_rk2, u_rk2[:, 0], 'bo--', label="u1 RK4 h=0.1")
plt.plot(t_rk2, u_rk2[:, 1], 'go--', label="u2 RK4 h=0.1")
plt.plot(t_rk1, u1_true, 'r-', label="u1 Exact")
plt.plot(t_rk1, u2_true, 'm-', label="u2 Exact")
plt.title("Problem 2: Correct RK4 for System of ODEs")
plt.xlabel("t")
plt.ylabel("u1, u2")
plt.grid(True)
plt.legend()
plt.savefig("problem2_correct_plot.png")

# 儲存表格 CSV
df = pd.DataFrame({
    "t": t_rk1,
    "u1 RK4 (h=0.05)": u_rk1[:, 0],
    "u2 RK4 (h=0.05)": u_rk1[:, 1],
    "u1 Exact": u1_true,
    "u2 Exact": u2_true,
    "u1 Error": np.abs(u_rk1[:, 0] - u1_true),
    "u2 Error": np.abs(u_rk1[:, 1] - u2_true)
})
df.to_csv("problem2_correct_table.csv", index=False)
