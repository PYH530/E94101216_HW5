import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 定義微分方程與解析解
def f1(t, y):
    return 1 + (y/t) + (y/t)**2

def exact_solution1(t):
    return t * np.tan(np.log(t))

def df1_dt(t, y):
    return (-y/t**2) + 2 * y * (1 + (y/t)) / t**2

# Euler 方法
def euler_method(f, t0, y0, h, t_end):
    t_values = [t0]
    y_values = [y0]
    t, y = t0, y0
    while t < t_end:
        y += h * f(t, y)
        t = round(t + h, 10)
        t_values.append(t)
        y_values.append(y)
    return np.array(t_values), np.array(y_values)

# Taylor 二階方法
def taylor2_method(f, df_dt, t0, y0, h, t_end):
    t_values = [t0]
    y_values = [y0]
    t, y = t0, y0
    while t < t_end:
        f_val = f(t, y)
        df_val = df_dt(t, y)
        y += h * f_val + (h**2 / 2) * df_val
        t = round(t + h, 10)
        t_values.append(t)
        y_values.append(y)
    return np.array(t_values), np.array(y_values)

# 初始化與計算
t0, y0, t_end, h = 1.0, 0.0, 2.0, 0.1
t_euler, y_euler = euler_method(f1, t0, y0, h, t_end)
t_taylor, y_taylor = taylor2_method(f1, df1_dt, t0, y0, h, t_end)
y_exact = exact_solution1(t_euler)

# 畫圖
plt.figure(figsize=(10, 6))
plt.plot(t_euler, y_euler, 'bo-', label="Euler's Method")
plt.plot(t_taylor, y_taylor, 'go-', label="Taylor's Method Order 2")
plt.plot(t_euler, y_exact, 'r-', label="Exact Solution")
plt.title("Problem 1: y' = 1 + (y/t) + (y/t)^2")
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.savefig("problem1_plot.png")

# 儲存表格
df = pd.DataFrame({
    "t": t_euler,
    "Euler y": y_euler,
    "Taylor y": y_taylor,
    "Exact y": y_exact,
    "Euler Error": np.abs(y_euler - y_exact),
    "Taylor Error": np.abs(y_taylor - y_exact)
})
df.to_csv("problem1_table.csv", index=False)
