import numpy as np
import pandas as pd

# 定義微分方程
def f1(t, u1, u2):
    return 9*u1 + 24*u2 + 5*np.cos(t) - (1/3)*np.sin(t)

def f2(t, u1, u2):
    return -24*u1 - 52*u2 - 9*np.cos(t) + (1/3)*np.sin(t)

# 精確解
def exact_u1(t):
    return 2*np.exp(-3*t) - np.exp(-39*t) + (1/3)*np.cos(t)

def exact_u2(t):
    return -np.exp(-3*t) + 2*np.exp(-39*t) - (1/3)*np.cos(t)

# Runge-Kutta 4方法
def runge_kutta(h, t_end):
    n_steps = int(t_end / h)  # 保證結束在t=1
    t_values = [0]
    u1_values = [4/3]
    u2_values = [2/3]

    t = 0
    u1 = 4/3
    u2 = 2/3

    for _ in range(n_steps):
        k1_1 = f1(t, u1, u2)
        k1_2 = f2(t, u1, u2)

        k2_1 = f1(t + h/2, u1 + h/2*k1_1, u2 + h/2*k1_2)
        k2_2 = f2(t + h/2, u1 + h/2*k1_1, u2 + h/2*k1_2)

        k3_1 = f1(t + h/2, u1 + h/2*k2_1, u2 + h/2*k2_2)
        k3_2 = f2(t + h/2, u1 + h/2*k2_1, u2 + h/2*k2_2)

        k4_1 = f1(t + h, u1 + h*k3_1, u2 + h*k3_2)
        k4_2 = f2(t + h, u1 + h*k3_1, u2 + h*k3_2)

        u1 += h/6 * (k1_1 + 2*k2_1 + 2*k3_1 + k4_1)
        u2 += h/6 * (k1_2 + 2*k2_2 + 2*k3_2 + k4_2)

        t += h
        t_values.append(t)
        u1_values.append(u1)
        u2_values.append(u2)

    return np.array(t_values), np.array(u1_values), np.array(u2_values)

# 跑 h=0.05 和 h=0.1
h_list = [0.05, 0.1]
all_data = []

for h in h_list:
    t, u1_num, u2_num = runge_kutta(h, 1.0)
    u1_exact_vals = exact_u1(t)
    u2_exact_vals = exact_u2(t)
    u1_error = np.abs(u1_num - u1_exact_vals)
    u2_error = np.abs(u2_num - u2_exact_vals)

    df_temp = pd.DataFrame({
        'h': h,
        't': t,
        'u1_numeric': u1_num,
        'u1_exact': u1_exact_vals,
        'u1_error': u1_error,
        'u2_numeric': u2_num,
        'u2_exact': u2_exact_vals,
        'u2_error': u2_error
    })
    all_data.append(df_temp)

final_df = pd.concat(all_data, ignore_index=True)

# 輸出
final_df.to_csv('problem2_table.csv', index=False)
print('CSV檔案產生完成！')
