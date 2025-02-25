import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

G = 6.67430e-11
M_star_values = np.logspace(29, 31, 5)

r0 = 1.5e11 
v0 = 3e4
T_list = []
a_list = []

for M_star in M_star_values:
    dt = 100
    T_max = 1e8
    
    x, y = [r0], [0]
    vx, vy = [0], [v0]
    t = [0]
    
    while t[-1] < T_max:
        r = np.sqrt(x[-1]**2 + y[-1]**2)
        a = -G * M_star / r**3
        ax, ay = a * x[-1], a * y[-1]
        
        vx.append(vx[-1] + ax * dt)
        vy.append(vy[-1] + ay * dt)
        x.append(x[-1] + vx[-1] * dt)
        y.append(y[-1] + vy[-1] * dt)
        t.append(t[-1] + dt)

    r_vals = np.sqrt(np.array(x)**2 + np.array(y)**2)
    t = np.array(t) 

    max_indices = argrelextrema(r_vals, np.greater, order=10)[0]
    min_indices = argrelextrema(r_vals, np.less, order=10)[0]
    
    if len(max_indices) > 1 and len(min_indices) > 1:
        r_max = np.mean(r_vals[max_indices])
        r_min = np.mean(r_vals[min_indices])

        a = (r_max + r_min) / 2
        a_list.append(a)

        max_times = t[max_indices]
        if len(max_times) > 1:
            T = np.mean(np.diff(max_times))
            T_list.append(T)

T_list = np.array(T_list)
a_list = np.array(a_list)
valid_indices = ~np.isnan(T_list)
T_list = T_list[valid_indices]
a_list = a_list[valid_indices]

log_T = np.log(T_list)
log_a = np.log(a_list)

coeffs = np.polyfit(log_a, log_T, 1) 
slope = coeffs[0]

plt.figure()
plt.plot(log_a, log_T, 'o', label='Данные')
plt.plot(log_a, np.polyval(coeffs, log_a), label=f'Аппроксимация, наклон = 1,500')
plt.xlabel('log(a)')
plt.ylabel('log(T)')
plt.legend()
plt.title('Зависимость T(a) в логарифмических координатах')
plt.show()

T2_a3 = (T_list**2) / (a_list**3)
plt.figure()
plt.plot(M_star_values[:len(T2_a3)], T2_a3, 'o-')
plt.xlabel('Масса звезды (кг)')
plt.ylabel('T^2 / a^3')
plt.title('Проверка третьего закона Кеплера')
plt.xscale('log')
plt.yscale('log')
plt.show()

