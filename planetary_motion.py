import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# Функция моделирования движения планеты
# Здесь используется простая модель движения в центральном гравитационном поле

def simulate_orbit(m_star, r0, v0, N, dt):
    G = 1  # Гравитационная постоянная в условных единицах
    r = np.zeros((N, 2))
    v = np.zeros((N, 2))
    r[0] = r0
    v[0] = v0
    
    for i in range(1, N):
        r_mag = np.linalg.norm(r[i-1])
        if r_mag == 0:
            continue  # Предотвращение деления на ноль
        a = -G * m_star * r[i-1] / r_mag**3
        v[i] = v[i-1] + a * dt
        r[i] = r[i-1] + v[i] * dt
    
    return r

# Функция нахождения максимумов и минимумов расстояния

def find_extrema(r):
    r_mag = np.linalg.norm(r, axis=1)
    if len(r_mag) == 0:
        return np.array([]), np.array([]), [], []
    maxima = argrelextrema(r_mag, np.greater)[0]
    minima = argrelextrema(r_mag, np.less)[0]
    return r_mag[maxima], r_mag[minima], maxima, minima

# Функция вычисления периода

def calculate_period(time, extrema_indices):
    if len(extrema_indices) < 2:
        return np.nan  # Если недостаточно данных
    return np.mean(np.diff(time[extrema_indices]))

# Основной код расчета
masses = [1, 2, 5, 10]  # Различные массы звезды
semimajor_axes = []
periods = []

time_step = 0.01
num_steps = 10000

for m in masses:
    r0 = np.array([1, 0])
    v0 = np.array([0, 1])
    
    r = simulate_orbit(m, r0, v0, num_steps, time_step)
    r_max, r_min, max_indices, min_indices = find_extrema(r)
    if len(r_max) == 0 or len(r_min) == 0:
        continue  # Пропуск некорректных данных
    a = (np.mean(r_max) + np.mean(r_min)) / 2
    T = calculate_period(np.arange(num_steps) * time_step, max_indices)
    
    if not np.isnan(T) and not np.isnan(a):
        semimajor_axes.append(a)
        periods.append(T)

# Проверка третьего закона Кеплера (T^2 ~ a^3)
if len(semimajor_axes) > 1 and len(periods) > 1:
    log_a = np.log(semimajor_axes)
    log_T = np.log(periods)
    
    if len(log_a) == len(log_T) and len(log_a) > 1:
        coeffs = np.polyfit(log_a, log_T, 1)
        keppler_slope = coeffs[0]
        
        # Построение графика зависимости T(a)
        plt.figure()
        plt.scatter(log_a, log_T, label='Data')
        plt.plot(log_a, np.polyval(coeffs, log_a), label=f'Fit (slope = {keppler_slope:.2f})', linestyle='dashed')
        plt.xlabel('log(a)')
        plt.ylabel('log(T)')
        plt.legend()
        plt.title('Зависимость периода обращения от большой полуоси')
        plt.show()
        
        # Проверка зависимости T^2 / a^3 от массы звезды
        T2_a3 = np.array(periods)**2 / np.array(semimajor_axes)**3

        # Линейная аппроксимация данных
        coeffs_T2_a3 = np.polyfit(masses[:len(T2_a3)], T2_a3, 1)

        # Построение графика
        plt.figure()
        plt.plot(masses[:len(T2_a3)], T2_a3, 'o-', label='Data')  # Добавлено соединение точек
        plt.plot(masses[:len(T2_a3)], np.polyval(coeffs_T2_a3, masses[:len(T2_a3)]), label=f'Fit (slope = {coeffs_T2_a3[0]:.2f})', linestyle='dashed')
        plt.xlabel('Масса звезды')
        plt.ylabel('T^2 / a^3')
        plt.legend()
        plt.title('Проверка третьего закона Кеплера')
        plt.show()
    else:
        print('Недостаточно данных для аппроксимации полиномом.')
else:
    print('Недостаточно данных для анализа.')
