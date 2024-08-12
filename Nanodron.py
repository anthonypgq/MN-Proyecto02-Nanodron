import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def nanodron_system(t, state, alpha, beta, gamma):
    x, y, z = state
    dxdt = alpha * (y - x)
    dydt = x * (beta - z) - y
    dzdt = x * y - gamma * z
    return [dxdt, dydt, dzdt]

def animate_trajectory(alpha, beta, gamma, initial_conditions, t_span, t_eval):
    solution = solve_ivp(nanodron_system, t_span, initial_conditions, args=(alpha, beta, gamma), t_eval=t_eval)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([min(solution.y[0]), max(solution.y[0])])
    ax.set_ylim([min(solution.y[1]), max(solution.y[1])])
    ax.set_zlim([min(solution.y[2]), max(solution.y[2])])
    
    line, = ax.plot([], [], [], lw=2)
    point, = ax.plot([], [], [], 'ro')

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return line, point

    def update(num):
        line.set_data(solution.y[0][:num], solution.y[1][:num])
        line.set_3d_properties(solution.y[2][:num])
        point.set_data(solution.y[0][num], solution.y[1][num])
        point.set_3d_properties(solution.y[2][num])
        ax.set_title(f't={t_eval[num]:.2f} seconds')
        return line, point

    ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=50)
    plt.show()

# Parámetros del caso 1
alpha_1, beta_1, gamma_1 = 0.1, 0.1, 0.1
initial_conditions_1a = [1, 1, 1]
initial_conditions_1b = [0.9, 0.9, 0.9]
t_span = (0, 10)
t_eval = np.linspace(0, 10, 500)

# Simulación y animación para la posición A
animate_trajectory(alpha_1, beta_1, gamma_1, initial_conditions_1a, t_span, t_eval)

# Simulación y animación para la posición B
animate_trajectory(alpha_1, beta_1, gamma_1, initial_conditions_1b, t_span, t_eval)

# Puedes agregar más experimentos aquí con diferentes valores de alpha, beta, gamma y condiciones iniciales.

# Caso 2
alpha_2, beta_2, gamma_2 = 10, 28, 8/3
initial_conditions_2 = [1, 1, 1]
animate_trajectory(alpha_2, beta_2, gamma_2, initial_conditions_2, t_span, t_eval)

# Caso 3 (Otro conjunto de parámetros)
alpha_3, beta_3, gamma_3 = 2, 5, 3
initial_conditions_3 = [0.8, 0.8, 0.8]
animate_trajectory(alpha_3, beta_3, gamma_3, initial_conditions_3, t_span, t_eval)