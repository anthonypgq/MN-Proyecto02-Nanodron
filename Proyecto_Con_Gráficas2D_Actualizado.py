import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import messagebox

# Definición del modelo de ecuaciones diferenciales del nanodron
def nanodrone_model(state, t, alpha, beta, gamma):
    x, y, z = state
    dxdt = alpha * (y - x)
    dydt = x * (beta - z) - y
    dzdt = x * y - gamma * z
    return [dxdt, dydt, dzdt]

# Método de Euler para resolver el sistema de ecuaciones diferenciales
def euler_method(f, y0, t, args=()):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(1, n):
        dt = t[i] - t[i - 1]
        dy = np.array(f(y[i - 1], t[i - 1], *args))
        y[i] = y[i - 1] + dt * dy
    return y

# Función que se ejecuta al presionar el botón
def run_simulation():
    try:
        alpha = float(entry_alpha.get())
        beta = float(entry_beta.get())
        gamma = float(entry_gamma.get())
    except ValueError:
        messagebox.showerror("Entrada inválida", "Por favor, ingrese valores numéricos válidos para α, β, y γ.")
        return

    # Condiciones iniciales
    initial_conditions_A = [1.0, 1.0, 1.0]
    initial_conditions_B = [0.9, 0.9, 0.9]
    initial_conditions_C = [0.8, 0.8, 0.8]

    # Intervalo de tiempo
    t = np.linspace(0, 10, 1000)

    # Resolución del sistema utilizando el método de Euler
    solution_A = euler_method(nanodrone_model, initial_conditions_A, t, args=(alpha, beta, gamma))
    solution_B = euler_method(nanodrone_model, initial_conditions_B, t, args=(alpha, beta, gamma))
    solution_C = euler_method(nanodrone_model, initial_conditions_C, t, args=(alpha, beta, gamma))

    # Configuración de la figura y el eje para la animación 3D
    fig_3d = plt.figure(figsize=(8, 6))
    ax_3d = fig_3d.add_subplot(111, projection='3d', facecolor='lightgrey')

    # Límites de los ejes
    x_min, x_max = min(solution_A[:, 0].min(), solution_B[:, 0].min(), solution_C[:, 0].min()), max(
        solution_A[:, 0].max(), solution_B[:, 0].max(), solution_C[:, 0].max())
    y_min, y_max = min(solution_A[:, 1].min(), solution_B[:, 1].min(), solution_C[:, 1].min()), max(
        solution_A[:, 1].max(), solution_B[:, 1].max(), solution_C[:, 1].max())
    z_min, z_max = min(solution_A[:, 2].min(), solution_B[:, 2].min(), solution_C[:, 2].min()), max(
        solution_A[:, 2].max(), solution_B[:, 2].max(), solution_C[:, 2].max())

    ax_3d.set_xlim(x_min, x_max)
    ax_3d.set_ylim(y_min, y_max)
    ax_3d.set_zlim(z_min, z_max)

    # Etiquetas y título
    ax_3d.set_xlabel('X', fontsize=14, color='darkblue')
    ax_3d.set_ylabel('Y', fontsize=14, color='darkgreen')
    ax_3d.set_zlabel('Z', fontsize=14, color='darkred')
    ax_3d.set_title('Trayectoria del Nanodron', fontsize=16, fontweight='bold', color='navy')

    # Inicialización de las líneas para las trayectorias
    line_A_3d, = ax_3d.plot([], [], [], label='Trayectoria A', color='orange', linewidth=2)
    line_B_3d, = ax_3d.plot([], [], [], label='Trayectoria B', color='purple', linewidth=2)
    line_C_3d, = ax_3d.plot([], [], [], label='Trayectoria C', color='green', linewidth=2)

    # Inicialización de las líneas para las proyecciones
    projection_A_x, = ax_3d.plot([], [], [], '--', color='orange', alpha=0.5)
    projection_A_y, = ax_3d.plot([], [], [], '--', color='orange', alpha=0.5)
    projection_A_z, = ax_3d.plot([], [], [], '--', color='orange', alpha=0.5)
    projection_B_x, = ax_3d.plot([], [], [], '--', color='purple', alpha=0.5)
    projection_B_y, = ax_3d.plot([], [], [], '--', color='purple', alpha=0.5)
    projection_B_z, = ax_3d.plot([], [], [], '--', color='purple', alpha=0.5)
    projection_C_x, = ax_3d.plot([], [], [], '--', color='green', alpha=0.5)
    projection_C_y, = ax_3d.plot([], [], [], '--', color='green', alpha=0.5)
    projection_C_z, = ax_3d.plot([], [], [], '--', color='green', alpha=0.5)

    # Añadir puntos para posiciones iniciales
    ax_3d.scatter(initial_conditions_A[0], initial_conditions_A[1], initial_conditions_A[2], color='red', s=100, edgecolor='black', label='Inicio A')
    ax_3d.scatter(initial_conditions_B[0], initial_conditions_B[1], initial_conditions_B[2], color='blue', s=100, edgecolor='black', label='Inicio B')
    ax_3d.scatter(initial_conditions_C[0], initial_conditions_C[1], initial_conditions_C[2], color='green', s=100, edgecolor='black', label='Inicio C')

    # Añadir una cuadrícula y cambiar el estilo
    ax_3d.grid(color='gray', linestyle='--', linewidth=0.5)

    # Texto para los parámetros del sistema
    params_text = ax_3d.text2D(0.05, 0.85, '', transform=ax_3d.transAxes, fontsize=12, color='black', backgroundcolor='white')

    # Indicador del tiempo transcurrido
    time_template = 'Tiempo = %.1f s'
    time_text = ax_3d.text2D(0.05, 0.80, '', transform=ax_3d.transAxes, fontsize=12, color='black', backgroundcolor='white')

    # Puntos finales
    final_point_A, = ax_3d.plot([], [], [], 'o', color='yellow', markersize=8, label='Final A')
    final_point_B, = ax_3d.plot([], [], [], 'o', color='green', markersize=8, label='Final B')
    final_point_C, = ax_3d.plot([], [], [], 'o', color='blue', markersize=8, label='Final C')

    # Función de inicialización para la animación
    def init_3d():
        line_A_3d.set_data([], [])
        line_A_3d.set_3d_properties([])
        line_B_3d.set_data([], [])
        line_B_3d.set_3d_properties([])
        line_C_3d.set_data([], [])
        line_C_3d.set_3d_properties([])
        projection_A_x.set_data([], [])
        projection_A_y.set_data([], [])
        projection_A_z.set_data([], [])
        projection_B_x.set_data([], [])
        projection_B_y.set_data([], [])
        projection_B_z.set_data([], [])
        projection_C_x.set_data([], [])
        projection_C_y.set_data([], [])
        projection_C_z.set_data([], [])
        final_point_A.set_data([], [])
        final_point_A.set_3d_properties([])
        final_point_B.set_data([], [])
        final_point_B.set_3d_properties([])
        final_point_C.set_data([], [])
        final_point_C.set_3d_properties([])
        time_text.set_text('')
        params_text.set_text('')
        return (line_A_3d, line_B_3d, line_C_3d, projection_A_x, projection_A_y, projection_A_z,
                projection_B_x, projection_B_y, projection_B_z, projection_C_x, projection_C_y,
                projection_C_z, final_point_A, final_point_B, final_point_C, time_text, params_text)

    # Función de actualización para la animación
    def update_3d(num):
        line_A_3d.set_data(solution_A[:num, 0], solution_A[:num, 1])
        line_A_3d.set_3d_properties(solution_A[:num, 2])
        line_B_3d.set_data(solution_B[:num, 0], solution_B[:num, 1])
        line_B_3d.set_3d_properties(solution_B[:num, 2])
        line_C_3d.set_data(solution_C[:num, 0], solution_C[:num, 1])
        line_C_3d.set_3d_properties(solution_C[:num, 2])

        # Proyecciones desde el nanodron A hacia los ejes
        projection_A_x.set_data([solution_A[num, 0], solution_A[num, 0]], [solution_A[num, 1], solution_A[num, 1]])
        projection_A_x.set_3d_properties([z_min, solution_A[num, 2]])
        projection_A_y.set_data([solution_A[num, 0], solution_A[num, 0]], [y_min, solution_A[num, 1]])
        projection_A_y.set_3d_properties([solution_A[num, 2], solution_A[num, 2]])
        projection_A_z.set_data([x_min, solution_A[num, 0]], [solution_A[num, 1], solution_A[num, 1]])
        projection_A_z.set_3d_properties([solution_A[num, 2], solution_A[num, 2]])

        # Proyecciones desde el nanodron B hacia los ejes
        projection_B_x.set_data([solution_B[num, 0], solution_B[num, 0]], [solution_B[num, 1], solution_B[num, 1]])
        projection_B_x.set_3d_properties([z_min, solution_B[num, 2]])
        projection_B_y.set_data([solution_B[num, 0], solution_B[num, 0]], [y_min, solution_B[num, 1]])
        projection_B_y.set_3d_properties([solution_B[num, 2], solution_B[num, 2]])
        projection_B_z.set_data([x_min, solution_B[num, 0]], [solution_B[num, 1], solution_B[num, 1]])
        projection_B_z.set_3d_properties([solution_B[num, 2], solution_B[num, 2]])

        # Proyecciones desde el nanodron C hacia los ejes
        projection_C_x.set_data([solution_C[num, 0], solution_C[num, 0]], [solution_C[num, 1], solution_C[num, 1]])
        projection_C_x.set_3d_properties([z_min, solution_C[num, 2]])
        projection_C_y.set_data([solution_C[num, 0], solution_C[num, 0]], [y_min, solution_C[num, 1]])
        projection_C_y.set_3d_properties([solution_C[num, 2], solution_C[num, 2]])
        projection_C_z.set_data([x_min, solution_C[num, 0]], [solution_C[num, 1], solution_C[num, 1]])
        projection_C_z.set_3d_properties([solution_C[num, 2], solution_C[num, 2]])

        final_point_A.set_data(solution_A[num-1:num, 0], solution_A[num-1:num, 1])
        final_point_A.set_3d_properties(solution_A[num-1:num, 2])
        final_point_B.set_data(solution_B[num-1:num, 0], solution_B[num-1:num, 1])
        final_point_B.set_3d_properties(solution_B[num-1:num, 2])
        final_point_C.set_data(solution_C[num-1:num, 0], solution_C[num-1:num, 1])
        final_point_C.set_3d_properties(solution_C[num-1:num, 2])
        
        time_text.set_text(time_template % t[num])
        params_text.set_text(f'α = {alpha:.2f}\nβ = {beta:.2f}\nγ = {gamma:.2f}')
        return (line_A_3d, line_B_3d, line_C_3d, projection_A_x, projection_A_y, projection_A_z,
                projection_B_x, projection_B_y, projection_B_z, projection_C_x, projection_C_y,
                projection_C_z, final_point_A, final_point_B, final_point_C, time_text, params_text)

    # Animación
    ani_3d = FuncAnimation(fig_3d, update_3d, frames=len(t), init_func=init_3d, blit=True, interval=20, repeat=False)

    # Configuración de la figura y los ejes para las gráficas 2D
    fig_2d, axes_2d = plt.subplots(3, 1, figsize=(6, 6))

    # Configuración de las gráficas 2D
    for ax, (xlabel, ylabel, title) in zip(axes_2d, [('X', 'Y', 'Plano XY'), ('X', 'Z', 'Plano XZ'), ('Y', 'Z', 'Plano YZ')]):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        ax.set_xlim(min(x_min, y_min, z_min), max(x_max, y_max, z_max))
        ax.set_ylim(min(x_min, y_min, z_min), max(x_max, y_max, z_max))

    line_A_xy, = axes_2d[0].plot([], [], label='Trayectoria A', color='orange', linewidth=2)
    line_B_xy, = axes_2d[0].plot([], [], label='Trayectoria B', color='purple', linewidth=2)
    line_C_xy, = axes_2d[0].plot([], [], label='Trayectoria C', color='green', linewidth=2)
    line_A_xz, = axes_2d[1].plot([], [], label='Trayectoria A', color='orange', linewidth=2)
    line_B_xz, = axes_2d[1].plot([], [], label='Trayectoria B', color='purple', linewidth=2)
    line_C_xz, = axes_2d[1].plot([], [], label='Trayectoria C', color='green', linewidth=2)
    line_A_yz, = axes_2d[2].plot([], [], label='Trayectoria A', color='orange', linewidth=2)
    line_B_yz, = axes_2d[2].plot([], [], label='Trayectoria B', color='purple', linewidth=2)
    line_C_yz, = axes_2d[2].plot([], [], label='Trayectoria C', color='green', linewidth=2)

    # Función de inicialización para las gráficas 2D
    def init_2d():
        line_A_xy.set_data([], [])
        line_B_xy.set_data([], [])
        line_C_xy.set_data([], [])
        line_A_xz.set_data([], [])
        line_B_xz.set_data([], [])
        line_C_xz.set_data([], [])
        line_A_yz.set_data([], [])
        line_B_yz.set_data([], [])
        line_C_yz.set_data([], [])
        return line_A_xy, line_B_xy, line_C_xy, line_A_xz, line_B_xz, line_C_xz, line_A_yz, line_B_yz, line_C_yz

    # Función de actualización para las gráficas 2D
    def update_2d(num):
        line_A_xy.set_data(solution_A[:num, 0], solution_A[:num, 1])
        line_B_xy.set_data(solution_B[:num, 0], solution_B[:num, 1])
        line_C_xy.set_data(solution_C[:num, 0], solution_C[:num, 1])
        line_A_xz.set_data(solution_A[:num, 0], solution_A[:num, 2])
        line_B_xz.set_data(solution_B[:num, 0], solution_B[:num, 2])
        line_C_xz.set_data(solution_C[:num, 0], solution_C[:num, 2])
        line_A_yz.set_data(solution_A[:num, 1], solution_A[:num, 2])
        line_B_yz.set_data(solution_B[:num, 1], solution_B[:num, 2])
        line_C_yz.set_data(solution_C[:num, 1], solution_C[:num, 2])
        return line_A_xy, line_B_xy, line_C_xy, line_A_xz, line_B_xz, line_C_xz, line_A_yz, line_B_yz, line_C_yz

    # Animación para las gráficas 2D
    ani_2d = FuncAnimation(fig_2d, update_2d, frames=len(t), init_func=init_2d, blit=True, interval=20, repeat=False)

    # Mostrar leyenda y graficar
    ax_3d.legend()
    for ax in axes_2d:
        ax.legend()

    plt.tight_layout()
    plt.show()

# Interfaz gráfica utilizando tkinter
root = tk.Tk()
root.title("Simulación de Trayectoria del Nanodron")
root.geometry("600x400")

# Título de la aplicación
title_label = tk.Label(root, text="Simulación de Trayectoria del Nanodron", font=("Arial", 18, "bold"))
title_label.grid(row=0, columnspan=2, pady=20)

# Campos de entrada para α, β y γ
tk.Label(root, text="α (Alpha):", font=("Arial", 14)).grid(row=1, column=0, padx=20, pady=10)
entry_alpha = tk.Entry(root, font=("Arial", 14))
entry_alpha.grid(row=1, column=1, padx=20, pady=10)

tk.Label(root, text="β (Beta):", font=("Arial", 14)).grid(row=2, column=0, padx=20, pady=10)
entry_beta = tk.Entry(root, font=("Arial", 14))
entry_beta.grid(row=2, column=1, padx=20, pady=10)

tk.Label(root, text="γ (Gamma):", font=("Arial", 14)).grid(row=3, column=0, padx=20, pady=10)
entry_gamma = tk.Entry(root, font=("Arial", 14))
entry_gamma.grid(row=3, column=1, padx=20, pady=10)

# Botón para ejecutar la simulación
run_button = tk.Button(root, text="Ejecutar Simulación", font=("Arial", 14), command=run_simulation)
run_button.grid(row=4, columnspan=2, pady=20)

# Etiquetas de los integrantes del grupo
group_label = tk.Label(root, text="Integrantes del Grupo:", font=("Arial", 12, "bold"))
group_label.grid(row=5, columnspan=2)

members_label = tk.Label(root, text="Anthony Goyes, David Arciniegas, Ozzy Loachamin, Angel Falcon", font=("Arial", 12))
members_label.grid(row=6, columnspan=2)

root.mainloop()
