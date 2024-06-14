import numpy as np
import gymnasium as gym
import fobo2_env
from stable_baselines3 import PPO

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import matplotlib.pyplot as plt


def value_in_range(value: float, center: float, offset: float):
    difference = np.abs(value - center)
    if difference <= offset:
        return True
    return False


def calculate_exponential_score(max_score, value, desired_value, k) -> float:
    delta = np.abs(value - desired_value)
    A = max_score

    if value < desired_value:
        delta = value
        A = -3

    reward = A * np.exp(-k * delta)

    return reward


def draw_circle_map():
    # Crear una malla de puntos (x, y) dentro de un círculo
    radius = 6
    num_points = 500
    theta = np.linspace(0, 2 * np.pi, num_points)
    r = np.linspace(0, radius, num_points)
    T, R = np.meshgrid(theta, r)
    X = R * np.cos(T)
    Y = R * np.sin(T)

    # Evaluar la función en cada punto de la malla, solo dependiendo del radio
    Z = np.zeros_like(R)
    for i in range(len(r)):
        y = calculate_exponential_score(
            max_score=3, value=r[i], desired_value=1.5, k=0.2
        )
        if value_in_range(value=r[i], center=1.5, offset=0.5):
            y = 5
        # if r[i] == -1:
        #     y = 0
        Z[i, :] = y

    # Crear el mapa de calor con contornos rellenos
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, levels=100, cmap="viridis")
    plt.colorbar(contour, label="Reward")

    # Añadir etiquetas y título
    plt.xlabel("Distance between human (m)")
    plt.ylabel("")
    plt.title("Reward based on distance")

    # Ocultar los números del eje y
    plt.gca().yaxis.set_visible(False)
    plt.gca().set_yticklabels([])

    # # Dibujar el círculo delimitador
    # circle = plt.Circle((0, 0), radius, color='white', fill=False, linewidth=2)
    # plt.gca().add_artist(circle)

    # Establecer límites para asegurar que el círculo esté bien representado
    plt.xlim(-radius, radius)
    plt.ylim(-radius, radius)

    # Guardar el gráfico como imagen
    plt.savefig("distance_reward_hot_map.png")


def plot_reward_function():
    # x_values = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12,13, 14]
    x_values = [-1] + [i / 1000 for i in range(0, 1000)]
    y_values = []
    for i in x_values:
        # y = calculate_exponential_score(max_score=3, value=i, desired_value=1.5, k=0.2)
        y = calculate_exponential_score(max_score=2, value=i, desired_value=0.5, k=6)

        if value_in_range(
            value=i,
            center=0.5,
            offset=0.1,
        ):
            y = 3

        if i == -1:
            y = 0

        y_values.append(y)
        print(f"for {i} distance, reward = {y}")

    x = np.array(x_values)
    y = np.array(y_values)

    # Expandir los datos a una malla bidimensional
    X, Y = np.meshgrid(x, x)
    Z = np.interp(X, x, y)  # Interpolación de los valores de Y

    plt.figure(figsize=(10, 6))
    plt.imshow(
        Z,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        aspect="auto",
        cmap="Oranges",
    )
    plt.colorbar(label="Reward")
    plt.gca().yaxis.set_visible(False)
    # Añadir etiquetas y título
    plt.xlabel("Pixel position in image")
    # plt.ylabel('Recompensa (Y)')
    plt.title("Reward function for human pixel")

    # Guardar el gráfico como imagen
    plt.savefig("pixel_reward_hot_map.png")

    # # Add a vertical line at x = 1.5, desired distance
    # plt.axvline(
    #     x=0.5, color="r", linestyle="--", label="x = 0.5"
    # )  # Adjust color and linestyle as needed

    # # Plot the function
    # plt.plot(x_values, y_values, label="r2(x)")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("reward based on pixel")
    # plt.legend()
    # plt.grid(True)
    # plt.show()


if __name__ == "__main__":
    draw_circle_map()
    # plot_reward_function()
    # plot_reward_function_px()
