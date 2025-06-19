import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def resolver_poisson_dfm(a, b, c, d, n, m, tol, N_max):
    """
    Resuelve la ecuación de Poisson utilizando el método de diferencias finitas
    según el algoritmo del documento original (Gauss-Seidel).
    """
    # Paso 1: Tamaños de paso
    h = (b - a) / n
    k = (d - c) / m

    # Paso 2 y 3: Puntos de malla
    x = np.linspace(a, b, n + 1)
    y = np.linspace(c, d, m + 1)
    x_internos = x[1:-1]
    y_internos = y[1:-1]

    # Paso 4: Inicializar w = 0 para puntos internos
    w = np.zeros((n - 1, m - 1))

    # Paso 5: Constantes
    lambda_val = h**2 / k**2
    mu = 2 * (1 + lambda_val)
    l = 1

    # Función f(x, y)
    def f(x_val, y_val):
        return -(np.cos(x_val + y_val) + np.cos(x_val - y_val))

    # Condiciones de contorno
    def g(x_val, y_val, tipo):
        if tipo == 'inferior': return np.cos(x_val)
        elif tipo == 'superior': return 0
        elif tipo == 'izquierda': return np.cos(y_val)
        elif tipo == 'derecha': return -np.cos(y_val)

    # Paso 6 al 20: Iteraciones de Gauss-Seidel
    while l <= N_max:
        NORM = 0.0
        j = m - 2  # Paso 7,8,9: fila superior

        # Paso 7
        z = (-h**2 * f(x_internos[0], y_internos[j]) +
             g(a, y_internos[j], 'izquierda') +
             lambda_val * g(x_internos[0], d, 'superior') +
             lambda_val * w[0, j - 1] +
             w[1, j]) / mu
        NORM = max(NORM, abs(z - w[0, j]))
        w[0, j] = z

        # Paso 8
        for i in range(1, n - 2):
            z = (-h**2 * f(x_internos[i], y_internos[j]) +
                 lambda_val * g(x_internos[i], d, 'superior') +
                 w[i - 1, j] + w[i + 1, j] +
                 lambda_val * w[i, j - 1]) / mu
            NORM = max(NORM, abs(z - w[i, j]))
            w[i, j] = z

        # Paso 9
        z = (-h**2 * f(x_internos[-1], y_internos[j]) +
             g(b, y_internos[j], 'derecha') +
             lambda_val * g(x_internos[-1], d, 'superior') +
             w[-2, j] + lambda_val * w[-1, j - 1]) / mu
        NORM = max(NORM, abs(z - w[-1, j]))
        w[-1, j] = z

        # Paso 10: j = m−2,...,2
        for j in range(m - 3, 0, -1):
            # Paso 11
            z = (-h**2 * f(x_internos[0], y_internos[j]) +
                 g(a, y_internos[j], 'izquierda') +
                 lambda_val * w[0, j + 1] +
                 lambda_val * w[0, j - 1] +
                 w[1, j]) / mu
            NORM = max(NORM, abs(z - w[0, j]))
            w[0, j] = z

            # Paso 12
            for i in range(1, n - 2):
                z = (-h**2 * f(x_internos[i], y_internos[j]) +
                     w[i - 1, j] + lambda_val * w[i, j + 1] +
                     w[i + 1, j] + lambda_val * w[i, j - 1]) / mu
                NORM = max(NORM, abs(z - w[i, j]))
                w[i, j] = z

            # Paso 13
            z = (-h**2 * f(x_internos[-1], y_internos[j]) +
                 g(b, y_internos[j], 'derecha') +
                 w[-2, j] + lambda_val * w[-1, j + 1] +
                 lambda_val * w[-1, j - 1]) / mu
            NORM = max(NORM, abs(z - w[-1, j]))
            w[-1, j] = z

        # Paso 14: esquina inferior izquierda
        z = (-h**2 * f(x_internos[0], y_internos[0]) +
             g(a, y_internos[0], 'izquierda') +
             lambda_val * g(x_internos[0], c, 'inferior') +
             lambda_val * w[0, 1] + w[1, 0]) / mu
        NORM = max(NORM, abs(z - w[0, 0]))
        w[0, 0] = z

        # Paso 15
        for i in range(1, n - 2):
            z = (-h**2 * f(x_internos[i], y_internos[0]) +
                 lambda_val * g(x_internos[i], c, 'inferior') +
                 w[i - 1, 0] + lambda_val * w[i, 1] +
                 w[i + 1, 0]) / mu
            NORM = max(NORM, abs(z - w[i, 0]))
            w[i, 0] = z

        # Paso 16: esquina inferior derecha
        z = (-h**2 * f(x_internos[-1], y_internos[0]) +
             g(b, y_internos[0], 'derecha') +
             lambda_val * g(x_internos[-1], c, 'inferior') +
             w[-2, 0] + lambda_val * w[-1, 1]) / mu
        NORM = max(NORM, abs(z - w[-1, 0]))
        w[-1, 0] = z

        # Paso 17: verificar convergencia
        if NORM <= tol:
            print(f"Convergencia alcanzada después de {l} iteraciones.")
            # Paso 18: salida completa con contorno
            full_w = np.zeros((n + 1, m + 1))
            full_w[1:-1, 1:-1] = w

            for i in range(n + 1):
                full_w[i, 0] = g(x[i], y[0], 'inferior')
                full_w[i, -1] = g(x[i], y[-1], 'superior')
            for j in range(m + 1):
                full_w[0, j] = g(x[0], y[j], 'izquierda')
                full_w[-1, j] = g(x[-1], y[j], 'derecha')

            return x, y, full_w

        # Paso 20
        l += 1

    # Paso 21
    print("Número máximo de iteraciones excedido.")
    return None, None, None

# Solución analítica: u(x,y) = cos(x) * cos(y)
def solucion_analitica(x, y):
    return np.cos(x) * np.cos(y)

# Parámetros del problema
a_val = 0
b_val = np.pi
c_val = 0
d_val = np.pi / 2
n_val = 20
m_val = 20
tol_val = 1e-5
N_max_val = 1000

print(f"Iniciando DFM para la ecuación de Poisson con n={n_val}, m={m_val}")
x_malla, y_malla, w_aprox = resolver_poisson_dfm(a_val, b_val, c_val, d_val, n_val, m_val, tol_val, N_max_val)

# Gráfica 3D si la solución fue encontrada
if w_aprox is not None:
    X, Y = np.meshgrid(x_malla, y_malla)
    Z_analitica = solucion_analitica(X, Y)

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z_analitica, cmap='viridis')
    ax1.set_title('Solución Analítica $u(x,y) = \cos(x)\cos(y)$')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y)')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, w_aprox.T, cmap='viridis')  # Transponer para que ejes coincidan
    ax2.set_title('Aproximación DFM')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('w(x,y)')

    plt.tight_layout()
    plt.show()
else:
    print("No se pudo obtener una aproximación.")
