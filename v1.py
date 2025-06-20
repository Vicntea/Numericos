import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
import datetime

def resolver_poisson_dfm(a, b, c, d, n, m, tol, N_max):
    """
    Resuelve la ecuación de Poisson usando el método de diferencias finitas
    con iteraciones de Gauss-Seidel.
    """
    h = (b - a) / n  # Paso 1: Cálculo del tamaño de la malla en x 
    k = (d - c) / m  # Paso 1: Cálculo del tamaño de la malla en y 

    # Paso 2 y 3: Construcción de los puntos de la malla. 
    # x y y incluyen los puntos de frontera.
    x = np.array([a + i * h for i in range(n + 1)])
    y = np.array([c + j * k for j in range(m + 1)])

    # x_internos e y_internos son los puntos de la malla donde se calculan las aproximaciones w_i,j.
    # Corresponden a los índices 1 a n-1 para x y 1 a m-1 para y en la notación del documento.
    x_internos = np.array([a + i * h for i in range(1, n)])
    y_internos = np.array([c + j * k for j in range(1, m)])

    # Paso 4: Inicialización de las aproximaciones w_i,j a cero. 
    # w tiene dimensiones (n-1) x (m-1) para los puntos internos.
    w = np.zeros((n - 1, m - 1))

    # Paso 5: Cálculo de lambda y mu. 
    lambda_val = h**2 / k**2
    mu = 2 * (1 + lambda_val)

    l = 1  # Paso 5: Inicialización del contador de iteraciones. 
    norm_history = [] # Para almacenar el NORM en cada iteración y graficar la convergencia.

    def f(x_val, y_val):
        """Define la función f(x,y) de la ecuación de Poisson."""
        return -(np.cos(x_val + y_val) + np.cos(x_val - y_val))

    def g(x_val, y_val, tipo_contorno):
        """Define las condiciones de frontera g(x,y)."""
        if tipo_contorno == 'inferior': # u(x,0) = cos(x) 
            return np.cos(x_val)
        elif tipo_contorno == 'superior': # u(x, pi/2) = 0 
            return 0
        elif tipo_contorno == 'izquierda': # u(0,y) = cos(y) 
            return np.cos(y_val)
        elif tipo_contorno == 'derecha': # u(pi,y) = -cos(y) 
            return -np.cos(y_val)
        else:
            raise ValueError("Tipo de contorno inválido")

    # Paso 6: Inicio del bucle principal de iteraciones de Gauss-Seidel. 
    while l <= N_max:
        NORM = 0.0

        # Paso 7: Cálculo de w_1,m-1 (índice [0, m-2] en Python) 
        # Este punto es (x_1, y_m-1) en la notación del documento.
        z = (-h**2 * f(x_internos[0], y_internos[m-2]) +
             g(a, y_internos[m-2], 'izquierda') +
             lambda_val * g(x_internos[0], d, 'superior') +
             lambda_val * w[0, m-3] + # w_1,m-2 
             w[1, m-2]) / mu          # w_2,m-1 
        diff = abs(z - w[0, m-2])
        NORM = max(NORM, diff)
        w[0, m-2] = z # 

        # Paso 8: Cálculo de w_i,m-1 para i = 2,...,n-2 
        # Esto corresponde a i_python = 1,...,n-3 en Python.
        for i_python in range(1, n - 2):
            z = (-h**2 * f(x_internos[i_python], y_internos[m-2]) +
                 lambda_val * g(x_internos[i_python], d, 'superior') +
                 w[i_python-1, m-2] + # w_i-1,m-1 
                 w[i_python+1, m-2] + # w_i+1,m-1 
                 lambda_val * w[i_python, m-3]) / mu # w_i,m-2 
            diff = abs(z - w[i_python, m-2])
            NORM = max(NORM, diff)
            w[i_python, m-2] = z # 

        # Paso 9: Cálculo de w_n-1,m-1 (índice [n-2, m-2] en Python) 
        z = (-h**2 * f(x_internos[n-2], y_internos[m-2]) +
             g(b, y_internos[m-2], 'derecha') +
             lambda_val * g(x_internos[n-2], d, 'superior') +
             w[n-3, m-2] + # w_n-2,m-1 
             lambda_val * w[n-2, m-3]) / mu # w_n-1,m-2 
        diff = abs(z - w[n-2, m-2])
        NORM = max(NORM, diff)
        w[n-2, m-2] = z # 

        # Paso 10: Bucle para j = m-2,...,2 
        # Corresponde a j_python = m-3,...,1 en Python.
        for j_python in range(m - 3, 0, -1):
            # Paso 11: Cálculo de w_1,j (índice [0, j_python] en Python) 
            z = (-h**2 * f(x_internos[0], y_internos[j_python]) +
                 g(a, y_internos[j_python], 'izquierda') +
                 lambda_val * w[0, j_python+1] + # w_1,j+1 
                 lambda_val * w[0, j_python-1] + # w_1,j-1 
                 w[1, j_python]) / mu          # w_2,j 
            diff = abs(z - w[0, j_python])
            NORM = max(NORM, diff)
            w[0, j_python] = z # 

            # Paso 12: Cálculo de w_i,j para i = 2,...,n-2 
            # Corresponde a i_python = 1,...,n-3 en Python.
            for i_python in range(1, n - 2):
                z = (-h**2 * f(x_internos[i_python], y_internos[j_python]) +
                     w[i_python-1, j_python] + # w_i-1,j 
                     lambda_val * w[i_python, j_python+1] + # w_i,j+1 
                     w[i_python+1, j_python] + # w_i+1,j 
                     lambda_val * w[i_python, j_python-1]) / mu # w_i,j-1 
                diff = abs(z - w[i_python, j_python])
                NORM = max(NORM, diff)
                w[i_python, j_python] = z # 

            # Paso 13: Cálculo de w_n-1,j (índice [n-2, j_python] en Python) 
            z = (-h**2 * f(x_internos[n-2], y_internos[j_python]) +
                 g(b, y_internos[j_python], 'derecha') +
                 w[n-3, j_python] + # w_n-2,j 
                 lambda_val * w[n-2, j_python+1] + # w_n-1,j+1 
                 lambda_val * w[n-2, j_python-1]) / mu # w_n-1,j-1 
            diff = abs(z - w[n-2, j_python])
            NORM = max(NORM, diff)
            w[n-2, j_python] = z # 

        # Paso 14: Cálculo de w_1,1 (índice [0, 0] en Python) 
        z = (-h**2 * f(x_internos[0], y_internos[0]) +
             g(a, y_internos[0], 'izquierda') +
             lambda_val * g(x_internos[0], c, 'inferior') +
             lambda_val * w[0, 1] + # w_1,2 
             w[1, 0]) / mu          # w_2,1 
        diff = abs(z - w[0, 0])
        NORM = max(NORM, diff)
        w[0, 0] = z # 

        # Paso 15: Cálculo de w_i,1 para i = 2,...,n-2 
        # Corresponde a i_python = 1,...,n-3 en Python.
        for i_python in range(1, n - 2):
            z = (-h**2 * f(x_internos[i_python], y_internos[0]) +
                 lambda_val * g(x_internos[i_python], c, 'inferior') +
                 w[i_python-1, 0] + # w_i-1,1 
                 lambda_val * w[i_python, 1] + # w_i,2 
                 w[i_python+1, 0]) / mu # w_i+1,1 
            diff = abs(z - w[i_python, 0])
            NORM = max(NORM, diff)
            w[i_python, 0] = z # 

        # Paso 16: Cálculo de w_n-1,1 (índice [n-2, 0] en Python) 
        z = (-h**2 * f(x_internos[n-2], y_internos[0]) +
             g(b, y_internos[0], 'derecha') +
             lambda_val * g(x_internos[n-2], c, 'inferior') +
             w[n-3, 0] + # w_n-2,1 
             lambda_val * w[n-2, 1]) / mu # w_n-1,2 
        diff = abs(z - w[n-2, 0])
        NORM = max(NORM, diff)
        w[n-2, 0] = z # 

        norm_history.append(NORM) # Almacena el NORM de la iteración actual

        # Paso 17: Condición de convergencia. 
        if NORM <= tol:
            print(f"Convergencia alcanzada después de {l} iteraciones.")
            # Crear una matriz completa con valores de frontera.
            full_w = np.zeros((n + 1, m + 1))
            # Asignar los valores internos calculados.
            full_w[1:n, 1:m] = w[:, :]

            # Asignar los valores de frontera.
            for i_idx in range(n + 1):
                full_w[i_idx, 0] = g(x[i_idx], y[0], 'inferior')
                full_w[i_idx, m] = g(x[i_idx], y[m], 'superior')

            for j_idx in range(m + 1):
                full_w[0, j_idx] = g(x[0], y[j_idx], 'izquierda')
                full_w[n, j_idx] = g(x[n], y[j_idx], 'derecha')
            
            # Asignar los valores de las esquinas (se sobrescriben con los correctos)
            full_w[0,0] = g(x[0], y[0], 'izquierda') # o 'inferior'
            full_w[n,0] = g(x[n], y[0], 'derecha')  # o 'inferior'
            full_w[0,m] = g(x[0], y[m], 'izquierda') # o 'superior'
            full_w[n,m] = g(x[n], y[m], 'derecha')  # o 'superior'

            # Paso 18 y 19: Salida exitosa. 
            return x, y, full_w, norm_history

        l += 1 # Paso 20: Incrementa el contador de iteraciones. 

    # Paso 21: Salida de error si se excede el número máximo de iteraciones. 
    print("Número máximo de iteraciones excedido.")
    return None, None, None, norm_history # Retorna la historia de NORM incluso si no converge


def solucion_analitica(x_val, y_val):
    """
    Define la solución analítica de la ecuación de Poisson para el problema dado.
    Según el análisis, u(x,y) = cos(x)cos(y) es la solución.
    """
    return np.cos(x_val) * np.cos(y_val)

# --- Configuración para guardar las imágenes ---
def setup_output_directory():
    base_dir = "resultados_simulacion"
    run_number = 1
    output_dir = base_dir
    while os.path.exists(output_dir):
        output_dir = f"{base_dir}_{run_number}"
        run_number += 1
    os.makedirs(output_dir)
    print(f"Las imágenes se guardarán en: {output_dir}")
    return output_dir

# Parámetros del dominio y la malla 
a_val = 0
b_val = np.pi
c_val = 0
d_val = np.pi / 2
n_val = 20 # Número de subdivisiones en x 
m_val = 20 # Número de subdivisiones en y 
tol_val = 1e-5 # Tolerancia para la convergencia 
N_max_val = 1000 # Número máximo de iteraciones 

# Crear el directorio de salida
output_directory = setup_output_directory()

print(f"Iniciando DFM para la ecuación de Poisson con n={n_val}, m={m_val}")
x_malla_dfm, y_malla_dfm, w_aprox, norm_hist = resolver_poisson_dfm(a_val, b_val, c_val, d_val, n_val, m_val, tol_val, N_max_val)

if w_aprox is not None:
    # Preparación de datos para gráficos 3D y de superficie
    X_dfm, Y_dfm = np.meshgrid(x_malla_dfm, y_malla_dfm)
    Z_analitica = solucion_analitica(X_dfm, Y_dfm)

    # --- Gráficos: Solución Analítica 3D, Aproximación DFM 3D ---
    print("\nGenerando gráficas 3D de Solución Analítica y Aproximación DFM...")
    fig_3d, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), subplot_kw={'projection': '3d'})

    # Gráfico 3D de la Solución Analítica
    ax1.plot_surface(X_dfm, Y_dfm, Z_analitica, cmap='viridis', rstride=1, cstride=1, edgecolor='none')
    ax1.set_title('Solución Analítica $u(x,y) = \cos(x)\cos(y)$')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('u(x,y)')

    # Gráfico 3D de la Aproximación DFM
    ax2.plot_surface(X_dfm, Y_dfm, w_aprox, cmap='viridis', rstride=1, cstride=1, edgecolor='none')
    ax2.set_title(f'Aproximación DFM (n={n_val}, m={m_val})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('w(x,y)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'soluciones_3d.png'))
    plt.close(fig_3d) # Cierra la figura para liberar memoria

    # --- Gráfico: Error Absoluto 3D ---
    print("\nGenerando gráfica 3D del Error Absoluto...")
    error_absoluto = np.abs(w_aprox - Z_analitica)
    fig_error_3d = plt.figure(figsize=(9, 7))
    ax3_error = fig_error_3d.add_subplot(111, projection='3d')
    ax3_error.plot_surface(X_dfm, Y_dfm, error_absoluto, cmap='inferno', rstride=1, cstride=1, edgecolor='none')
    ax3_error.set_title('Error Absoluto $|u(x,y) - w(x,y)|$')
    ax3_error.set_xlabel('X')
    ax3_error.set_ylabel('Y')
    ax3_error.set_zlabel('Error')
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'error_absoluto_3d.png'))
    plt.close(fig_error_3d)

    # --- Mapas de Calor 2D: Solución Analítica, Aproximación DFM y Error Absoluto ---
    print("\nGenerando mapas de calor 2D (Solución Analítica, Aproximación DFM, Error Absoluto)...")
    fig_heatmaps, axes_heatmaps = plt.subplots(1, 3, figsize=(18, 6))

    # Mapa de calor de la Solución Analítica
    im1 = axes_heatmaps[0].imshow(Z_analitica.T, origin='lower', extent=[a_val, b_val, c_val, d_val], cmap='viridis', aspect='auto')
    axes_heatmaps[0].set_title('Mapa de calor: Solución Analítica')
    axes_heatmaps[0].set_xlabel('X')
    axes_heatmaps[0].set_ylabel('Y')
    fig_heatmaps.colorbar(im1, ax=axes_heatmaps[0])

    # Mapa de calor de la Aproximación DFM
    im2 = axes_heatmaps[1].imshow(w_aprox.T, origin='lower', extent=[a_val, b_val, c_val, d_val], cmap='viridis', aspect='auto')
    axes_heatmaps[1].set_title(f'Mapa de calor: Aproximación DFM (n={n_val}, m={m_val})')
    axes_heatmaps[1].set_xlabel('X')
    axes_heatmaps[1].set_ylabel('Y')
    fig_heatmaps.colorbar(im2, ax=axes_heatmaps[1])

    # Mapa de calor del Error Absoluto
    im3 = axes_heatmaps[2].imshow(error_absoluto.T, origin='lower', extent=[a_val, b_val, c_val, d_val], cmap='inferno', aspect='auto')
    axes_heatmaps[2].set_title('Mapa de calor: Error Absoluto $|u - w|$')
    axes_heatmaps[2].set_xlabel('X')
    axes_heatmaps[2].set_ylabel('Y')
    fig_heatmaps.colorbar(im3, ax=axes_heatmaps[2])

    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'mapas_calor_2d.png'))
    plt.close(fig_heatmaps)

    # --- Cortes Transversales (1D): Comparación de Soluciones ---
    print("\nGenerando cortes transversales (comparación 1D)...")
    fig_slices, axes_slices = plt.subplots(1, 2, figsize=(14, 6))

    # Corte en x = pi/2
    idx_x_slice = np.argmin(np.abs(x_malla_dfm - np.pi/2)) # Encuentra el índice más cercano a pi/2
    y_slice_vals = y_malla_dfm
    analytic_slice_x = Z_analitica[idx_x_slice, :]
    dfm_slice_x = w_aprox[idx_x_slice, :]

    axes_slices[0].plot(y_slice_vals, analytic_slice_x, label='Solución Analítica', color='blue')
    axes_slices[0].plot(y_slice_vals, dfm_slice_x, '--', label='Aproximación DFM', color='red')
    axes_slices[0].set_title(f'Corte en X = {x_malla_dfm[idx_x_slice]:.3f} (aprox. π/2)')
    axes_slices[0].set_xlabel('Y')
    axes_slices[0].set_ylabel('Valor de u(x,y)')
    axes_slices[0].legend()
    axes_slices[0].grid(True)

    # Corte en y = pi/4
    idx_y_slice = np.argmin(np.abs(y_malla_dfm - np.pi/4)) # Encuentra el índice más cercano a pi/4
    x_slice_vals = x_malla_dfm
    analytic_slice_y = Z_analitica[:, idx_y_slice]
    dfm_slice_y = w_aprox[:, idx_y_slice]

    axes_slices[1].plot(x_slice_vals, analytic_slice_y, label='Solución Analítica', color='blue')
    axes_slices[1].plot(x_slice_vals, dfm_slice_y, '--', label='Aproximación DFM', color='red')
    axes_slices[1].set_title(f'Corte en Y = {y_malla_dfm[idx_y_slice]:.3f} (aprox. π/4)')
    axes_slices[1].set_xlabel('X')
    axes_slices[1].set_ylabel('Valor de u(x,y)')
    axes_slices[1].legend()
    axes_slices[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'cortes_transversales.png'))
    plt.close(fig_slices)

    # --- Gráfico de Convergencia del NORM por Iteración ---
    print("\nGenerando gráfico de Convergencia del NORM...")
    if norm_hist: # Asegura que la lista no esté vacía
        fig_norm_hist = plt.figure(figsize=(10, 6))
        # Escala logarítmica en el eje Y para ver la convergencia claramente
        plt.semilogy(range(1, len(norm_hist) + 1), norm_hist, marker='o', linestyle='-', markersize=4, color='darkgreen')
        plt.axhline(y=tol_val, color='red', linestyle='--', label=f'Tolerancia ({tol_val})')
        plt.title('Convergencia del NORM (Máximo Error Absoluto entre Iteraciones)')
        plt.xlabel('Número de Iteración')
        plt.ylabel('NORM (Escala Logarítmica)')
        plt.grid(True, which="both", ls="-", alpha=0.7)
        plt.legend()
        plt.savefig(os.path.join(output_directory, 'convergencia_norm.png'))
        plt.close(fig_norm_hist)
    else:
        print("No hay historial de NORM para graficar (posiblemente N_max muy bajo o error inicial).")

else:
    print("No se pudo obtener una aproximación. Revise los parámetros o la convergencia.")

print(f"\nTodos los gráficos se han guardado en la carpeta '{output_directory}'.")