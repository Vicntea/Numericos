import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def resolver_poisson_dfm(a, b, c, d, n, m, tol, N_max):
    """
    Resuelve la ecuación de Poisson utilizando el método de diferencias finitas
    siguiendo estrictamente el algoritmo proporcionado.

    Argumentos:
        a, b, c, d (float): Límites del dominio.
        n (int): Número de intervalos a lo largo del eje x (número de puntos internos n-1).
        m (int): Número de intervalos a lo largo del eje y (número de puntos internos m-1).
        tol (float): Tolerancia para la convergencia.
        N_max (int): Número máximo de iteraciones.

    Retorna:
        tuple: (x_malla, y_malla, w), donde w es la solución aproximada,
               o (None, None, None) si se excede el número máximo de iteraciones.
    """

    # Paso 1: Definir el espaciado de la malla e inicializar los puntos
    h = (b - a) / n  # 
    k = (d - c) / m  # 

    x = np.array([a + i * h for i in range(n + 1)]) # Puntos de la malla incluyendo los límites
    y = np.array([c + j * k for j in range(m + 1)]) # Puntos de la malla incluyendo los límites

    # Malla para los puntos internos
    x_internos = np.array([a + i * h for i in range(1, n)]) # 
    y_internos = np.array([c + j * k for j in range(1, m)]) # 

    # Inicializar w (aproximaciones para u(x_i, y_j)) a ceros para los puntos internos 
    # w tendrá dimensiones (n-1) x (m-1) correspondientes a los índices (1 a n-1, 1 a m-1)
    w = np.zeros((n - 1, m - 1)) # 

    # Paso 2: Establecer constantes
    lambda_val = h**2 / k**2  # 
    mu = 2 * (1 + lambda_val)  # 

    l = 1  # 
    
    # Definir la función f(x,y) de la ecuación de Poisson
    def f(x_val, y_val):
        return -(np.cos(x_val + y_val) + np.cos(x_val - y_val)) # 

    # Definir la función de condiciones de contorno g(x,y) 
    def g(x_val, y_val, tipo_contorno):
        if tipo_contorno == 'inferior': # u(x,0) = cos(x) 
            return np.cos(x_val) # 
        elif tipo_contorno == 'superior': # u(x,pi/2) = 0 
            return 0 # 
        elif tipo_contorno == 'izquierda': # u(0,y) = cos(y) 
            return np.cos(y_val) # 
        elif tipo_contorno == 'derecha': # u(pi,y) = -cos(y) 
            return -np.cos(y_val) # 
        else:
            raise ValueError("Tipo de contorno inválido")

    # Paso 3: Iteraciones de Gauss-Seidel 
    while l <= N_max: # 
        NORM = 0.0 # 

        # Iterar sobre los puntos internos (i de 0 a n-2, j de 0 a m-2)
        # El algoritmo usa indexación basada en 1 para i,j, mientras que Python usa indexación basada en 0.
        # Entonces w[i_python, j_python] corresponde a w[i_algo, j_algo] donde i_algo = i_python + 1, j_algo = j_python + 1

        # Región: j = m-1 (fila superior) 
        # Corresponde a j_python = m-2

        # Punto (1, m-1) en el algoritmo -> w[0, m-2] en Python 
        z = (-h**2 * f(x_internos[0], y_internos[m-2]) + \
             g(a, y_internos[m-2], 'izquierda') + \
             lambda_val * g(x_internos[0], d, 'superior') + \
             lambda_val * w[0, m-3] + \
             w[1, m-2]) / mu # 
        
        diff = abs(z - w[0, m-2])
        if diff > NORM: # 
            NORM = diff # 
        w[0, m-2] = z # 

        # Puntos (i, m-1) para i = 2,...,n-2 en el algoritmo -> w[i-1, m-2] en Python para i_python = 1,...,n-3 
        for i_python in range(1, n - 2): # i de 2 a n-2 en el algoritmo
            z = (-h**2 * f(x_internos[i_python], y_internos[m-2]) + \
                 lambda_val * g(x_internos[i_python], d, 'superior') + \
                 w[i_python-1, m-2] + \
                 w[i_python+1, m-2] + \
                 lambda_val * w[i_python, m-3]) / mu # 
            
            diff = abs(w[i_python, m-2] - z)
            if diff > NORM: # 
                NORM = diff # 
            w[i_python, m-2] = z # 

        # Punto (n-1, m-1) en el algoritmo -> w[n-2, m-2] en Python 
        z = (-h**2 * f(x_internos[n-2], y_internos[m-2]) + \
             g(b, y_internos[m-2], 'derecha') + \
             lambda_val * g(x_internos[n-2], d, 'superior') + \
             w[n-3, m-2] + \
             lambda_val * w[n-2, m-3]) / mu # 
        
        diff = abs(w[n-2, m-2] - z)
        if diff > NORM: # 
            NORM = diff # 
        w[n-2, m-2] = z # 


        # Región: j = m-2 hasta 2 (filas intermedias) 
        # Corresponde a j_python = m-3 hasta 1
        for j_python in range(m - 3, 0, -1): # j de m-2 hasta 2 en el algoritmo
            # Punto (1, j) en el algoritmo -> w[0, j_python] en Python 
            z = (-h**2 * f(x_internos[0], y_internos[j_python]) + \
                 g(a, y_internos[j_python], 'izquierda') + \
                 lambda_val * w[0, j_python+1] + \
                 lambda_val * w[0, j_python-1] + \
                 w[1, j_python]) / mu # 
            
            diff = abs(w[0, j_python] - z)
            if diff > NORM: # 
                NORM = diff # 
            w[0, j_python] = z # 

            # Puntos (i, j) para i = 2,...,n-2 en el algoritmo -> w[i-1, j_python] en Python para i_python = 1,...,n-3 
            for i_python in range(1, n - 2): # i de 2 a n-2 en el algoritmo
                z = (-h**2 * f(x_internos[i_python], y_internos[j_python]) + \
                     w[i_python-1, j_python] + \
                     lambda_val * w[i_python, j_python+1] + \
                     w[i_python+1, j_python] + \
                     lambda_val * w[i_python, j_python-1]) / mu # 
                
                diff = abs(w[i_python, j_python] - z)
                if diff > NORM: # 
                    NORM = diff # 
                w[i_python, j_python] = z # 

            # Punto (n-1, j) en el algoritmo -> w[n-2, j_python] en Python 
            z = (-h**2 * f(x_internos[n-2], y_internos[j_python]) + \
                 g(b, y_internos[j_python], 'derecha') + \
                 w[n-3, j_python] + \
                 lambda_val * w[n-2, j_python+1] + \
                 lambda_val * w[n-2, j_python-1]) / mu # 
            
            diff = abs(w[n-2, j_python] - z)
            if diff > NORM: # 
                NORM = diff # 
            w[n-2, j_python] = z # 

        # Región: j = 1 (fila inferior) 
        # Corresponde a j_python = 0

        # Punto (1, 1) en el algoritmo -> w[0, 0] en Python 
        z = (-h**2 * f(x_internos[0], y_internos[0]) + \
             g(a, y_internos[0], 'izquierda') + \
             lambda_val * g(x_internos[0], c, 'inferior') + \
             lambda_val * w[0, 1] + \
             w[1, 0]) / mu # 
        
        diff = abs(w[0, 0] - z)
        if diff > NORM: # 
            NORM = diff # 
        w[0, 0] = z # 

        # Puntos (i, 1) para i = 2,...,n-2 en el algoritmo -> w[i-1, 0] en Python para i_python = 1,...,n-3 
        for i_python in range(1, n - 2): # i de 2 a n-2 en el algoritmo
            z = (-h**2 * f(x_internos[i_python], y_internos[0]) + \
                 lambda_val * g(x_internos[i_python], c, 'inferior') + \
                 w[i_python-1, 0] + \
                 lambda_val * w[i_python, 1] + \
                 w[i_python+1, 0]) / mu # 
            
            diff = abs(w[i_python, 0] - z)
            if diff > NORM: # 
                NORM = diff # 
            w[i_python, 0] = z # 

        # Punto (n-1, 1) en el algoritmo -> w[n-2, 0] en Python 
        z = (-h**2 * f(x_internos[n-2], y_internos[0]) + \
             g(b, y_internos[0], 'derecha') + \
             lambda_val * g(x_internos[n-2], c, 'inferior') + \
             w[n-3, 0] + \
             lambda_val * w[n-2, 1]) / mu # 
        
        diff = abs(w[n-2, 0] - z)
        if diff > NORM: # 
            NORM = diff # 
        w[n-2, 0] = z # 

        # Verificar convergencia 
        if NORM <= tol: # 
            print(f"Convergencia alcanzada después de {l} iteraciones.")
            # Preparar la malla completa de la solución incluyendo las condiciones de contorno para la salida
            full_w = np.zeros((n+1, m+1))
            # Rellenar los puntos internos
            full_w[1:n, 1:m] = w[:, :]

            # Aplicar las condiciones de contorno a la malla full_w
            # Contorno inferior: u(x,0) = cos(x) 
            for i_idx in range(n + 1):
                full_w[i_idx, 0] = g(x[i_idx], y[0], 'inferior')

            # Contorno superior: u(x, pi/2) = 0 
            for i_idx in range(n + 1):
                full_w[i_idx, m] = g(x[i_idx], y[m], 'superior')

            # Contorno izquierdo: u(0,y) = cos(y) 
            for j_idx in range(m + 1):
                full_w[0, j_idx] = g(x[0], y[j_idx], 'izquierda')
            
            # Contorno derecho: u(pi,y) = -cos(y) 
            for j_idx in range(m + 1):
                full_w[n, j_idx] = g(x[n], y[j_idx], 'derecha')
            
            # Reajustar los valores de las esquinas para consistencia
            full_w[0,0] = np.cos(0) # u(0,0) = cos(0)
            full_w[n,0] = -np.cos(0) # u(pi,0) = -cos(0)
            full_w[0,m] = np.cos(np.pi/2) # u(0,pi/2) = cos(pi/2)
            full_w[n,m] = -np.cos(np.pi/2) # u(pi,pi/2) = -cos(pi/2)

            return x, y, full_w

        l += 1 #

    print("Número máximo de iteraciones excedido.") #
    return None, None, None


# Parte 1: Resolver analíticamente el problema planteado 
# El problema pide resolver analíticamente la ecuación de Poisson.
# La ecuación es: $\frac{\partial^{2}u}{\partial x^{2}}+\frac{\partial^{2}u}{\partial y^{2}}=-(cos(x+y)+cos(x-y))$ 
# Usando identidades trigonométricas: $cos(x+y) + cos(x-y) = 2*cos(x)*cos(y)$
# Entonces, $\frac{\partial^{2}u}{\partial x^{2}}+\frac{\partial^{2}u}{\partial y^{2}}=-2*cos(x)*cos(y)$
# Una solución analítica común para esta forma es $u(x,y) = cos(x)cos(y)$.
# Verifiquemos:
# $\frac{\partial u}{\partial x} = -sin(x)cos(y)$, $\frac{\partial^{2}u}{\partial x^{2}} = -cos(x)cos(y)$
# $\frac{\partial u}{\partial y} = -cos(x)sin(y)$, $\frac{\partial^{2}u}{\partial y^{2}} = -cos(x)cos(y)$
# Suma: $-cos(x)cos(y) - cos(x)cos(y) = -2*cos(x)*cos(y)$
# Esto coincide con el lado derecho de la ecuación.
# Ahora, comprobemos las condiciones de contorno: 
# $u(x,0) = cos(x)cos(0) = cos(x)$ -> Coincide 
# $u(x, \frac{\pi}{2}) = cos(x)cos(\frac{\pi}{2}) = 0$ -> Coincide 
# $u(0,y) = cos(0)cos(y) = cos(y)$ -> Coincide 
# $u(\pi,y) = cos(\pi)cos(y) = -cos(y)$ -> Coincide 
# Por lo tanto, la solución analítica es $u(x,y) = cos(x)cos(y)$.

def solucion_analitica(x, y):
    return np.cos(x) * np.cos(y)

# Parte 2 y 3: Implementar el algoritmo y encontrar una aproximación 
# Parámetros para el algoritmo FDM
a_val = 0
b_val = np.pi
c_val = 0
d_val = np.pi / 2
n_val = 20  # Número de intervalos en x 
m_val = 20  # Número de intervalos en y 
tol_val = 1e-5 # 
N_max_val = 1000 # 

print(f"Iniciando DFM para la ecuación de Poisson con n={n_val}, m={m_val}")
x_malla_dfm, y_malla_dfm, w_aprox = resolver_poisson_dfm(a_val, b_val, c_val, d_val, n_val, m_val, tol_val, N_max_val)

if w_aprox is not None:
    # Crear malla para graficar
    X_dfm, Y_dfm = np.meshgrid(x_malla_dfm, y_malla_dfm)

    # Calcular la solución analítica para comparación 
    Z_analitica = solucion_analitica(X_dfm, Y_dfm)

    # Parte 4: Realizar una comparación mediante una gráfica 3D 
    fig = plt.figure(figsize=(12, 6))

    # Graficar Solución Analítica
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X_dfm, Y_dfm, Z_analitica, cmap='viridis')
    ax1.set_title('Solución Analítica $u(x,y) = \cos(x)\cos(y)$')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('u(x,y)')

    # Graficar Aproximación DFM
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X_dfm, Y_dfm, w_aprox, cmap='viridis')
    ax2.set_title(f'Aproximación DFM (n={n_val}, m={m_val})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('w(x,y)')

    plt.tight_layout()
    plt.show()
else:
    print("No se pudo obtener una aproximación.")