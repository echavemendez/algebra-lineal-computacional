import numpy as np
import matplotlib.cm as cm
import scipy
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib import gridspec
import networkx as nx
import geopandas as gpd

def construye_adyacencia(D,m): 
    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-ésimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)
#%%
def calculaLU(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]  # Obtiene el número de filas (y columnas) de la matriz cuadrada A.

    L = np.identity(n)  # Inicializa la matriz L como la matriz identidad de tamaño n x n.
    P = np.identity(n)  # Inicializa la matriz de permutación P como la matriz identidad.
    U = A.copy()  # Crea una copia de A para trabajar en la matriz U.

    for i in range(n):  # Itera sobre cada columna de la matriz.
        U, P = permuta(U, P, i)  # Llama a la función 'permuta', reordenando las filas para buscar el máximo.
                                 # Así se minimiza el error de representación.

        for j in range(i + 1, n):  # Itera sobre las filas por debajo de la fila i.

            if U[i, i] == 0:
              raise ValueError(f"Error: U[i,i] == {U[i,i]}.") # Atrapa un caso (que nunca debería ocurrir) de diagonal = 0.

            factor = U[j, i] / U[i, i]  # Calcula el factor de eliminación.
            L[j, i] = factor  # Almacena el factor en la matriz L.
            U[j, i:] -= U[i, i:] * factor  # Resta el múltiplo de la fila i de U
                                           # a la fila j para obtener un cero.

    return L, U  # Devuelve las matrices L, U 

def permuta(A: np.ndarray, P: np.ndarray, fila: int) -> tuple[np.ndarray, np.ndarray]:
    A = A.copy()  # Crea una copia de la matriz A para no modificar la original.
    P = P.copy()  # Crea una copia de la matriz P para no modificar la original.

    filaMax = maximoCoefColumna(A, fila)  # Encuentra la fila con el máximo coeficiente
                                          # en la columna 'fila' de la matriz A.

    # Intercambia las filas en la matriz de permutación P y en la matriz A
    # usando la función 'cambiaFilas', que toma las matrices y las filas a intercambiar.
    if fila != filaMax:
      P = cambiaFilas(P, fila, filaMax)
      A = cambiaFilas(A, fila, filaMax)

    return A, P  # Devuelve las matrices P y A después de la permutación.

def cambiaFilas(M: np.ndarray, fila1: int, fila2: int) -> np.ndarray:
    M[[fila1, fila2], :] = M[[fila2, fila1], :] # Realiza el intercambio de las filas fila1 y fila2 en la matriz M.
    return M  # Devuelve la matriz M con las filas intercambiadas.

def maximoCoefColumna(A: np.ndarray, col: int) -> int:
    #testarray(A, col) #

    return np.argmax(np.abs(A[col:, col])) + col  # np.argmax(A) devuelve el indice del valor máximo de un array
                                                  # Se le suma "col" ya que el "0" de este array es la columna diagonal.

def inversaLU(B: np.ndarray) -> np.ndarray:
    # Calculamos la inversa de una matriz B usando factorización LU 
    n = B.shape[0]
    L, U = calculaLU(B)  
    I = np.identity(n)
    B_inv = np.zeros_like(B)

    for i in range(n):
        e = I[:, i]  # Vector columna e_i
        # Resolvemos Ly = e 
        y = np.zeros(n)
        for j in range(n):
            y[j] = e[j] - np.dot(L[j, :j], y[:j])
        
        # Resolvemos Ux = y 
        x = np.zeros(n)
        for j in reversed(range(n)):
            x[j] = (y[j] - np.dot(U[j, j+1:], x[j+1:])) / U[j, j]

        B_inv[:, i] = x  # Guardamos la columna i-ésima

    return B_inv
#%%
def calcula_matriz_C(A:np.ndarray) -> np.ndarray: 
    # Función para calcular la matriz de transiciones C
    # A: Matriz de adyacencia
    # Retorna la matriz C
    grados = np.sum(A, axis=1) # Vector con suma de cada fila 
    Kinv = np.diag(1 / grados) # Matriz diagonal con los inversos de los grados
    C = A.T @ Kinv             # Transpuesta de A por Kinv, da matriz de transición por columnas
    return C

def calcula_pagerank(A: np.ndarray, alfa: float) -> np.array:
    # Función para calcular PageRank usando factorización LU
    # A: Matriz de adyacencia
    # alfa: coeficiente de amortiguación (damping)
    # Retorna: Un vector p con los coeficientes de PageRank de cada museo

    C = calcula_matriz_C(A)                         # Matriz de transiciones
    N = A.shape[0]                                   # Cantidad de nodos
    M = (N / alfa) * (np.eye(N) - (1 - alfa) * C)    # Construcción de M

    L, U = calculaLU(M)                              # Factorización LU
    b = np.ones(N)                                   # Vector de unos

    Up = scipy.linalg.solve_triangular(L, b, lower=True)  # Resolución Ly = b
    p = scipy.linalg.solve_triangular(U, Up)              # Resolución Ux = y

    return p
#%%
def calcula_3a(D, G, museos, barrios):
    # Construimos la matriz de adyacencia
    A = construye_adyacencia(D, 3)
    # Calculamos PageRank
    alfa = 1 / 5
    p = calcula_pagerank(A, alfa)
    pr = p / p.sum()  # Normalizamos

    # Obtenemos coordenadas reales (proyección EPSG:22184)
    museos_proj = museos.to_crs("EPSG:22184")
    G_layout = {i: (geom.x, geom.y) for i, geom in enumerate(museos_proj.geometry)}

    # Preparamos la figura
    factor_escala = 1e4
    fig, ax = plt.subplots(figsize=(10, 10))

    # Dibujamos los barrios
    barrios.to_crs("EPSG:22184").boundary.plot(color='gray', ax=ax)

    # Dibujamos la red
    nx.draw_networkx(G, G_layout, node_size=pr * factor_escala, ax=ax, with_labels=False)

    plt.title("Red de museos con PageRank", fontsize=14)
    plt.axis("off")
    plt.show()

def calcula_top3_m(D: np.ndarray, museos) -> tuple:
    valores_m = [1, 3, 5, 10]
    top3_por_m = []
    pagerank_por_m = {}
    museos_top3_m = set()  # Conjunto para almacenar los museos que alguna vez estuvieron en el top 3
    
    for m in valores_m:
        A = construye_adyacencia(D, m)  
        p = calcula_pagerank(A, 0.2)  
        pr = p / p.sum()
        pagerank_por_m[m] = pr
        
        # Arrancamos con 3 lugares vacíos con score muy bajo
        top3 = [(-1, -1.0), (-1, -1.0), (-1, -1.0)]  # (indice, score)
    
        for j in range(len(pr)):
            score_actual = pr[j]
    
            # Si es más grande que alguno de los 3, lo metemos donde corresponde
            if score_actual > top3[0][1]:
                top3[2] = top3[1]
                top3[1] = top3[0]
                top3[0] = (j, score_actual)
            elif score_actual > top3[1][1]:
                top3[2] = top3[1]
                top3[1] = (j, score_actual)
            elif score_actual > top3[2][1]:
                top3[2] = (j, score_actual)
        
        # Agregar los museos del top 3 a la lista
        top3_por_m.append(tuple(top3))
        museos_top3_m.update([top3[0][0], top3[1][0], top3[2][0]])  # Almacenar los índices de los museos
    
    # Llamamos a la función de graficado después de realizar los cálculos
    graficar_pagerank(valores_m, pagerank_por_m, museos_top3_m, museos)
    
def graficar_pagerank(valores_m, pagerank_por_m, museos_top3_m, museos):
    # Generamos una lista de colores para cada museo
    colores = cm.get_cmap('tab20', len(museos_top3_m))  # Usamos un mapa de colores con tantos colores como museos

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])  # ancho 3:1

    ax_graph = fig.add_subplot(gs[0])
    for i, museo_id in enumerate(museos_top3_m):
        pr_vals = [pagerank_por_m[m][museo_id] for m in valores_m]
        ax_graph.plot(valores_m, pr_vals, marker='o', label=f"Museo {museo_id}", color=colores(i))  # Asignamos color único

    ax_graph.set_xlabel('m [cantidad de vecinos]')
    ax_graph.set_ylabel('PageRank')
    ax_graph.set_title('Evolución del PageRank en función de m (para museos alguna vez en el Top 3)')
    ax_graph.grid(True)
    ax_graph.legend(title="Museos")  # Añadimos leyenda para los museos

    ax_text = fig.add_subplot(gs[1])
    ax_text.axis("off")

    # Mostramos los nombres de los museos y sus colores
    nombres = [museos.iloc[i]["name"] for i in museos_top3_m]
    texto = "\n".join([f"Museo {i}: {nombre}" for i, nombre in zip(museos_top3_m, nombres)])

    ax_text.text(0, 1, texto, fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.show()

def calcula_top3_alpha(D: np.ndarray, museos) -> tuple:
    m = 5  # Fijamos m en 5
    alphas = [6/7, 4/5, 2/3, 1/2, 1/3, 1/5, 1/7]  # Lista de valores de alpha
    top3_por_alpha = []
    pagerank_por_alpha = {}
    museos_top3_alpha = set()  # Conjunto para almacenar los museos que alguna vez estuvieron en el top 3
    
    for alpha in alphas:
        A = construye_adyacencia(D, m)  
        p = calcula_pagerank(A, alpha)  
        pr = p / p.sum()
        pagerank_por_alpha[alpha] = pr
        
        # Arrancamos con 3 lugares vacíos con score muy bajo
        top3 = [(-1, -1.0), (-1, -1.0), (-1, -1.0)]  # (indice, score)
    
        for j in range(len(pr)):
            score_actual = pr[j]
    
            # Si es más grande que alguno de los 3, lo metemos donde corresponde
            if score_actual > top3[0][1]:
                top3[2] = top3[1]
                top3[1] = top3[0]
                top3[0] = (j, score_actual)
            elif score_actual > top3[1][1]:
                top3[2] = top3[1]
                top3[1] = (j, score_actual)
            elif score_actual > top3[2][1]:
                top3[2] = (j, score_actual)
        
        # Agregamos los museos del top 3 a la lista
        top3_por_alpha.append(tuple(top3))
        museos_top3_alpha.update([top3[0][0], top3[1][0], top3[2][0]])  # Almacenamos los índices de los museos
    
    # Llamamos a la función de graficado después de realizar los cálculos
    graficar_pagerank_alpha(alphas, pagerank_por_alpha, museos_top3_alpha, museos)
    
def graficar_pagerank_alpha(alphas, pagerank_por_alpha, museos_top3_alpha, museos):
    # Generamos una lista de colores para cada museo
    colores = cm.get_cmap('tab20', len(museos_top3_alpha))  # Usamos un mapa de colores con tantos colores como museos

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])  # ancho 3:1

    ax_graph = fig.add_subplot(gs[0])
    for i, museo_id in enumerate(museos_top3_alpha):
        pr_vals = [pagerank_por_alpha[alpha][museo_id] for alpha in alphas]
        ax_graph.plot(alphas, pr_vals, marker='o', label=f"Museo {museo_id}", color=colores(i))  # Asignamos color único

    ax_graph.set_xlabel('Alpha [factor de amortiguamiento]')
    ax_graph.set_ylabel('PageRank')
    ax_graph.set_title('Evolución del PageRank en función de alpha (para museos alguna vez en el Top 3)')
    ax_graph.grid(True)
    ax_graph.legend(title="Museos")  # Añadimos leyenda para los museos

    ax_text = fig.add_subplot(gs[1])
    ax_text.axis("off")

    # Mostramos los nombres de los museos y sus colores
    nombres = [museos.iloc[i]["name"] for i in museos_top3_alpha]
    texto = "\n".join([f"Museo {i}: {nombre}" for i, nombre in zip(museos_top3_alpha, nombres)])

    ax_text.text(0, 1, texto, fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.show()


def visualizar_red_m_variable(D, museos, barrios):
    valores_m = [1, 3, 5, 10]
    alpha_fijo = 1/5

    # Proyectamos museos y barrios en el CRS EPSG:22184
    museos_proj = museos.to_crs("EPSG:22184")
    barrios_proj = barrios.to_crs("EPSG:22184")

    # Configuración para graficar en subgráficos
    fig, axes = plt.subplots(1, len(valores_m), figsize=(20, 10))

    for i, m in enumerate(valores_m):
        ax = axes[i]

        # Construimos la matriz de adyacencia con m vecinos más cercanos
        A = construye_adyacencia(D, m)

        # Calculamos el PageRank para este m con el alpha fijo
        pagerank = calcula_pagerank(A, alpha_fijo)

        # Creamos el grafo de la red
        G = nx.from_numpy_array(A)

        # Layout usando las coordenadas reales de los museos
        G_layout = {i: (geom.x, geom.y) for i, geom in enumerate(museos_proj.geometry)}

        # Normalizamos el PageRank
        pr = pagerank / pagerank.sum()

        # Graficamos barrios
        barrios_proj.boundary.plot(color='gray', ax=ax)

        # Dibujamos la red de museos con PageRank (sin etiquetas)
        nx.draw_networkx(G, G_layout, node_size=pr * 1e4, ax=ax, with_labels=False, node_color='skyblue')

        # Título de cada subgráfico
        ax.set_title(f"Red con m={m} y α={alpha_fijo}", fontsize=14)
        ax.axis("off")  # Oculta los ejes para una mejor presentación

    plt.tight_layout()
    plt.show()


def visualizar_red_alpha_variable(D, museos, barrios):
    alphas = [6/7, 4/5, 2/3, 1/2, 1/3, 1/5, 1/7]
    m_fijo = 5

    # Proyectamos museos y barrios en el CRS EPSG:22184
    museos_proj = museos.to_crs("EPSG:22184")
    barrios_proj = barrios.to_crs("EPSG:22184")

    # Configuración para graficar en subgráficos: 2 filas, 4 columnas
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()  # Para poder indexar linealmente

    for i, alpha in enumerate(alphas):
        ax = axes[i]

        # Construimos la matriz de adyacencia con m vecinos más cercanos
        A = construye_adyacencia(D, m_fijo)

        # Calculamos el PageRank para este alpha con el m fijo
        pagerank = calcula_pagerank(A, alpha)

        # Creamos el grafo de la red
        G = nx.from_numpy_array(A)

        # Layout usando las coordenadas reales de los museos
        G_layout = {i: (geom.x, geom.y) for i, geom in enumerate(museos_proj.geometry)}

        # Normalizamos el PageRank
        pr = pagerank / pagerank.sum()

        # Graficamos barrios
        barrios_proj.boundary.plot(color='gray', ax=ax)

        # Dibujamos la red de museos con PageRank (sin etiquetas)
        nx.draw_networkx(G, G_layout, node_size=pr * 1e4, ax=ax, with_labels=False, node_color='skyblue')

        # Título del subgráfico
        ax.set_title(f"Red con m={m_fijo} y α={alpha:.2f}", fontsize=14)
        ax.axis("off")  # Oculta los ejes

    # Ocultar subplot vacío si hay uno
    if len(alphas) < len(axes):
        for j in range(len(alphas), len(axes)):
            axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def calcula_matriz_C_continua(D: np.ndarray) -> np.ndarray: 
    # Funcion para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C en version continua
    # D: Matriz de distancias entre museos
    D = D.copy()
    np.fill_diagonal(D, np.inf)  # Evita dividir por cero en la diagonal
    F = 1 / D
    np.fill_diagonal(F, 0)       # La diagonal de F debe ser 0 (no hay transición a sí mismo)
    grados = np.sum(F, axis=1)   # Vector con suma de cada fila
    Kinv = np.diag(1 / grados)   # Matriz diagonal con los inversos de los grados
    C = F.T @ Kinv                   # Calcula C multiplicando Kinv y F
    return C



def calcula_B(C: np.ndarray, cantidad_de_visitas: int) -> np.ndarray:
    # Recibe la matriz C de transiciones, y calcula la matriz B que representa la relacion 
    # entre el total de visitas y el numero inicial de visitantes
    # suponiendo que cada visitante realiza "cantidad_de_visitas" pasos
    # C: Matirz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    B = np.eye(C.shape[0])
    C_potencia = np.eye(C.shape[0])
    for _ in range(1, cantidad_de_visitas):    # Desde 1 hasta cantidad_de_visitas
        C_potencia = C_potencia @ C   # C^i = C^{i-1} @ C vamos acumulando las "cantidad_de_visitas" distintas potencias de C en B
        B += C_potencia     
    return B

w = np.array([3.866000000000000000e+03, 2.310000000000000000e+03, 3.922000000000000000e+03, 2.942000000000000000e+03, 1.932000000000000000e+03, 2.779000000000000000e+03, 3.209000000000000000e+03, 5.472000000000000000e+03, 1.862000000000000000e+03, 2.501000000000000000e+03, 2.218000000000000000e+03, 2.575000000000000000e+03, 2.362000000000000000e+03, 3.039000000000000000e+03, 2.781000000000000000e+03, 3.449000000000000000e+03, 3.811000000000000000e+03, 3.208000000000000000e+03, 4.049000000000000000e+03, 3.120000000000000000e+03, 1.922000000000000000e+03, 3.224000000000000000e+03, 2.166000000000000000e+03, 2.919000000000000000e+03, 2.731000000000000000e+03, 3.053000000000000000e+03, 1.275000000000000000e+03, 2.659000000000000000e+03, 2.073000000000000000e+03, 2.873000000000000000e+03, 2.620000000000000000e+03, 2.185000000000000000e+03, 3.097000000000000000e+03, 2.773000000000000000e+03, 4.572000000000000000e+03, 3.800000000000000000e+03, 2.586000000000000000e+03, 1.582000000000000000e+03, 2.986000000000000000e+03, 2.509000000000000000e+03, 3.345000000000000000e+03, 3.290000000000000000e+03, 3.682000000000000000e+03, 3.607000000000000000e+03, 2.795000000000000000e+03, 2.165000000000000000e+03, 2.884000000000000000e+03, 3.533000000000000000e+03, 2.631000000000000000e+03, 2.615000000000000000e+03, 1.554000000000000000e+03, 3.938000000000000000e+03, 2.463000000000000000e+03, 1.842000000000000000e+03, 2.765000000000000000e+03, 2.736000000000000000e+03, 2.998000000000000000e+03, 2.512000000000000000e+03, 2.884000000000000000e+03, 3.158000000000000000e+03, 1.276000000000000000e+03, 3.169000000000000000e+03, 2.956000000000000000e+03, 3.304000000000000000e+03, 3.112000000000000000e+03, 4.547000000000000000e+03, 3.267000000000000000e+03, 3.245000000000000000e+03, 2.441000000000000000e+03, 3.752000000000000000e+03, 2.937000000000000000e+03, 4.926000000000000000e+03, 3.830000000000000000e+03, 3.335000000000000000e+03, 4.975000000000000000e+03, 2.706000000000000000e+03, 2.097000000000000000e+03, 2.399000000000000000e+03, 3.147000000000000000e+03, 2.539000000000000000e+03, 2.351000000000000000e+03, 2.178000000000000000e+03, 2.647000000000000000e+03, 3.660000000000000000e+03, 1.985000000000000000e+03, 2.308000000000000000e+03, 3.339000000000000000e+03, 3.328000000000000000e+03, 4.326000000000000000e+03, 2.902000000000000000e+03, 3.401000000000000000e+03, 2.818000000000000000e+03, 3.099000000000000000e+03, 4.125000000000000000e+03, 5.419000000000000000e+03, 2.209000000000000000e+03, 2.686000000000000000e+03, 2.743000000000000000e+03, 3.625000000000000000e+03, 2.701000000000000000e+03, 2.110000000000000000e+03, 2.084000000000000000e+03, 2.420000000000000000e+03, 2.044000000000000000e+03, 2.126000000000000000e+03, 2.371000000000000000e+03, 5.254000000000000000e+03, 4.351000000000000000e+03, 1.677000000000000000e+03, 2.605000000000000000e+03, 3.511000000000000000e+03, 5.923000000000000000e+03, 2.723000000000000000e+03, 2.601000000000000000e+03, 1.792000000000000000e+03, 2.921000000000000000e+03, 3.579000000000000000e+03, 4.253000000000000000e+03, 2.770000000000000000e+03, 2.690000000000000000e+03, 3.838000000000000000e+03, 3.361000000000000000e+03, 2.263000000000000000e+03, 3.825000000000000000e+03, 3.540000000000000000e+03, 4.364000000000000000e+03, 3.652000000000000000e+03, 1.362000000000000000e+03, 3.277000000000000000e+03, 2.661000000000000000e+03, 4.439000000000000000e+03, 2.976000000000000000e+03, 3.645000000000000000e+03, 3.174000000000000000e+03, 1.612000000000000000e+03, 3.802000000000000000e+03])
def obtener_w():
  return w

def calcula_v(D, r):
    # Calculamos la matriz de transiciones C
    C = calcula_matriz_C_continua(D)

    # Calculamos la matriz B de la ecuación 5
    B = calcula_B(C, r)

    # Factorizacion LU de B
    L, U = calculaLU(B)

    # Resolución del sistema Bv = w → Ly = w → Uv = y
    y = scipy.linalg.solve_triangular(L, w, lower=True)
    v = scipy.linalg.solve_triangular(U, y)

    return v, B
#%%
def calcula_norma_1(C: np.ndarray) -> float:
    # Si C tiene más de una dimensión (matriz)
    if C.ndim == 2:
        n = C.shape[0]
        m = C.shape[1]
        columnas = []
        for i in range(m):  # Iteramos sobre las columnas
            suma = 0
            for j in range(n):
                suma += abs(C[j, i])  # Accedo a los elementos 
            columnas.append(suma)
        return max(columnas)
    
    # Si C es un vector (unidimensional)
    elif C.ndim == 1:
        return np.sum(np.abs(C))          
    


