import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import funciones_TP1

#Punto 3
#a)
def calcula_L(A: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz laplaciana L = K - A del grafo.

    Parámetros:
    - A: np.ndarray. Matriz de adyacencia (simétrica).

    Retorna:
    - L: np.ndarray. Matriz laplaciana.
    """
    grados = A.sum(axis=1)
    K = np.diag(grados)
    return K - A


def calcula_R(A: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz de modularidad R = A - P del grafo,
    usando el modelo de configuración.

    Parámetros:
    - A: np.ndarray. Matriz de adyacencia (simétrica).

    Retorna:
    - R: np.ndarray. Matriz de modularidad.
    """
    grados = np.sum(A, axis=1)
    E = np.sum(A) / 2  
    P = np.outer(grados, grados) / (2 * E)
    R = A - P
    return R


def calcula_lambda(L: np.ndarray, v: np.ndarray) -> float:
    """
    Calcula el valor del corte Λ asociado al vector v y matriz L.

    Parámetros:
    - L: np.ndarray. Matriz laplaciana.
    - v: np.ndarray. Autovector (por ejemplo, el segundo de L).

    Retorna:
    - Lambda: float. Valor del corte mínimo.
    """
    return (v.T @ L @ v) / 4

def calcula_Q(R: np.ndarray, v: np.ndarray) -> float:
    """
    Calcula el valor heurístico de modularidad para un autovector v.

    Parámetros:
    - R: np.ndarray. Matriz de modularidad.
    - v: np.ndarray. Autovector (por ejemplo, el primero de R).

    Retorna:
    - Q: float. Valor heurístico de la modularidad (no escalado por 4E).
    """
    return float(v.T @ R @ v)

#b)
def norma_vector(vector: np.ndarray) -> float:
    # Calculamos la norma euclídea de un vector manualmente
    suma_cuadrados = np.sum(vector ** 2)
    return np.sqrt(suma_cuadrados)


def metpot1(matriz: np.ndarray, tolerancia: float = 1e-8, max_iteraciones: int = 1000, seed: int = None) -> tuple[np.ndarray, float, bool]:
    dimension = matriz.shape[0]

    if seed is not None:
        np.random.seed(seed)

    # Arrancamos con un vector aleatorio entre -1 y 1
    autovector = np.random.uniform(-1, 1, dimension)
    autovector = autovector / norma_vector(autovector)

    nuevo_autovector = matriz @ autovector
    nuevo_autovector = nuevo_autovector / norma_vector(nuevo_autovector)

    autovalor_aprox = autovector @ (matriz @ autovector)
    nuevo_autovalor = nuevo_autovector @ (matriz @ nuevo_autovector)

    iteraciones = 0

    while np.abs(nuevo_autovalor - autovalor_aprox) / np.abs(autovalor_aprox) > tolerancia and iteraciones < max_iteraciones:
        autovector = nuevo_autovector
        autovalor_aprox = nuevo_autovalor

        nuevo_autovector = matriz @ autovector
        nuevo_autovector = nuevo_autovector / norma_vector(nuevo_autovector)

        nuevo_autovalor = nuevo_autovector @ (matriz @ nuevo_autovector)
        iteraciones += 1

    convergio = iteraciones < max_iteraciones
    return nuevo_autovector, nuevo_autovalor, convergio


def deflaciona(matriz: np.ndarray, tolerancia: float = 1e-8, max_iteraciones: int = 1000, seed: int = None) -> np.ndarray:
    # Buscamos el autovector dominante
    autovector_1, autovalor_1, _ = metpot1(matriz, tolerancia, max_iteraciones, seed)

    # Calculamos el producto externo del autovector
    producto_externo = np.outer(autovector_1, autovector_1)

    # Aplicamos la fórmula de deflación: M - lambda * vv^T
    matriz_deflacionada = matriz - autovalor_1 * producto_externo

    return matriz_deflacionada


def metpot2(matriz: np.ndarray, autovector_1: np.ndarray, autovalor_1: float, tolerancia: float = 1e-8, max_iteraciones: int = 1000, seed: int = None) -> tuple[np.ndarray, float, bool]:
    # Construimos la matriz deflacionada
    producto_externo = np.outer(autovector_1, autovector_1)
    matriz_deflacionada = matriz - autovalor_1 * producto_externo

    # Aplicamos el método de la potencia otra vez
    return metpot1(matriz_deflacionada, tolerancia, max_iteraciones, seed)


def metpotI(matriz: np.ndarray, desplazamiento: float, tolerancia: float = 1e-8, max_iteraciones: int = 1000, seed: int = None) -> tuple[np.ndarray, float, bool]:
    dimension = matriz.shape[0]

    # Hacemos shifting con mu: M + mu * I
    matriz_shifted = matriz + desplazamiento * np.identity(dimension)

    # Invertimos usando nuestra LU
    matriz_invertida = funciones_TP1.inversaLU(matriz_shifted)

    # Aplicamos el método de la potencia a la inversa
    autovector, autovalor_inverso, convergio = metpot1(matriz_invertida, tolerancia, max_iteraciones, seed)

    # Volvemos al autovalor original
    autovalor = 1 / autovalor_inverso - desplazamiento

    return autovector, autovalor, convergio


def metpotI2(matriz: np.ndarray, desplazamiento: float, tolerancia: float = 1e-8, max_iteraciones: int = 1000, seed: int = None) -> tuple[np.ndarray, float, bool]:
    dimension = matriz.shape[0]

    # Shift y luego invertimos con LU
    matriz_shifted = matriz + desplazamiento * np.identity(dimension)
    matriz_invertida = funciones_TP1.inversaLU(matriz_shifted)

    # Deflacionamos la matriz invertida
    matriz_deflacionada = deflaciona(matriz_invertida, tolerancia, max_iteraciones, seed)

    # Aplicamos el método de la potencia a la matriz deflacionada
    autovector, autovalor_inverso, convergio = metpot1(matriz_deflacionada, tolerancia, max_iteraciones, seed)

    # Revertimos el cambio: pasamos del valor de la inversa al valor real
    autovalor = 1 / autovalor_inverso - desplazamiento

    return autovector, autovalor, convergio


#c)

def laplaciano_iterativo(A, niveles, nombres_s=None, seed: int = None):
    """
    Realiza particiones iterativas usando el segundo autovector más chico
    de la matriz Laplaciana, calculado con el método de la potencia inversa.

    Parámetros:
    - A: np.ndarray. Matriz de adyacencia simétrica.
    - niveles: int. Número de niveles de partición (2^niveles grupos).
    - nombres_s: list. Lista con los índices (o nombres) actuales de nodos.
    - seed: int. Semilla para la aleatoriedad del método de la potencia.

    Retorna:
    - Lista de listas, donde cada sublista contiene los índices de una comunidad.
    """
    if nombres_s is None:
        nombres_s = list(range(A.shape[0]))

    if niveles == 0 or A.shape[0] <= 1:
        return [nombres_s]

    L = calcula_L(A)
    v, _, _ = metpotI2(L, desplazamiento=0.1, seed=seed)

    # Particionamos por el signo del autovector
    pos = [i for i, vi in enumerate(v) if vi >= 0]
    neg = [i for i, vi in enumerate(v) if vi < 0]

    # Submatrices y nodos correspondientes
    Ap = A[np.ix_(pos, pos)]
    Am = A[np.ix_(neg, neg)]
    nombres_pos = [nombres_s[i] for i in pos]
    nombres_neg = [nombres_s[i] for i in neg]

    # Llamada recursiva
    return laplaciano_iterativo(Ap, niveles - 1, nombres_pos, seed=seed) + laplaciano_iterativo(Am, niveles - 1, nombres_neg, seed=seed)

def modularidad_iterativo(A=None, R=None, nombres_s=None, tol=1e-8, maxrep= 1000, seed: int = None):
    # Recibe una matriz A, una matriz R de modularidad, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.

    if A is None and R is None:
        print('Dame una matriz')
        return np.nan

    if R is None:
        R = calcula_R(A)

    if nombres_s is None:
        nombres_s = range(R.shape[0])

    if len(nombres_s) == 1:
        return [list(nombres_s)]

    # Submatriz de R correspondiente a los nodos actuales
    R_sub = R[np.ix_(nombres_s, nombres_s)]

    # Autovector y autovalor dominante de R_sub
    v, l, _ = metpot1(R_sub, tol, maxrep, seed)

    # Modularidad actual
    Q0 = np.sum(R_sub[v > 0, :][:, v > 0]) + np.sum(R_sub[v < 0, :][:, v < 0])

    if Q0 <= 0 or all(v > 0) or all(v < 0):
        return [list(nombres_s)]
    else:
        # Particiones por signo del autovector
        idx_pos = [i for i, vi in enumerate(v) if vi > 0]
        idx_neg = [i for i, vi in enumerate(v) if vi < 0]

        Rp = R_sub[np.ix_(idx_pos, idx_pos)]
        Rm = R_sub[np.ix_(idx_neg, idx_neg)]

        vp, _, _ = metpot1(Rp, tol, maxrep, seed)
        vm, _, _ = metpot1(Rm, tol, maxrep, seed)

        Q1 = 0
        if not all(vp > 0) and not all(vp < 0):
            Q1 += np.sum(Rp[vp > 0, :][:, vp > 0]) + np.sum(Rp[vp < 0, :][:, vp < 0])
        if not all(vm > 0) and not all(vm < 0):
            Q1 += np.sum(Rm[vm > 0, :][:, vm > 0]) + np.sum(Rm[vm < 0, :][:, vm < 0])

        if Q0 >= Q1:
            return [[ni for ni, vi in zip(nombres_s, v) if vi > 0],
                    [ni for ni, vi in zip(nombres_s, v) if vi < 0]]
        else:
            nombres_p = [ni for i, ni in enumerate(nombres_s) if v[i] > 0]
            nombres_m = [ni for i, ni in enumerate(nombres_s) if v[i] < 0]

            return modularidad_iterativo(R=R, nombres_s=nombres_p, tol=tol, maxrep=maxrep, seed=seed) + \
                   modularidad_iterativo(R=R, nombres_s=nombres_m, tol=tol, maxrep=maxrep, seed=seed)

# Punto 4

def simetrizar_matriz(A):
    A_sim = np.ceil((A + A.T) / 2).astype(int)
    np.fill_diagonal(A_sim, 0)
    return A_sim

def plot_comunidades(A, grupos, layout, barrios, titulo, ax):
    G = nx.from_numpy_array(A)
    colores = plt.cm.tab20(np.linspace(0, 1, len(grupos)))

    barrios = barrios.to_crs("EPSG:22184")
    barrios.boundary.plot(ax=ax, color='black')

    for i, grupo in enumerate(grupos):
        nx.draw_networkx_nodes(G, layout, nodelist=grupo, node_color=[colores[i]], label=f'Grupo {i+1}', ax=ax, node_size=150)

    nx.draw_networkx_edges(G, layout, alpha=0.3, ax=ax)

    ax.set_title(titulo)
    ax.axis('off')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

def correr_metodos(D, m, layout, barrios):
    A = funciones_TP1.construye_adyacencia(D, m)
    A_sim = simetrizar_matriz(A)

    # MODULARIDAD
    grupos_mod = modularidad_iterativo(A=A_sim)

    # LAPLACIANO
    for n in range(1, 5):
        grupos_lap = laplaciano_iterativo(A_sim, niveles=n)
        if abs(len(grupos_lap) - len(grupos_mod)) <= 2:
            
            # Subplot con ambos
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 12))
            plot_comunidades(A_sim, grupos_mod, layout, barrios, f"Modularidad (m={m}, {len(grupos_mod)} grupos)", ax1)
            plot_comunidades(A_sim, grupos_lap, layout, barrios, f"Laplaciano (m={m}, niveles={n}, {len(grupos_lap)} grupos)", ax2)
            plt.tight_layout()
            plt.show()
            break

#Analisis de estabilidad
def analizar_estabilidad_modularidad(D, valores_m, lista_seeds):
    resultados = {}

    for m in valores_m:
        A = funciones_TP1.construye_adyacencia(D, m)
        A_sim = simetrizar_matriz(A)

        resultados[m] = []

        for seed in lista_seeds:
            grupos = modularidad_iterativo(A=A_sim, seed=seed)
            cantidad_grupos = len(grupos)
            resultados[m].append(cantidad_grupos)

    # Graficar resultados
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in valores_m:
        ax.plot(lista_seeds, resultados[m], marker='o', label=f"m = {m}")

    ax.set_title("Estabilidad de la cantidad de grupos según la seed (Modularidad)")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Cantidad de grupos")
    ax.legend(title="Valores de m")
    plt.xticks(lista_seeds)
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    
def analizar_estabilidad_laplaciano(D, valores_m, lista_seeds, niveles):
    for m, nivel in zip(valores_m, niveles):
        A = funciones_TP1.construye_adyacencia(D, m)
        A_sim = simetrizar_matriz(A)
        
        tamaños_por_seed = []

        for seed in lista_seeds:
            grupos = laplaciano_iterativo(A_sim, niveles=nivel, seed=seed)
            tamaños = sorted([len(grupo) for grupo in grupos], reverse=True)
            tamaños_por_seed.append(tamaños)

        # Convertimos a matriz, completando con NaN si hay grupos de distinto tamaño
        max_grupos = max(len(t) for t in tamaños_por_seed)
        matriz_tamaños = np.full((len(lista_seeds), max_grupos), np.nan)

        for i, t in enumerate(tamaños_por_seed):
            matriz_tamaños[i, :len(t)] = t

        # Graficamos
        plt.figure(figsize=(10, 6))
        for j in range(max_grupos):
            plt.plot(lista_seeds, matriz_tamaños[:, j], marker='o', label=f'Grupo {j+1}')
        
        plt.title(f"Estabilidad del Laplaciano - Tamaño de comunidades (m = {m})")
        plt.xlabel("Seed")
        plt.ylabel("Cantidad de museos por comunidad")
        plt.xticks(lista_seeds)  
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
