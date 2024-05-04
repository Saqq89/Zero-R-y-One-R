import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from tabulate import tabulate
# Uso de clases y programacion estructurada
# Con la biblioteca pandas podemos realizar la manipulación y el análisis de datos.
# Con la biblioteca de sklearn nos proporciona herramientas para el analisis y modelado de datos para el aprendizaje automatico.
# Usamos pandas para leer el archivo cervezas.txt, limpiarlo y transformarlo en un DataSet.

ruta_archivo = 'C:/Users/Cristian Lopez/Desktop/Algoritmosr/pythonProject/cervezas.txt'

#D
# Aqui leeremos el archivo de datos y estara por '|' y eliminamos las  filas y columnas vacías.

datos = pd.read_csv(ruta_archivo, delimiter='|', skipinitialspace=True).dropna(axis=1, how='all').dropna(axis=0, how='all')
datos.columns = datos.columns.str.strip()  # Limpiar nombres de columnas

# Aqui mostraremos los nombres de las columnas.

print("\nNombres de columnas:")
print(datos.columns)

# Convertiremos las columnas categóricas en valores numéricos y utilizaremos el tipo de datos 'category'.
for col in datos.columns:
    datos[col] = datos[col].astype('category').cat.codes

# Mostraremos las primeras filas de los datos para revisar la transformación del dataset.

print("\nPrimeras filas:")
print(tabulate(datos.head(), headers='keys', tablefmt='pretty'))

# Definiremos el nombre de la columna que contiene las etiquetas de clase.
nombre_clase = 'Prefiere'

if nombre_clase not in datos.columns:
    raise KeyError(f"'{nombre_clase}' not found in dataset columns")

X = datos.drop(nombre_clase, axis=1)
y = datos[nombre_clase]
# Solicitaremos al usuario el número de iteraciones para evaluar los modelos de entrenamiento y prueba

iteraciones = int(input("\nIntroduzca el número de iteraciones para la evaluación: "))

# La siguiente función es la que se en encargara de determinar la clase mas frecuente en el conjunto de entrenamiento.
def modelo_zero_r(y_entrenamiento):
    return Counter(y_entrenamiento).most_common(1)[0][0]

#C
# Aqui Evaluaremos el modelo ZeroR utilizando la clase mas frecuente para poder predecir y comparar las predicciones
# con las etiquetas reales del dataset para calcular la precisión y los errores.
def evaluar_zero_r(y_entrenamiento, y_prueba):
    prediction = modelo_zero_r(y_entrenamiento)
    exactitud = sum(prediction == actual for actual in y_prueba) / len(y_prueba)
    errores = len(y_prueba) - sum(prediction == actual for actual in y_prueba)
    return exactitud, errores

# Lista para almacenar la precision del modelo Zero en cada iteración
precisiones_zero_r = []
#aqui imprimimos las iteraciones solicitadas por el usuario para los datos de entrenamiento y datos de prueba
for _ in range(iteraciones):
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.3, random_state=None, shuffle=True)

    print("\nDatos de entrenamiento (ZeroR):")
    print(tabulate(pd.concat([X_entrenamiento, y_entrenamiento], axis=1).head(), headers='keys', tablefmt='pretty'))
    print("\nDatos de prueba (ZeroR):")
    print(tabulate(pd.concat([X_prueba, y_prueba], axis=1).head(), headers='keys', tablefmt='pretty'))

    exactitud_zero_r, errores_zero_r = evaluar_zero_r(y_entrenamiento, y_prueba)
    precisiones_zero_r.append(exactitud_zero_r)

    print(f'Zero-R Exactitud: {exactitud_zero_r * 100:.2f}% | Errores: {errores_zero_r}')

precision_promedio_zero_r = sum(precisiones_zero_r) / iteraciones
print(f'\nPrecisión promedio ZeroR después de {iteraciones} iteraciones: {precision_promedio_zero_r * 100:.2f}%')

#S
# Nos encargaremos de encontrar el mejor atributo para clasificar en funcion del menor numero de errores y construiremos las
# reglas de clasificacion basadas en el atributo conforme al Modelo de One R.
def modelo_one_r(X_entrenamiento, y_entrenamiento):
    reglas = {}
    mejor_atributo, menor_errores = None, float('inf')

    for columna in X_entrenamiento.columns:
        conteos = {}
        for valor, clase in zip(X_entrenamiento[columna], y_entrenamiento):
            if valor not in conteos:
                conteos[valor] = Counter()
            conteos[valor][clase] += 1
        reglas_temp = {v: c.most_common(1)[0][0] for v, c in conteos.items()}
        errores = sum(X_entrenamiento[columna].map(reglas_temp) != y_entrenamiento)

        if errores < menor_errores:
            mejor_atributo, reglas = columna, reglas_temp
            menor_errores = errores

    return mejor_atributo, reglas
# En esta funcion Evaluaremos el modelo OneR utilizando las reglas construidas a partir del atributo mejor encontrado
# y comparemos las predicciones con las etiquetas para calcular la precisión y los errores.
def evaluar_one_r(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba):
    mejor_atributo, regla = modelo_one_r(X_entrenamiento, y_entrenamiento)
    # Asignar un valor predeterminado -1 a las predicciones que no estén en las reglas
    predicciones = X_prueba[mejor_atributo].map(regla).fillna(y_entrenamiento.mode()[0]).astype(int)
    exactitud = sum(y_prueba == predicciones) / len(y_prueba)
    errores = len(y_prueba) - sum(y_prueba == predicciones)
    return exactitud, errores


#D
# Lista para almacenar la precisión del modelo One R en cada iteración
precisiones_one_r = []


#Este fragmento evalúa el modelo OneR en múltiples iteraciones
#cada iteración see divide aleatoriamente el conjunto de datos en entrenamiento 70% y prueba 30% y luego se entrena y evalúa el modelo
# almacenando la precisión obtenida
# Finalmente, calcula y muestra la precisión promedio del modelo.
for _ in range(iteraciones):
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.3, random_state=None, shuffle=True)

    print("\nDatos de entrenamiento (OneR):")
    print(tabulate(pd.concat([X_entrenamiento, y_entrenamiento], axis=1).head(), headers='keys', tablefmt='pretty'))
    print("\nDatos de prueba (OneR):")
    print(tabulate(pd.concat([X_prueba, y_prueba], axis=1).head(), headers='keys', tablefmt='pretty'))

    exactitud_one_r, errores_one_r = evaluar_one_r(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba)
    precisiones_one_r.append(exactitud_one_r)

    print(f'One R Exacttitud: {exactitud_one_r * 100:.2f}% | Errores : {errores_one_r}')
#Comilacion del programa
#fin del codigo
precision_promedio_one_r = sum(precisiones_one_r) / iteraciones
print(f'\nPrecisión promedio OneR después de {iteraciones} iteraciones: {precision_promedio_one_r * 100:.2f}%')
