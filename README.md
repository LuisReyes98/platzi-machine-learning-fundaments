# Curso de Fundamentos Prácticos de Machine Learning

## Otros recursos y lecturas

https://smartai-blog.com/what-is-google-colab/

https://smartai-blog.com/machine-learning-heavy-industries/

https://smartai-blog.com/

## Los fundamentos de machine learning que aprenderás

Capacidad de un algoritmo de adquirir conocimiento a partir de los datos recolectados para mejorar, describir y predecir resultados

### Estrategias de Aprendizaje

- **Aprendizaje Supervisado:** Permite al algoritmo aprender a partir de datos previamente etiquetados.
Aprendizaje no Supervisado: El algoritmo aprende de datos sin etiquetas, es decir encuentra similitudes y relaciones, agrupando y clasificando los datos.
- **Aprendizaje Profundo (Deep Learning):** Está basado en redes Neuronales

**Importancia del ML**
Permite a los algoritmos aprender a partir de datos históricos recolectados por las empresas permitiendo así tomar mejores decisiones.

Pasos a Seguir para Desarrollar un modelo en ML

- **Definición del Problema:** Es necesario definir previamente el problema que va a resolver nuestro algoritmo
Construcción de un modelo y Evaluación: Una vez definido el problema procedemos a tratar los datos y a entrenar nuestro modelo que debe tener una evaluación cercana al 100%
- **Deploy y mejoras:** El algoritmo es llevado a producción (aplicación o entorno para el que fue creado), en este entorno podemos realizar las mejoras pertinentes de acuerdo al comportamiento con los usuario e incluso aprovechando los datos generados en esta interacción

### Ventajas

- Definir problema: poder identificar la verdadera causa de los problemas
- Construccion del modelo y evaluacion
- Deploy y mejoras

### Aplicaciones

Empresas como Google, Netflix etc
Usan machine learning ya que les permite obtener resutlados en base a datos historicos o un enorme conjunto de datos

Sistemas de recomendacion

Identificadores de rostros e imagenes

Busqueda de la ruta mas optima

Estas aplicaciones se logran principalmente con aprendizaje automatico y Deeplearning

## Introducción a Numpy

Biblioteca de python comunmente usada en la ciencia de datos

- Sencilla de usuar
- Adecuada para el manejo de arreglos
- Rapida

## Introducción y manipulación de datos con Pandas

Biblioteca de código abierto que proporciona estructuras de datos y herramientas de análisis de datos de alto rendimiento y fáciles de usar.

- Manejo de archivos
- Series (1D)
- Dataframes (2D)
- Panels(3D)

## Introducción a ScikitLearn

Biblioteca de código abierto para el aprendizaje automático, incluye algoritmos como árboles de decisión, regresión, máquina de soporte vectorial, entre otros.

- Variedad de módulos
- Versatilidad
- Facilidad de uso

```python
from sklearn
import [modelo]
```

paquetes utiles

```python
# para hacer preprocesamiento de los datos
from sklearn import preprocessing

# para establecer nuestros conjuntos de entrenamiento y evaluacion.
from sklearn import train_test_split

# para usar las metricas necesarias para analizar la ejecución de nuestros modelos
from sklearn import metrics
```

## Comandos básicos de las librerías usadas en el curso (Numpy, Pandas y ScikitLearn)

### Numpy

Biblioteca de Python comúnmente usada en la ciencias de datos y aprendizaje automático (Machine Learning). Proporciona una estructura de datos de matriz que tiene diversos beneficios sobre las listas regulares.

Importar la biblioteca:
import numpy as np

Crear arreglo unidimensional:
my_array = np.array([1, 2, 3, 4, 5])
Resultado: array([1, 2, 3, 4, 5])

Crear arreglo bidimensional:
np.array( [[‘x’, ‘y’, ‘z’], [‘a’, ‘c’, ‘e’]])
Resultado:
[[‘x’ ‘y’ ‘z’]
[‘a’ ‘c’ ‘e’]]

Mostrar el número de elementos del arreglo:
len(my_array)

Sumar todos los elementos de un arreglo unidimensional:
np.sum(my_array)

Obtener el número máximo de los elementos del arreglo unidimensional
np.max(my_array)

Crear un arreglo de una dimensión con el número 0:
np.zeros(5)
Resultado: array([0., 0., 0., 0., 0.])

Crear un arreglo de una dimensión con el número 1:
np.ones(5)
Resultado: array([1., 1., 1., 1., 1.])

Comando de Python para conocer el tipo del dato:
type(variable)

Ordenar un arreglo:
np.order(x)

Ordenar un arreglo por su llave:
np.sort(arreglo, order = ‘llave’)

Crear un arreglo de 0 a N elementos:
np.arange(n)
Ej.
np.arange(25)
Resultado:
array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
17, 18, 19, 20, 21, 22, 23, 24])

Crear un arreglo de N a M elementos:
np.arange(n, m)
Ej.
np.arange(5, 30)
Resultado:
array([ 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
22, 23, 24, 25, 26, 27, 28, 29])

Crear un arreglo de N a M elementos con un espacio de cada X valores:
np.arange(n, m, x)
Ej.
np.arange(5, 50, 5)
Resultado:
array([ 5, 10, 15, 20, 25, 30, 35, 40, 45])

Crear arreglo de NxM:
np.full( (n, m), x )
Ej.
np.full( (3, 5), 10)
Resultado:
array([
[10, 10, 10, 10, 10],
[10, 10, 10, 10, 10],
[10, 10, 10, 10, 10]
])

Número de elementos del arreglo:
len(my_array)

### Pandas

Pandas es una herramienta de manipulación de datos de alto nivel, es construido con la biblioteca de Numpy. Su estructura de datos más importante y clave en la manipulación de la información es DataFrame, el cuál nos va a permitir almacenar y manejar datos tabulados observaciones (filas) y variables (columnas).

Importar la biblioteca:
import pandas as pd

Generar una serie con Pandas:
pd.Series([5, 10, 15, 20, 25])
Resultado:
0 5
1 10
2 15
3 20
4 25

Crear un Dataframe:
lst = [‘Hola’, ‘mundo’, ‘robótico’]
df = pd.DataFrame(lst)
Resultado:
0
0 Hola
1 mundo
2 robótico

Crear un Dataframe con llave y dato:
data = {‘Nombre’:[‘Juan’, ‘Ana’, ‘Toño’, ‘Arturo’],
‘Edad’:[25, 18, 23, 17],
‘Pais’: [‘MX’, ‘CO’, ‘BR’, ‘MX’] }
df = pd.DataFrame(data)
Resultado:

Resultado Data Frame.png
Leer archivo CSV:
pd.read_csv(“archivo.csv”)

Mostrar cabecera:
data.head(n)

Mostrar columna del archivo leído:
data.columna

Mostrar los últimos elementos:
data.tail()

Mostrar tamaño del archivo leído:
data.shape

Mostrar columnas:
data.columns

Describe una columna:
data[‘columna’].describe()

Ordenar datos del archivo leído:
data.sort_index(axis = 0, ascending = False)

### Scikit Learn

Scikit Learn es una biblioteca de Python que está conformada por algoritmos de clasificación, regresión, reducción de la dimensionalidad y clustering. Es una biblioteca clave en la aplicación de algoritmos de Machine Learning, tiene los métodos básicos para llamar un algoritmo, dividir los datos en entrenamiento y prueba, entrenarlo, predecir y ponerlo a prueba.

Importar biblioteca:
from sklearn import [modulo]

División del conjunto de datos para entrenamiento y pruebas:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

Entrenar modelo:
[modelo].fit(X_train, y_train)

Predicción del modelo:
Y_pred = [modelo].predict(X_test)

Matriz de confusión:
metrics.confusion_matrix(y_test, y_pred)

Calcular la exactitud:
metrics.accuracy_score(y_test, y_pred)

### Cheatsheets

[Numpy](https://www.datacamp.com/community/blog/python-numpy-cheat-sheet)
[Pandas](https://www.datacamp.com/community/blog/python-pandas-cheat-sheet)
[Scikit](https://www.datacamp.com/community/blog/scikit-learn-cheat-sheet)

[Scikit documentation](https://scikit-learn.org/stable/)

## ¿Qué es la predicción de datos?
