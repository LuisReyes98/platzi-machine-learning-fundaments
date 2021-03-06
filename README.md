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

- Regresión lineal
- Regresión logística
- Regresión multiple

**¿Qué es la predicción de datos?**
Algoritmos que se definen como "clasificadores" que indentifican a qué conjunto de categorías pertenecen los datos.

la informacion que se provee a los modelos debe estar sumamente cuidada

lo primero a pensar es cual es mi problema a resolver y lo siguiente es obtener el conjunto de datos

Podemos entrenar con datos historicos que entreguen resultados para ser aplicados al negocio

los algoritmos supervisados y no supervisados estan alimentados por un conjunto de datos con diferentes atributos, los cuales son analizados por el modelo, en busqueda de una optimizacion.

## Sobreajuste y subajuste en los datos

Nuestro modelo lo "obligamos" a ajustarse a los datos de entrada y salida

### Sobreajuste (overfiting)

Todos los datos deben estar divididos de forma variada, ya que, si los datos estan demasiado agrupados a un conjunto, al algoritmo ver un nuevo dato que no se parezca al conjunto a pesar de pertecer a el lo identificara erroneamente

Ejemplo: se quiere identificar gatos en una foto y en todas la fotos los gatos son blancos, por lo cual si al algoritmo se le muestra un gato negro no podra identificarlo como un gato.

por lo cual los datos deben ser variados, limpiados y clasificados con anterioridad para evitar el sobreajuste a los datos

Es cuando intentamos obligar a nuestro algoritmo a que se ajuste demasiado a todos los datos posibles en lugar de aprender. Es muy importante proveer con información abundante a nuestro modelo pero también esta debe ser lo suficientemente variada para que nuestro algoritmo pueda generalizar lo aprendido.

### Subajuste (underfiting)

El modelo fallará en el reconocimiento por fallará en el reconocimiento por falta de muestras suficientes. No generaliza el conocimiento.

el modelo falla por tener poca informacion y no logra aprender los rasgos caracteristicos de los datos.

Es cuando le suministramo a nuestro modelo un conjunto de datos es muy pequeño, en este caso nuestro modelo no sera capas de aprender lo suficiente ya que tiene muy poca infomación. La recomendación cuando se tienen muy pocos datos es usar el 70% de los datos para que el algoritmo aprenda y usar el resto para entrenamiento.

### Otras notas

hay varios metodos de cross-validation, y no necesariamente, debe de ser Train-Test. puede ser tambien: Train-Test-Validation, k-folds, LOOCV, etc.

[Las Redes Neuronales... ¿Aprenden o Memorizan? - Overfitting y Underfitting - Parte 1](https://www.youtube.com/watch?v=7-6X3DTt3R8)

## Regresión lineal simple y regresión lineal múltiple

Son algoritmos de tipo supervisado

**Regresión lineal simple**
Algoritmo de aprendizaje supervisado que nos indica la tendencia de un conjunto de datos continuos, modelando la relación entre una variable dependiente Y y una variable explicativa llamada X.

Esto se representa en una gráfica bidimensional X, Y

cada punto en la gráfica es un elemento de la muestra de datos

Se busca obtener la ecuación de la pendiente optimizada para los datos

$$
Yi = b + mXi
$$

Con la regresión lineal simple se puede buscar la relación entre una característica y otra, Ejemplo: salario de las personas a lo largo del tiempo

pero en el caso de tener mas de una caracteristica es decir un ejemplo como: Salario de las personas en base a años de experiencia, edad, estudios, años en la compañia, cantidad de empleos anteriores etc...

para poder trabajar con multiples características se usa la

**Regresión lineal multiple**
Si nuestro problema tiene más de dos variables se le considera lineal multiple

Al tener multiples dimensiones X, Y, Z, etc. Son representados gráficamente por un hiperplano.

### Otras notas

algo que vale la pena mencionar es que cuando tienes muchas variables X1, X2, …Xn puede suceder que la información mas trascendente se encuentre contenida en tan solo 3 , 5 o unas pocas variables asi que se intenta realizar una “reducción de la dimensión” de manera que solo trabajemos con aquellas variables que representan mayor importancia. métodos para esto pueden ver el más conocido como PCA (principal component analysis)

## Regresión lineal simple con Scikit-Learn: división de los datos

para subir el CSV de salarios a su entorno de collab, en una celda aparte (o bien en la que por convención, estamos dejando para el tema de los módulos) escriban el siguente código

```python
from google.colab import files

import io
from google.colab import files
uploaded = files.upload()
```

Esto les abrirá una interfaz para que puedan subir archivos, eso si, tienen que correr la celda dos veces, luego elijen el CSV y lo suben, luego la linea de código se modifica un poco, quedando de la siguiente forma para meterla en un dataframe

```python
dataset = pd.read_csv(io.BytesIO(uploaded['salarios.csv']))
```

## Regresión lineal simple con Scikit-Learn: creación del modelo

## Regresión logística con Scikit-Learn: definición y división de datos

Si nuestro dato de salida tiene un valor cualitativo utilizamos y aplicamos la regresión logística

Lo comun es que el ingeniero de machine learning recibe la información limpia y ordenada

## Regresión logística con Scikit-Learn: evaluación del modelo

Formas de validar un modelo de regresión logística

Matriz de confusión

|              |           |                           | Prediccion                |
| ------------ | --------- | ------------------------- | ------------------------- |
|              |           | **Positivos**             | **Negativos**             |
| Obeservacion | Positivos | Verdaderos Positivos (VP) | Falsos negativos (FN)     |
| Obeservacion | Negativos | Falsos Positivos (FP)     | Verdaderos Negativos (VN) |

## Matriz de confusión

Los modelos de clasificación son capaces de predecir cuál es la etiqueta correspondiente a cada ejemplo o instancia basado en aquello que ha aprendido del conjunto de datos de entrenamiento. Estos modelos necesitan ser evaluados de alguna manera y posteriormente comparar los resultados obtenidos con aquellos que fueron entrenados.

Una manera de hacerlo es mediante la matriz de confusión la cual nos permite evaluar el desempeño de un algoritmo de clasificación a partir del conteo de los aciertos y errores en cada una de las clases del algoritmo.

Como su nombre lo dice tenemos una matriz que nos ayuda a evaluar la predicción mediante positivos y negativos como se muestra en la figura.

Los verdaderos positivos (VP) son aquellos que fueron clasificados correctamente como positivos como el modelo.
Los verdaderos negativos (VN) corresponden a la cantidad de negativos que fueron clasificados correctamente como negativos por el modelo.
Los falsos negativos (FN) es la cantidad de positivos que fueron clasificados incorrectamente como negativos.
Los falsos positivos (FP) indican la cantidad de negativos que fueron clasificados incorrectamente como positivos.

Para que lo anterior quede más claro consideremos el siguiente ejemplo.

Un médico tiene cuatro pacientes y a cada uno se les solicitó un examen de sangre y por error el laboratorio realizó también un estudio de embarazo, cuando los pacientes llegan el médico les da los resultado.

A la primera paciente le da la noticia que está embarazada y ella ya lo sabía dado que tiene 3 meses de embarazo, es decir, un verdadero positivo.

El siguiente paciente llega y le dice que no está embarazada y es una clasificación evidente dado que es hombre (Verdadero negativo).

El tercer paciente llega y los resultados le indican que no está embarazada sin embargo tiene cuatro meses de embarazo, es decir, que la ha clasificado como falso negativo.

Y por último el cuarto paciente sus resultados han indicado que está embarazado sin embargo es hombre por lo cual es imposible, dando como resultado un falso positivo.

Lo anterior es un proceso que se realiza por cada instancia a clasificar y nos permite calcular la exactitud y su tasa de error con las siguientes fórmulas.Los verdaderos positivos (VP) son aquellos que fueron clasificados correctamente como positivos como el modelo.
Los verdaderos negativos (VN) corresponden a la cantidad de negativos que fueron clasificados correctamente como negativos por el modelo.
Los falsos negativos (FN) es la cantidad de positivos que fueron clasificados incorrectamente como negativos.
Los falsos positivos (FP) indican la cantidad de negativos que fueron clasificados incorrectamente como positivos.

![alt](./Prediccion.jpg)

Para que lo anterior quede más claro consideremos el siguiente ejemplo.

Un médico tiene cuatro pacientes y a cada uno se les solicitó un examen de sangre y por error el laboratorio realizó también un estudio de embarazo, cuando los pacientes llegan el médico les da los resultado.

A la primera paciente le da la noticia que está embarazada y ella ya lo sabía dado que tiene 3 meses de embarazo, es decir, un verdadero positivo.

El siguiente paciente llega y le dice que no está embarazada y es una clasificación evidente dado que es hombre (Verdadero negativo).

El tercer paciente llega y los resultados le indican que no está embarazada sin embargo tiene cuatro meses de embarazo, es decir, que la ha clasificado como falso negativo.

Y por último el cuarto paciente sus resultados han indicado que está embarazado sin embargo es hombre por lo cual es imposible, dando como resultado un falso positivo.

Lo anterior es un proceso que se realiza por cada instancia a clasificar y nos permite calcular la exactitud y su tasa de error con las siguientes fórmulas.

![exactitud](./Exactitud.jpg)

![Tase de error](Tasa_de_error.webp)

## ¿Qué es un árbol de decisión y cómo se divide?

Es una forma gráfica y analítica que presenta sucesos y sus posibles consecuencias

Es una manera de dividir todos los sucesos y por último obtener una decisión

Es un grafo con forma de arbol que en cada una de sus ramas tiene una decisión

Ejemplo:
Una tienda en base a los datos que recolecta del comportamiento de un cliente puede descifrar que producto tiene la intención de comprar

### Ventajas

- Claridad en los datos
Se sabe que hace en base a los caminos

- Tolerante al ruido y valores faltantes

- Las reglas extraídas permiten hacer predicciones

### Desventajas

- Criterio de division es deficiente
- Sobreajuste
- Ramas poco significativas
Puedes llegar a tener ramas que su peso en la decisión en muy bajo

### Como se divide un arbol de decisión

En base a los ejemplos y las características que llevaron a una decision se crea la relación entre los datos que lleva a una decision

### Optimización de nuestro modelo

- Evitar sobreajuste
Ocasionara que el modelo no se adapte a la vida real
- Selección de atributos
Identificar cuáles son los atributos relevantes a la hora de tomar la decisión
- Campos nulos
El tener campos nulos hara que se genere mucho ruido en los datos ocasionando deficiencia en el modelo

## Comprendiendo nuestro data set para la creación de un árbol de decisión

[Dataset de sobrevivientes del Titanic](https://www.kaggle.com/c/titanic/data)

## Creando un clasificador con Scikit-Learn

Notas Alumnos

**johanR**
algunas cosas que comentar que pueden ser utiles

- cuando se hace el drop intentando eliminar algunas variables que no serán de interés, axis = 1 indica que estas variables son "columnas ", axis = 0 , indicaría que son filas.
- las dummy variables que se mencionan ligeramente convierten las variables categóricas en indicadoras como 0,1,2,…etc
- cuando se completaron los valores faltantes en las variables edad y la clase del pasajero (embarked), faltó mencionar un comando muy util para saber en que variables se tienen valores faltantes. Se puede usar train_df.isnull.any().

**David Rueda**
Cuando se llenan los espacios con vacíos (fillna), para el caso de datos numéricos se utiliza la mediana porque es una de las medidas de tendencia central que menos se afecta por los datos atípicos.

Para el caso de los datos categóricos relacionados con el embarque, se utiliza la letra S porque representa el embarque en la ciudad de South Hampton, en donde más personas se unieron al viaje.

Estos datos se obtienen de un análisis previo a los datos trabajados.

## Entrenamiento del modelo de clasificación

Links de informacion util de porque usar la score para medir la eficacia de un clasificador es una mala decisión.

- [Classification vs. Prediction](https://www.fharrell.com/post/classification/)

- [Damage Caused by Classification Accuracy and Other Discontinuous Improper Accuracy Scoring Rules](https://www.fharrell.com/post/class-damage/)

- [Why is accuracy not the best measure for assessing classification models?](https://stats.stackexchange.com/questions/312780/why-is-accuracy-not-the-best-measure-for-assessing-classification-models?newreg=0e20e6e4acda4d64b19b600a61b12665)

- [Example when using accuracy as an outcome measure will lead to a wrong conclusion](https://stats.stackexchange.com/questions/368949/example-when-using-accuracy-as-an-outcome-measure-will-lead-to-a-wrong-conclusio)

- [Modelo de prediccion de riesgo](https://medium.com/@mohcine.madkour/use-scikit-learn-pipelines-to-select-a-risk-prediction-model-for-type-2-diabetes-14db7a47efa5)

## Visualización del árbol de decisión

## ¿Qué es K-Means?

Es un algoritmo de tipo no supervisado

Crea K grupos a partir de un conjunto de observaciones, los elementos deben tener similitudes

Clasifica la informacion en grupos en base a las similitudes

1. Selecciona un valor para K (centroides)
2. Asignamos cada uno de los elementos restantes al centro mas cercano
3. Asignamos cada punto a su centroide más cercano
4. Repetimos paso 2 y 3 hasta que los centros no se modifiquen

### Método del codo

- Calcula el agrupamiento para diferentes de K
- El error al cuadrado para cada punto es el cuadrado de la distancia del punto desde su centro

### otras notas

- Algoritmo no supervisado.
- Crea K grupos a partir de observaciones de un set de datos.
- Trata información que no tiene etiquetas asignadas.
- Agrupa información basada en sus características.
- K = centroides
- Aproximación a K: método del codo
- Aplicaciones:

Segmentación por comportamiento:

- por historial de compras
- actividad en una aplicación móvil, web
- Definir personas basadas en sus intereses.
- Crear perfiles basado en el monitoreo de actividad.
- Ordenando medidas de sensores:
- Detecta tipos de actividades en sensores de movimiento.
- Grupos de imágenes.
- Separar audio.
- Identificar grupos en monitoreo de salud.

[**Ejemplo:**](https://github.com/sergio2526/Modelo-Accidentes/blob/master/Codigo/m%C3%A9todo_del_codo.ipynb)

## Cargando el data set de Iris

- Virginica, Versicolor y Setosa
- 50 muestras de cada especie
- Largo y ancho del sépalo y pétalo

## Construcción y evaluación del modelo con K-Means

## Graficación del modelo

## Introducción al aprendizaje profundo

Deep learning

Permite trabajar con redes neuronales

Los niveles son:

- Inteligencia Artificial
  - Aprendizaje automático
    - Aprendizaje profundo

### ¿Qué es el aprendizaje profundo?

Subcategoria del ML que crea diferentes niveles de abstracción que representa los datos

Se suele utilizar para reconocimiento de imagenes en las cuales puede reconocer formas, colores, objetos e identificar que esta viendo

### ¿Qué es y cómo funciona una red neuronal?

Se compone por multiples capas

- Capa de entrada
- Capas ocultas
- Capa de salida

### ¿Qué es ReLU?

Funcion de activación permite el paso de todos los valores positivos sin cambiarlos pero asigna todos los valores negativos a 0

para poder aprender hace falta una función de activación

a traves de la activación de la funcion podemos concluir el aprendizaje, a traves de las diferentes conexiones entre las neuronas

### Tensor Flow

Biblioteca de codigo abierto desarrollado por Google capaz de construir y entrenar redes neuronales

### Otras notas

- Herramientas: pyTorch y TensorFlow
- Subcategoría de ML que crea diferentes niveles, de abstracción que representa los datos.
- Se usan tensores para representar estructuras de datos más complejas.
- Neuronas: Capa de entrada, capas ocultas y capa de salida.
- Para poder aprender se necesita una función de activación: ReLU: Permite el paso de todos los valores positivos sin cambiarlos, pero asigna todos los valores negativos a 0.
- TensorFlow: Biblioteca de código abierto desarrollado por google, capaz de construir y entrenar redes neuronales.

## Conceptos básicos de Tensor Flow

Tensor Flow es una biblioteca de software de código abierto que permite construir y entrenar redes neuronales, permite detectar y descifrar patrones en los datos. Es un desarrollo de Google y que debido a su flexibilidad y extensa comunidad de programadores ha crecido rápidamente y se ha posicionado como la herramienta líder en el estudio del aprendizaje profundo o también conocido como Deep Learning.

Tensor Flow puede ser usado para ayudar al diagnóstico médico, detectar objetos, procesar imágenes, detección de emociones en el rostro, entre otras aplicaciones. En este curso usamos Tensor Flow para crear nuestra primera red neuronal y diseñar un clasificador de imágenes a partir de un conjunto de datos.

Importar la biblioteca:

```python
import tensorflow as tf
```

Importar el modelo:

```python
from tensorflow import keras
```

Cargar conjunto de datos de Tensor Flow:

```python
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```

Crear modelo secuencial:

```python
model = keras.Sequential([keras.layers.Flatten(input_shape = (28, 28)), keras.layers.Dense(128, activation = tf.nn.relu), keras.layers.Dense(10, activation = tf.nn.softmax)])
```

Compilación del modelo:

```python
model.compile(optimizer = tf.optimizers.Adam(), loss = ‘sparse_categorical_crossentropy’, metrics = [‘accuracy’])
```

Entrenamiento:

```python
model.fit(train_images, train_labels, epochs = 5)
```

Evaluación del modelo:

```python
test_loss, test_acc = model.evaluate( test_images, test_labels )
```

Predicción del modelo:
model.predict(test_images)

## Red neuronal convolucional

Modelan de forma consecutiva pequeñas piezas de información, al final combinan información en las capas más profundas de la red.

Existen diferentes tipos de redes neuronales para diversas tareas, a la hora de manejar imágenes o audio se usan redes neuronales convolucionales

Las redes neuronales convolucionales se caracterizan porque tienen una sección que va dividiendo la información de una manera consecutiva

De esta forma se puede enfocar en aprender únicamente ciertas secciones

logrando así encontrar diferentes patrones que existen en los datos

en el caso de las imágenes se divide en multiples segmentos y cada segmento se vuelve una capa que es analizada para luego procesarla con las funciones de activación

la capa de salida se calcula en base a los pesos de cada una de las características para así poder asignarles una etiqueta a los datos.

### Otras notas

básicamente cada capa capta cierto atributos de la imagen, generalmente las primeras captan “bordes” , pero entre mas profunda la red capta caracteristicas mas complejas. En si aplicar una convolucion resulta en aplicar una matriz de numeros a la imagende input vista igualmente como una matriz de números. donde dichos numeros hacen referencia a los pixeles de la misma que van de 0 a 256. El tema es valga la redundancia muy profundo pero espero el comentario sirva a algunos.

[Video explicativo de Dot CSV](https://www.youtube.com/watch?v=V8j1oENVz00)

[CNN para deteccion de rostros y ojos en tiempo real con OpenCV](https://platzi.com/tutoriales/1866-keras-neural-networks/6365-cnn-para-deteccion-de-rostros-y-ojos-en-tiempo-real-con-opencv/)

## Conociendo el set de datos para la creación de la red neuronal

[Clasificacion basica con Fashion MNIST](https://www.tensorflow.org/tutorials/keras/classification)

## Crea y entrena tu primera red neuronal convolucional con Tensor Flow

## Evaluación de la red convolucional

[Computer vision Tensor flow](https://www.youtube.com/watch?v=JivjbFTlyNU)

[Tensorflow Zero to Hero](https://www.youtube.com/playlist?list=PLQY2H8rRoyvz3rEFpW2I3gPSru5xm8Bf7)

## PlatziDoro Cápsula 3

K-Means

es un algoritmo de tipo no supervisado que sirve para agrupar información con características similares.

## Recomendaciones para analizar correctamente tu problema

A lo largo del curso hemos visto modelos supervisados, no supervisados y profundos

para saber que modelo funciona más para un problema hace falta analizar qué datos se tienen y que se quiere lograr

### Regresión

Cuando se busca entender tendencias en el timepo para pronosticar un valor futuro

se puede usar para saber cuándo el precio de un producto baja

para hacer entregas también en una tienda sirve saber en qué época del año se vende más un producto

### Árboles de decisión

Los árboles de decisión permiten saber en base a los datos que acción se debe tomar

en la vida real Netflix usa arboles de decisión para saber que películas y series nos debería recomendar

del mismo modo en las tiendas online con árboles de decisión se le recomiendan productos a los usuarios los cuales son los productos que el usuario es más propenso a comprar

### K-means

Permite crear agrupaciones en bases a las similitudes

lo cual permite agrupar de forma automático libros por los temas que trata la historia, autores, géneros etc...

también al abrir una tienda se pueden agrupar las zonas de las tiendas más exitosas y menos exitosas

también permite agrupar los diferentes tipos de transacciones que hace un usuario en una tienda online y saber si la compra fue o no un fraude

### Aprendizaje profundo

Permite usar redes neuronales convolucionales para el análisis de imágenes

Se usan en los filtros de camara que usan las redes sociales

Los autos Tesla que se conducen solos utilizan cámaras y Aprendizaje profundo para reconocer el mundo a su alrededor

se usan para analizar texto y generar modelos de generaciones de texto como GPT-3

se puede aplicar para analizar sonido y las palabras dichas

se puede usar para clasificación de imágenes en la cual identifique cuales son los elementos que existen en la imagen

## Siguientes pasos para continuar aprendendiendo de Machine Learning

Datasets para practicar Machine Learning

[Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php)

[Kaggle](https://www.kaggle.com/) la cual posee datasets y retos de machine learning
