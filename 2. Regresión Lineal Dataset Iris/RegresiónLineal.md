# Clasificación de Plantas Usando Regresión Lineal con el Dataset IRIS
## Presentado por: Kevin Esteban Chiquillo Díaz

En este trabajo se aplica un modelo de regresión lineal con el objetivo de realizar una tarea de clasificación supervisada. Aunque la regresión lineal es un método pensado originalmente para predecir valores continuos, en este caso se adapta el enfoque para clasificar especies de flores. El procedimiento consiste en entrenar el modelo usando dos variables predictoras y luego transformar la salida continua del modelo en una etiqueta discreta mediante redondeo.

El análisis se enfoca en el uso de las variables petal length y petal width, ya que estas presentan una mayor capacidad discriminativa entre especies.

## Explicación de Archivos
- `Iris.py` : Script para entrenar el modelo de regresión lineal.
- `RegresiónLineal.md` : Este archivo.

---

## 1. Importación de Librerías
En este proyecto se importan varias librerías esenciales para el desarrollo del modelo. **NumPy** se utiliza para el manejo de arreglos y operaciones matemáticas necesarias durante el procesamiento de datos y la transformación de predicciones. **Matplotlib**, a través de su módulo pyplot, permite generar las visualizaciones que representan tanto las rectas de regresión como la frontera de decisión y la matriz de confusión. Desde **scikit-learn** se emplea **load_iris** para cargar directamente el dataset de Iris, **LinearRegression** para implementar el modelo de regresión lineal, y **train_test_split** con el fin de dividir los datos en conjuntos de entrenamiento y prueba de manera controlada. Finalmente, se importan las funciones de métricas **accuracy_score**, **classification_report** y **confusion_matrix**, que permiten evaluar el rendimiento del modelo a través de medidas de exactitud, reportes detallados por clase y un resumen gráfico de los aciertos y errores.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

## 2. Descripción y Selección de Características del Dataset Iris

El dataset de Iris es un conjunto de datos clásico en el aprendizaje automático que contiene 150 observaciones de flores pertenecientes a tres especies: Setosa, Versicolor y Virginica. Cada flor está descrita por cuatro características: longitud y ancho del sépalo, y longitud y ancho del pétalo. En este caso, se cargan los datos con la función `load_iris()` de **scikit-learn** y, para simplificar el análisis, se seleccionan únicamente dos variables: la longitud del pétalo (`petal length`) y el ancho del pétalo (`petal width`), ya que estas ofrecen una mejor separación entre las especies. La variable X contiene los valores de estas dos características para todas las flores, mientras que y almacena las etiquetas numéricas de cada especie (0 para Setosa, 1 para Versicolor y 2 para Virginica). Además, se define la lista `class_names` para asociar cada número con el nombre de la especie correspondiente, lo que facilita la interpretación de resultados y gráficas posteriores.

```python
iris = load_iris()
X = iris.data[:, 2:4]   # petal length, petal width
y = iris.target
class_names = ["Setosa", "Versicolor", "Virginica"]
```

## 3. Entrenamiento y Test del Modelo

En esta etapa se procede a dividir el conjunto de datos en subconjuntos de entrenamiento y prueba utilizando la función train_test_split, donde el 70% de los datos se destina al entrenamiento y el 30% restante a la validación. El parámetro stratify=y asegura que las proporciones de cada clase (Setosa, Versicolor y Virginica) se mantengan tanto en el conjunto de entrenamiento como en el de prueba, evitando desbalances en la distribución. Con shuffle=True se garantiza que los datos se mezclen de manera aleatoria antes de la división, reduciendo posibles sesgos derivados del orden en el que se encuentran. Finalmente, random_state=None indica que no se fija una semilla aleatoria, por lo que en cada ejecución los datos se dividen de forma distinta, permitiendo mayor variabilidad en las pruebas. Se aplica el modelo de regresión lineal con el fin de aproximar la clasificación de las especies de iris en función de la longitud y el ancho de los pétalos. Posteriormente, se realizan predicciones sobre el conjunto de prueba y, dado que la regresión lineal produce valores continuos, se convierten en categorías discretas mediante redondeo y limitación de valores. Finalmente, se evalúa el desempeño del modelo calculando la exactitud, el reporte de clasificación y la matriz de confusión normalizada, lo cual permite observar en porcentajes el nivel de acierto en cada clase.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, shuffle=True, random_state=None
)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_cont = model.predict(X_test)
y_pred = np.clip(np.rint(y_pred_cont), 0, 2).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
```

### 4. Visualización de las rectas de regresión

En este apartado se representan gráficamente los datos de prueba junto con las rectas de regresión obtenidas por el modelo. Primero, se extraen los coeficientes y el intercepto de la regresión lineal, los cuales permiten construir las ecuaciones de las rectas. Los puntos del conjunto de prueba se muestran en un diagrama de dispersión, donde cada color corresponde a una clase de flor. Además, los puntos que fueron mal clasificados se resaltan con un círculo rojo vacío, facilitando la identificación de errores del modelo.

Las líneas trazadas corresponden a los umbrales de decisión que separan las clases, en este caso con valores aproximados de 0.5 y 1.5. Estas rectas dividen el espacio de características en regiones donde el modelo predice una u otra especie. De este modo, se puede observar visualmente cómo el modelo intenta clasificar las plantas en función de la longitud y el ancho de los pétalos.

```python
plt.figure(figsize=(7,6))
a, b = model.coef_
c = model.intercept_

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="viridis", edgecolor="k", s=60)
mis = (y_test != y_pred)
plt.scatter(X_test[mis, 0], X_test[mis, 1],
            facecolors='none', edgecolors='red', s=120, linewidths=2, label="Mal clasificados")

x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
for threshold, color in zip([0.5, 1.5], ["red", "blue"]):
    y_vals = -(a * x_vals + c - threshold) / b
    plt.plot(x_vals, y_vals, color=color, linewidth=2, label=f"Frontera ≈ {threshold}")

plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("Rectas de regresión (Iris)")
plt.legend(loc="best")
plt.show()
```
<img width="697" height="657" alt="image" src="https://github.com/user-attachments/assets/2ea611e0-daf4-4301-8803-26a8c4db07cc"/>

### 5. Frontera de decisión del modelo

En este apartado se construye una visualización de la frontera de decisión generada por la regresión lineal. Para ello, primero se define una malla de puntos que cubre todo el espacio de las variables seleccionadas: la longitud y el ancho de los pétalos. A cada punto de la malla se le aplica el modelo entrenado, lo que permite predecir a qué clase pertenecería.

Los resultados se representan con un mapa de colores mediante contourf, donde cada región coloreada corresponde a una clase diferente según el modelo. Encima de esta superficie se grafican los puntos reales del conjunto de prueba, lo que permite comparar visualmente las predicciones del modelo con los datos observados.

De esta manera, se puede analizar cómo la regresión lineal divide el espacio de características en distintas zonas, y hasta qué punto esas divisiones coinciden con la distribución real de las especies en el dataset.

```python
plt.figure(figsize=(7,6))
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.clip(np.rint(Z), 0, 2).astype(int).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="viridis", edgecolor="k", s=60)
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("Frontera de decisión (Iris)")
plt.show()
```

<img width="697" height="659" alt="image" src="https://github.com/user-attachments/assets/328edc7f-a175-42cd-bdfb-bde446e3316c" />

### 6. Matriz de Confusión

En este caso, la matriz se ha normalizado para mostrar los resultados en porcentajes, lo cual facilita la interpretación del desempeño del modelo en cada clase.

En la visualización, cada celda indica el porcentaje de muestras de una clase real que fueron clasificadas como pertenecientes a otra clase (o a la misma, en el caso de la diagonal principal). Una diagonal con valores altos cercanos al 100% refleja un buen desempeño, mientras que porcentajes elevados fuera de la diagonal indican errores de clasificación.

```python
plt.figure(figsize=(6,5))
im = plt.imshow(cm_norm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, fraction=0.046, pad=0.04)

plt.title("Matriz de confusión (%)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha='right')
plt.yticks(np.arange(len(class_names)), class_names)

for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        text = f"{cm_norm[i, j]*100:.1f}%"
        color = "white" if cm_norm[i, j] > 0.5 else "black"
        plt.text(j, i, text, ha="center", va="center", color=color)

plt.tight_layout()
plt.show()
```

<img width="597" height="560" alt="image" src="https://github.com/user-attachments/assets/ce8cb0f8-80f7-4f33-ad85-7378c0ad3866" />

### 7. Clasificación de todo el dataset

En esta sección, el modelo entrenado se aplica a la totalidad del conjunto de datos de Iris. Cada muestra se clasifica según las predicciones obtenidas a partir de la regresión lineal, redondeadas y ajustadas para corresponder a una de las tres clases posibles.

El gráfico resultante muestra todas las observaciones del dataset, coloreadas de acuerdo con la clase asignada por el modelo. Esto permite visualizar cómo se distribuyen las predicciones sobre el espacio de características definido por la longitud y el ancho de los pétalos.

Con esta representación global, es posible identificar patrones generales en la clasificación y evaluar de manera visual si el modelo logra una separación clara entre las especies de iris en el plano bidimensional.

```python
y_all_pred_cont = model.predict(X)
y_all_pred = np.clip(np.rint(y_all_pred_cont), 0, 2).astype(int)

plt.figure(figsize=(7,6))
plt.scatter(X[:, 0], X[:, 1], c=y_all_pred, cmap="viridis", edgecolor="k", s=60)

plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("Clasificación de TODO el dataset (Iris)")
plt.show()
```

<img width="694" height="660" alt="image" src="https://github.com/user-attachments/assets/cc80a630-7bbd-4cdb-84b6-8368d474660d" />

En las representaciones gráficas, se utilizó los siguientes colores para representar cada planta:
- Puntos morados → Setosa
- Puntos azulados → Versicolor
- Puntos amarillos → Virginica


