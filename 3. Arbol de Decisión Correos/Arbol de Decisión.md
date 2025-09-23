# Clasificación de Correos Electrónicos (HAM/SPAM)
## Presentado por: Kevin Esteban Chiquillo Díaz

Este trabajo implementa un modelo de Árbol de Decisión para clasificar correos electrónicos como HAM o SPAM, utilizando un dataset artificial previamente generado y transformado. El objetivo es comparar el desempeño del modelo, analizar su estabilidad en múltiples ejecuciones y visualizar gráficamente el árbol de decisión entrenado.

## Explicación de Archivos
- `dataset_correos_final.csv` : Dataset final con las transformaciones aplicadas.  
- `ArbolDecisionCorreos.py` Script para entrenar, evaluar y visualizar el árbol de decisión.  
- `Arbol de Decisión.md` : Este archivo.
---

### Importación de Librerías

En la sección de importación de librerías se incluyen los recursos necesarios para el manejo de datos, el entrenamiento del modelo y la visualización de resultados. La librería pandas permite trabajar con los datos en estructuras tipo DataFrame, facilitando su organización, filtrado y exportación. NumPy se utiliza para realizar operaciones numéricas y estadísticas, como el cálculo de medias y desviaciones estándar. Desde scikit-learn se importan varias herramientas: train_test_split de model_selection para dividir el dataset en entrenamiento y prueba manteniendo la proporción de clases, DecisionTreeClassifier, plot_tree y export_text de tree para construir el clasificador, graficar el árbol y extraer reglas en formato textual, así como accuracy_score y f1_score de metrics para evaluar el rendimiento del modelo. Finalmente, matplotlib.pyplot se emplea en la generación de gráficos, permitiendo representar tanto las métricas como la estructura del árbol de decisión.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.tree import export_text
```

### Carga y Selección de variables del Dataset

En esta parte se carga el dataset utilizando la función `read_csv` de pandas, la cual permite leer el archivo `dataset_correos_final.csv` ubicado en la carpeta. Luego, se reorganizan aleatoriamente los registros con `sample(frac=1)` y se restablecen los índices mediante `reset_index(drop=True)`.
Posteriormente, se definen las variables independientes **(X)** eliminando la columna **TipoCorreo**, mientras que la variable dependiente **(y)** queda conformada por dicha columna, que indica el tipo de correo.

```python
df = pd.read_csv("3. Arbol de Decisión Correos/dataset_correos_final.csv")
df = df.sample(frac=1).reset_index(drop=True)  

X = df.drop(columns=["TipoCorreo"])
y = df["TipoCorreo"]
```
### Función de Entrenamiento

Se define una función llamada `entrenar_arbol`, encargada de entrenar y evaluar un modelo de Árbol de Decisión. Dentro de la función, los datos se dividen en conjuntos de entrenamiento y prueba mediante `train_test_split`, manteniendo la proporción de clases con el parámetro `stratify` (30% para pruebas).
Posteriormente, se crea un modelo de `DecisionTreeClassifier` con la profundidad máxima indicada en el argumento `max_depth`. El modelo se ajusta a los datos de entrenamiento con **fit**, y luego se realizan predicciones sobre el conjunto de prueba mediante **predict**.
Finalmente, se calculan las métricas de desempeño *Accuracy* y *F1-Score* usando las funciones `accuracy_score` y `f1_score`, retornando ambos valores junto con el modelo entrenado.

```python
def entrenar_arbol(X, y, test_size=0.3, max_depth=None) :
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=None
    )
    modelo = DecisionTreeClassifier(max_depth=max_depth, random_state=None)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return acc, f1, modelo
```

### Repetición para evaluar variabilidad

Se establece un número de iteraciones igual a 50 con el fin de evaluar la estabilidad del modelo. Para almacenar los resultados, se inicializan tres listas vacías: `resultados_acc` para los valores de exactitud, `resultados_f1` para los valores de F1-Score, y `modelos` para guardar los modelos entrenados en cada ejecución.

Dentro de un ciclo *for*, en cada iteración se llama a la función `entrenar_arbol`, la cual devuelve el *accuracy*, el *F1-Score* y el modelo entrenado. Los valores obtenidos se agregan a las listas correspondientes, mientras que el modelo se almacena en la lista de modelos. Finalmente, en cada paso del ciclo, se imprime en pantalla el número de iteración junto con los resultados de exactitud y F1-Score expresados en porcentaje.

```python
# Repetir varias veces para ver estabilidad del modelo
n_iteraciones = 50
resultados_acc = []
resultados_f1 = []
modelos = []

for i in range(n_iteraciones):
    acc, f1, modelo = entrenar_arbol(X, y, max_depth=None) 
    resultados_acc.append(acc)
    resultados_f1.append(f1)
    modelos.append(modelo)
    print(f"Iteración {i+1}: Accuracy = {acc*100:.2f}%, F1 Score = {f1:.2f}")
```

<img width="369" height="606" alt="image" src="https://github.com/user-attachments/assets/e265880e-e009-44c1-a8c9-6017ea44c108" />

<img width="375" height="255" alt="image" src="https://github.com/user-attachments/assets/ddbafbbe-1c35-4207-8385-082ccff6bff9" />


### Cálculo de Estadisticas

Primero, se obtiene la media de los resultados de exactitud y F1-Score a partir de las listas generadas en las iteraciones previas. Luego, se calcula la desviación estándar para cada una de estas métricas, con el objetivo de medir la variabilidad de los resultados.
Posteriormente, se determina el Z-score para ambas métricas. En el caso de la exactitud, si la desviación estándar resulta ser cero (lo que indicaría ausencia de variación en los valores), todos los Z-scores se asignan a 0.0. De lo contrario, se calcula el Z-score de cada valor restando la media y dividiendo entre la desviación estándar. El mismo procedimiento se aplica para el F1-Score.
Finalmente, se imprimen en pantalla los estadísticos generales, mostrando la media y la desviación estándar tanto para la exactitud como para el F1-Score, expresados en porcentaje.

```python
acc_media = np.mean(resultados_acc)
f1_media = np.mean(resultados_f1)
acc_desvest = np.std(resultados_acc)
f1_desvest = np.std(resultados_f1)

if acc_desvest == 0:
    z_scores_acc = [0.0] * len(resultados_acc)
else:
    z_scores_acc = [(a - acc_media) / acc_desvest for a in resultados_acc]

if f1_desvest == 0:
    z_scores_f1 = [0.0] * len(resultados_f1)
else:
    z_scores_f1 = [(f - f1_media) / f1_desvest for f in resultados_f1]

print("\nEstadísticos generales:")
print(f"Accuracy medio: {acc_media*100:.2f}% | Desviación estándar: {acc_desvest*100:.2f}%")
print(f"F1-Score medio: {f1_media*100:.2f}% | Desviación estándar: {f1_desvest*100:.2f}%")
```
<img width="394" height="65" alt="image" src="https://github.com/user-attachments/assets/93e0e003-0130-4cc7-9ad0-91d1ee3a9ff4" />


### Selección de la mejor Iteración

En esta parte del código se selecciona la mejor iteración entre todas las realizadas. Para ello, se calcula el índice de la iteración que maximiza la suma de las métricas de exactitud y F1-Score, utilizando la función `np.argmax`.
Con este índice, se identifican los valores correspondientes de exactitud, F1-Score, así como sus respectivos Z-scores. Además, se almacena el modelo entrenado en esa iteración, considerándolo como el mejor modelo obtenido.
Finalmente, se muestran en pantalla los resultados de esta mejor iteración, incluyendo la exactitud y el F1-Score expresados en porcentaje, además de los Z-scores asociados a cada métrica.

```python
mejor_iter = int(np.argmax([(a + f) for a, f in zip(resultados_acc, resultados_f1)]))
mejor_acc = resultados_acc[mejor_iter]
mejor_f1 = resultados_f1[mejor_iter]
mejor_zscore_acc = z_scores_acc[mejor_iter]
mejor_zscore_f1 = z_scores_f1[mejor_iter]
mejor_modelo = modelos[mejor_iter]
print(f"\nMejor iteración: {mejor_iter+1} con Accuracy = {mejor_acc*100:.2f}% y F1 = {mejor_f1:.2f}")
print(f"   Exactitud = {mejor_acc*100:.2f}% | Z-Score Exactitud = {mejor_zscore_acc:.2f}")
print(f"   F1-Score  = {mejor_f1:.2f} | Z-Score F1 = {mejor_zscore_f1:.2f}")
print("\n")
```
<img width="379" height="63" alt="image" src="https://github.com/user-attachments/assets/1284e463-245b-4ee4-9c59-0b0ab8387df5" />


### Gráfica de Exactitud y F1-Score por Épocas

En esta sección se genera una gráfica comparativa entre los valores de Accuracy y F1-Score obtenidos a lo largo de todas las iteraciones.
Primero, se configuran dos curvas: una para los valores de exactitud y otra para los valores de F1-Score, ambas expresadas en porcentaje. Además, se trazan líneas horizontales que representan las medias de cada métrica, lo que permite comparar rápidamente el rendimiento de cada iteración con el promedio general.
La gráfica incluye un título descriptivo, etiquetas en los ejes, leyenda para identificar cada curva y un estilo visual con cuadrícula. De esta forma, se facilita la visualización del comportamiento del modelo a lo largo de las múltiples ejecuciones, mostrando tanto su estabilidad como las posibles variaciones en el desempeño.

```python
plt.figure(figsize=(10,5))
plt.plot(range(1, n_iteraciones+1), [a*100 for a in resultados_acc], label="Accuracy (%)", marker="o", alpha=0.7)
plt.plot(range(1, n_iteraciones+1), [f*100 for f in resultados_f1], label="F1 Score (%)", marker="s", alpha=0.7)

plt.axhline(acc_media*100, color="blue", linestyle="--", alpha=0.6, label=f"Media Accuracy ({acc_media*100:.2f}%)")
plt.axhline(f1_media*100, color="orange", linestyle="--", alpha=0.6, label=f"Media F1 ({f1_media*100:.2f}%)")

plt.title(f"Desempeño del Árbol de Decisión en {n_iteraciones} Iteraciones")
plt.xlabel("Iteración")
plt.ylabel("Porcentaje (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

<img width="997" height="562" alt="image" src="https://github.com/user-attachments/assets/a8d14c86-d628-4ca5-a7ad-134b618f88a1" />


### Gráfica de Z-Score por Épocas

En este apartado se genera una gráfica de los valores de Z-Score correspondientes a las métricas de exactitud (Accuracy) y F1-Score obtenidas en cada iteración.
Se trazan dos curvas: una en color rojo para la exactitud y otra en verde oscuro para el F1-Score. Ambas muestran cómo se comportan los resultados de cada iteración en relación con la media, ya que el Z-Score mide la desviación estándar respecto al promedio.
Además, se añade una línea horizontal en Z=0, que representa la media de referencia. Esto permite identificar con facilidad en qué iteraciones los valores están por encima o por debajo del promedio, y en qué magnitud.
La gráfica incluye título, etiquetas en los ejes, leyenda y cuadrícula.

```python
plt.figure(figsize=(10,5))

plt.plot(range(1, n_iteraciones+1), z_scores_acc, 
         label="Z-Score Exactitud", marker="o", color="red", alpha=0.8)

plt.plot(range(1, n_iteraciones+1), z_scores_f1, 
         label="Z-Score F1", marker="s", color="darkgreen", alpha=0.8)
plt.axhline(0, color="gray", linestyle="--", alpha=0.7, label="Media (Z=0)")

plt.title(f"Z-Score de Exactitud y F1 en {n_iteraciones} Iteraciones")
plt.xlabel("Iteración")
plt.ylabel("Z-Score")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

<img width="997" height="557" alt="image" src="https://github.com/user-attachments/assets/9ccb6e75-42c5-4bb1-9567-aaf62dab1746" />

### Gráfica del mejor Árbol de Decisión

Se utiliza la función `plot_tree` para representar gráficamente la estructura del árbol, mostrando de manera detallada las divisiones realizadas en cada nodo a partir de las variables predictoras. Se incluyen los nombres de las características (*feature_names*) y las clases objetivo, en este caso HAM y SPAM, lo que facilita comprender cómo se clasifican los correos electrónicos.
El parámetro `filled=True` permite colorear los nodos en función de la clase predominante, mientras que `rounded=True` mejora la estética de las cajas. Además, se configura un tamaño amplio de la figura (30x15) y una fuente pequeña para que sea posible apreciar con claridad la mayor cantidad de nodos posibles.
Finalmente, se añade un título que indica el número de iteración en la que se obtuvo este modelo, de modo que el gráfico representa el árbol de decisión más representativo y con mejor desempeño dentro del proceso de evaluación.

```python
plt.figure(figsize=(30,15))
plot_tree(mejor_modelo, filled=True, feature_names=X.columns, class_names=["HAM", "SPAM"], rounded=True, fontsize=8)
plt.title(f"Árbol de Decisión - Mejor Modelo (Iter {mejor_iter+1})")
plt.show()
```
<img width="1358" height="712" alt="image" src="https://github.com/user-attachments/assets/8e77fb05-e918-4653-92a1-4b2a4b6f60d5" />


### Reglas de Decisión 

En esta sección se imprimen las reglas de decisión correspondientes al mejor modelo encontrado.
Se emplea la función `export_tex`t, la cual transforma el árbol de decisión en un formato textual que representa las condiciones de división utilizadas en cada nodo. Estas reglas muestran de manera jerárquica las comparaciones realizadas sobre las variables predictoras para clasificar una instancia como HAM o SPAM.
El parámetro `feature_names` asegura que en lugar de mostrar índices numéricos, se presenten los nombres reales de las características, facilitando la interpretación de las reglas generadas.
De esta manera, el modelo no solo se visualiza gráficamente, sino que también se describe mediante reglas lógicas explícitas, lo que permite comprender con mayor detalle el proceso de clasificación.

```python
print("Reglas del mejor modelo:")
reglas = export_text(mejor_modelo, feature_names=list(X.columns))
print(reglas)
```
<img width="580" height="607" alt="image" src="https://github.com/user-attachments/assets/15321b7b-8356-4299-84b1-430617c98fde" />
<img width="583" height="607" alt="image" src="https://github.com/user-attachments/assets/605dab91-a24e-4280-a9a1-54cdc9548077" />
<img width="586" height="605" alt="image" src="https://github.com/user-attachments/assets/f5e5a6d4-655c-4700-84ed-a94c40768e6c" />
<img width="583" height="602" alt="image" src="https://github.com/user-attachments/assets/aa8f7ee5-280d-470e-98b4-dd75fe4c77de" />
<img width="583" height="606" alt="image" src="https://github.com/user-attachments/assets/ddce9994-6487-47d0-bf42-d1cc742efadd" />
<img width="582" height="255" alt="image" src="https://github.com/user-attachments/assets/d6f4d540-c9fc-4307-a13f-3f27dbf8d2cb" />

