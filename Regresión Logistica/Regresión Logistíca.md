# Clasificación de Correos Electrónicos (HAM/SPAM)
## Presentado por: Kevin Esteban Chiquillo Díaz

Este trabajo implementa un modelo de **Regresión Logística** para clasificar correos electrónicos como HAM o SPAM, usando un dataset simulado artificial con características de envío, dominio y contenido.

## Explicación de Archivos
- `dataset_correos_final.csv` : Primera versión del dataset.
- `dataset_correos_final.csv` : Dataset final con las transformaciones aplicadas.  
- `CrearDatasetArtificial.py` : Script para generar el dataset con ruido controlado.  
- `Regresion Logistica.py` : Script para entrenar el modelo de regresión logística.  
- `RegresionLogistica.md` : Este archivo.

---

## Generación del Dataset Artificial

El script `CrearDatasetArtificial.py` genera un dataset simulado de correos electrónicos con las siguientes características:

### Importación de Librerías

Para la creación del dataset artificial se importan varias librerías. pandas permite manejar y organizar los datos en DataFrames, mientras que numpy facilita operaciones numéricas y generación de números aleatorios. La librería random se utiliza para seleccionar valores aleatorios de listas, simular variabilidad en los datos y generar configuraciones diversas. Por último, datetime y timedelta se emplean para generar fechas y horas aleatorias dentro de un rango específico, representando el momento de envío de los correos electrónicos.

```python
import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta
```
 
### Configuración Inicial

Acá se definen la cantidad de instancias (1000) y la distribución de correos (50% SPAM y 50% HAM).

```python
n = 1000
n_spam = int(n * 0.50)  # 50% SPAM
n_ham = n - n_spam      # 50% HAM
```

Se definen dominios, eventos y asuntos para cada tipo de correo:

```python
dominios_comunes = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com", "aol.com", "protonmail.com"]
dominios_corporativos = ["microsoft.com", "amazon.com", "google.com", "apple.com", "facebook.com", "tesla.com", "ibm.com"]
dominios_fakes = [
    "freemail.xyz", "cheapoffers.biz", "clicknow.info", "promo-mail.net", "lottery-win.org",
    "spamworld.co", "getrichfast.ru", "prizefree.cn", "offerx.io", "phishmail.in",
    "randomfake.site", "nowfree.win", "trustme.click"
]
dominios = dominios_comunes + dominios_corporativos + dominios_fakes

eventos = ["Navidad", "San Valentín", "Black Friday", "Año Nuevo", "Ninguno"]
prob_eventos = [0.05, 0.05, 0.05, 0.05, 0.8]

asuntos_ham = [
    "Confirmación de tu compra", "Recordatorio de reunión a las 10 AM", "Informe mensual adjunto",
    "Actualización de términos y condiciones", "Factura electrónica disponible",
    "Invitación al evento corporativo", "Resumen de actividades semanales",
    "Tu pedido ha sido enviado", "Nueva actualización de seguridad disponible",
    "Notificación de inicio de sesión", "Felices fiestas de parte de nuestro equipo",
    "Cambio de contraseña solicitado", "Boletín de novedades de la empresa",
    "Detalles de tu reservación", "Recibo de pago electrónico"
]

asuntos_spam = [
    "Gana $10,000 ahora mismo", "Última oportunidad para reclamar tu premio",
    "Tu cuenta bancaria ha sido bloqueada", "Haz clic aquí para recibir tu herencia",
    "Descubre cómo bajar 10kg en una semana", "Oferta exclusiva solo para ti",
    "Entra ya y recibe tu regalo", "Confirma tus datos para no perder acceso",
    "Invierte hoy y duplica tu dinero", "Felicidades, eres el ganador",
    "Haz clic para obtener tu cupón gratis", "Consigue seguidores en minutos",
    "Accede a contenido exclusivo para adultos", "Compra medicamentos sin receta aquí",
    "Increíble promoción válida por tiempo limitado"
]
```

### Funciones Auxiliares

Se usan funciones para generar fechas aleatorias, codificar dominios y eventos, y categorizar horas y asuntos:

```python
def generar_fecha():
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    delta = end_date - start_date
    random_days = random.randrange(delta.days)
    random_seconds = random.randrange(86400)
    fecha = start_date + timedelta(days=random_days, seconds=random_seconds)
    return fecha.strftime("%Y-%m-%d %H:%M:%S")

def codificar_dominio(dominio):
    if dominio in dominios_comunes:
        return 1  # Gratuito
    elif dominio in dominios_corporativos:
        return 2  # Empresa
    else:
        return 3  # Fake

def codificar_evento(evento):
    mapa = {"Navidad": 1, "San Valentín": 2, "Black Friday": 3, "Año Nuevo": 4, "Ninguno": 5}
    return mapa[evento]

def categorizar_hora(fecha_str):
    hora = pd.to_datetime(fecha_str).hour
    if 6 <= hora < 12:
        return 1  # Mañana
    elif 12 <= hora < 18:
        return 2  # Tarde
    elif 18 <= hora < 24:
        return 3  # Noche
    else:
        return 4  # Madrugada

def categorizar_asunto(asunto):
    asunto = asunto.lower()
    if any(pal in asunto for pal in ["compra", "factura", "pago", "pedido"]):
        return 1  # Compra/Facturación
    elif any(pal in asunto for pal in ["seguridad", "contraseña", "acceso", "inicio de sesión"]):
        return 2  # Seguridad/Acceso
    elif any(pal in asunto for pal in ["evento", "reunión", "invitación", "recordatorio"]):
        return 3  # Evento/Notificación
    elif any(pal in asunto for pal in ["oferta", "promoción", "premio", "dinero", "gratis", "ganar"]):
        return 4  # Promoción/Oferta
    else:
        return 5  # Otros
```

### Generación del Dataset

Se crea una lista vacía (data) y se agrega ruido para que algunos correos SPAM se parezcan a HAM y viceversa:
```python
data = []

prob_ruido_ham = 0.15
prob_ruido_spam = 0.1
```
Esto evita que el modelo aprenda patrones triviales y permite que clasifique mejor los casos ambiguos.

### Generación de SPAM y HAM

Durante la generación del dataset, cada correo (SPAM o HAM) se crea eligiendo al azar características como el dominio del remitente, el asunto, la hora de envío y la presencia de enlaces, emojis o palabras específicas. Además, se introduce **ruido** para simular casos atípicos: un correo SPAM puede tener características típicas de HAM y viceversa, según una probabilidad definida. Esto permite que el modelo de clasificación no encuentre patrones perfectos, haciendo que el dataset sea más realista y que la regresión logística tenga que aprender a diferenciar correos incluso cuando presentan rasgos inesperados. Además se crea el dataframe y se mezcla para que no siempre en el entrenamiento se usen los mismos datos.

```python
# SPAM
for _ in range(n_spam):
    dominio = random.choice(dominios)
    evento = random.choices(eventos, weights=prob_eventos, k=1)[0]
    
    # ¿Este SPAM parece HAM? 
    ruido = np.random.rand() < prob_ruido_spam
    
    fila = {
        "FechaHoraEnvio": generar_fecha(),
        "Asunto": random.choice(asuntos_spam),
        "DominioRemitente": codificar_dominio(dominio),
        "PalabrasPremios": 0 if ruido else random.choice([0, 1]),
        "TerminosFinancieros": 0 if ruido else random.choice([0, 1]),
        "CantidadDestinatarios": random.randint(1, 5) if ruido else random.randint(5, 50),
        "TieneEmojis": 0 if ruido else random.choice([0, 1]),
        "PalabrasMalEscritas": 0 if ruido else random.choice([0, 1]),
        "CantidadEnlaces": random.randint(0, 2) if ruido else random.randint(2, 10),
        "EnlacesAcortados": 0 if ruido else random.choice([0, 1]),
        "EventosEspecificos": codificar_evento(evento),
        "TipoCorreo": 1
    }
    data.append(fila)

# HAM
for _ in range(n_ham):
    dominio = random.choice(dominios)
    evento = random.choices(eventos, weights=prob_eventos, k=1)[0]
    
    # ¿Este HAM parece SPAM? 
    ruido = np.random.rand() < prob_ruido_ham
    
    fila = {
        "FechaHoraEnvio": generar_fecha(),
        "Asunto": random.choice(asuntos_ham),
        "DominioRemitente": codificar_dominio(dominio),
        "PalabrasPremios": 1 if ruido else 0,
        "TerminosFinancieros": 1 if ruido else random.choices([0, 1], weights=[0.8, 0.2])[0],
        "CantidadDestinatarios": random.randint(10, 50) if ruido else random.randint(1, 5),
        "TieneEmojis": 1 if ruido else random.choices([0, 1], weights=[0.9, 0.1])[0],
        "PalabrasMalEscritas": 1 if ruido else random.choices([0, 1], weights=[0.9, 0.1])[0],
        "CantidadEnlaces": random.randint(5, 10) if ruido else random.randint(0, 2),
        "EnlacesAcortados": 1 if ruido else 0,
        "EventosEspecificos": codificar_evento(evento),
        "TipoCorreo": 0
    }
    data.append(fila)

# Crear DataFrame
df = pd.DataFrame(data)

# Mezclar
df = df.sample(frac=1).reset_index(drop=True)
```

### Transformaciones Finales y guardar Dataset

Se transforman las columnas para que sean numéricas y listas para entrenamiento, para posteriormente guardar el dataset en formato .csv:
```python
df["FechaHoraEnvio"] = df["FechaHoraEnvio"].apply(categorizar_hora)
df["Asunto"] = df["Asunto"].apply(categorizar_asunto)

# Guardar CSV
df.to_csv("Regresión Logistica/dataset_correos_final.csv", index=False, encoding="utf-8")
```
### Resultados

Se procede a verificar la proporción de HAM y SPAM, así como revisar las primeras filas.

```python
print("Dataset final generado con transformaciones incluidas")
print(df["TipoCorreo"].value_counts(normalize=True))
print(df.head())
```
<img width="950" height="175" alt="image" src="https://github.com/user-attachments/assets/81e54a82-554b-49ce-ad13-91044c441853" />

## Modelo de Regresión Logística

El script `Regresion Logistica.py` implementa un modelo de Regresión Logística para clasificar correos electrónicos en HAM o SPAM, evaluando su desempeño mediante Accuracy, F1 Score y matriz de confusión.

### Importación de Librerías

Se importan las librerías necesarias para manipular datos, entrenar el modelo y visualizar resultados. pandas se utiliza para trabajar con DataFrames, mientras que numpy permite realizar operaciones numéricas eficientes. De sklearn se importa train_test_split para dividir los datos en conjuntos de entrenamiento y prueba, LogisticRegression para implementar el modelo de regresión logística, y accuracy_score, f1_score y confusion_matrix para evaluar el desempeño del modelo. Finalmente, matplotlib.pyplot y seaborn se emplean para crear gráficos y visualizar la matriz de confusión.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```

### Cargar el Dataset

Se carga el dataset previamente generado y se mezcla aleatoriamente para evitar sesgos por el orden de los datos:

```python
df = pd.read_csv("Regresión Logistica/dataset_correos_final.csv")
df = df.sample(frac=1).reset_index(drop=True)
```

### Separación de Variables

Se definen las variables independientes X donde se almacenan todos los features del dataset, menos el tipo de correo, y la variable dependiente y que es donde se almacena el tipo de correo:

```python
X = df.drop(columns=["TipoCorreo"])
y = df["TipoCorreo"]
```

### Función para entrenar y evaluar el modelo

La función `entrenar_modelo` se encarga de preparar los datos, entrenar el modelo de Regresión Logística y evaluar su desempeño. Primero, divide el dataset en conjuntos de entrenamiento y prueba usando `train_test_split` (70% entrenamiento, 30% pruebas), manteniendo la proporción original de correos HAM y SPAM gracias al parámetro `stratify=y`. Luego, se crea un objeto de `LogisticRegression` con un límite de iteraciones suficientemente alto (`max_iter=1000`) para asegurar la convergencia del algoritmo. El modelo se entrena con los datos de entrenamiento (`X_train` y `y_train`) y posteriormente se generan predicciones sobre el conjunto de prueba (`X_test`). Para medir el rendimiento, se calculan métricas clave como **Accuracy** (porcentaje de predicciones correctas), **F1 Score** (equilibrio entre precisión y recall) y **la matriz de confusión** que muestra cómo se clasifican realmente los correos HAM y SPAM. Esta función devuelve estas métricas, permitiendo compararlas en diferentes ejecuciones y seleccionar el mejor modelo según el desempeño.

```python
def entrenar_modelo(X, y, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=None
    )
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return acc, f1, cm
```

### Repetición para evaluar variabilidad

Se repite el entrenamiento 50 veces (en este caso) para observar la variabilidad de los resultados y elegir la mejor iteración:

```python
n_iteraciones = 50
resultados_acc = []
resultados_f1 = []
matrices = []

for i in range(n_iteraciones):
    acc, f1, cm = entrenar_modelo(X, y)
    resultados_acc.append(acc)
    resultados_f1.append(f1)
    matrices.append(cm)
    print(f"Iteración {i+1}: Accuracy = {acc*100:.2f}%, F1 Score = {f1:.2f}")
```

### Promedio y Selección del Mejor Modelo

Después de entrenar el modelo varias veces, se analiza el desempeño de cada iteración para evaluar la variabilidad de los resultados y seleccionar el modelo más representativo. Se calculan los promedios de Accuracy y F1 Score a lo largo de todas las iteraciones, lo que permite observar la consistencia del modelo frente a distintas particiones del dataset. Para identificar la iteración óptima, se utiliza una combinación de ambas métricas, sumando el Accuracy y el F1 Score de cada iteración y seleccionando aquella que maximiza esta suma. La matriz de confusión correspondiente a esta iteración refleja la proporción de correos correctamente clasificados y los errores cometidos, permitiendo visualizar claramente cómo se desempeña el mejor modelo en la clasificación de HAM y SPAM. Además se imprimen los valores de Acurracy y F1 Score por cada Iteración.

```python
mejor_iter = np.argmax([(a + f) for a, f in zip(resultados_acc, resultados_f1)])
mejor_acc = resultados_acc[mejor_iter]
mejor_f1 = resultados_f1[mejor_iter]
mejor_cm = matrices[mejor_iter]

for i in range(n_iteraciones):
    acc, f1, cm = entrenar_modelo(X, y)
    resultados_acc.append(acc)
    resultados_f1.append(f1)
    matrices.append(cm)
    print(f"Iteración {i+1}: Accuracy = {acc*100:.2f}%, F1 Score = {f1:.2f}")
```

<img width="878" height="980" alt="image" src="https://github.com/user-attachments/assets/8e1c7735-5a20-48af-9318-dad16f758d8b" />
<img width="877" height="405" alt="image" src="https://github.com/user-attachments/assets/0ec578ba-f592-41c2-a42c-0eb4192da4b9" />

### Visualización de Métricas

Para interpretar el desempeño del modelo de forma gráfica, se generan gráficas de línea que muestran la evolución del Accuracy y del F1 Score a lo largo de todas las iteraciones. Estas gráficas permiten identificar tendencias, detectar variaciones significativas entre iteraciones y comparar cada ejecución con el promedio general. Además, se traza una línea discontinua que indica el promedio de cada métrica, proporcionando una referencia clara del comportamiento típico del modelo.

```python
plt.figure(figsize=(10,5))
plt.plot(range(1, n_iteraciones+1), [a*100 for a in resultados_acc], label="Accuracy (%)", marker="o", alpha=0.7)
plt.plot(range(1, n_iteraciones+1), [f*100 for f in resultados_f1], label="F1 Score (%)", marker="s", alpha=0.7)

plt.axhline(np.mean(resultados_acc)*100, color="blue", linestyle="--", alpha=0.6, label=f"Media Accuracy ({np.mean(resultados_acc)*100:.2f}%)")
plt.axhline(np.mean(resultados_f1)*100, color="orange", linestyle="--", alpha=0.6, label=f"Media F1 ({np.mean(resultados_f1)*100:.2f}%)")

plt.title(f"Desempeño del Modelo en {n_iteraciones} Iteraciones")
plt.xlabel("Iteración")
plt.ylabel("Porcentaje (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

<img width="921" height="486" alt="image" src="https://github.com/user-attachments/assets/5fbdab44-7c76-4edc-a673-c6efd0cc3ec8" />

### Matriz de Confusión

Una vez identificada la mejor iteración, se calcula y visualiza su matriz de confusión en porcentaje, normalizando los valores para mostrar la proporción de correos correctamente clasificados y los errores cometidos para cada clase (HAM o SPAM). Esta representación permite evaluar visualmente cómo el modelo distribuye los aciertos y fallos, evidenciando su capacidad para diferenciar correctamente entre correos legítimos y correos no deseados.

```python
print("\n Mejor modelo encontrado:")
print(f"Iteración {mejor_iter+1} con Accuracy = {mejor_acc*100:.2f}%, F1 Score = {mejor_f1:.2f}")
```
<img width="877" height="99" alt="image" src="https://github.com/user-attachments/assets/9cf444e8-7df6-41cc-bb06-6896084dfc00" />

```python
mejor_cm_pct = mejor_cm.astype('float') / mejor_cm.sum(axis=1)[:, None] * 100

plt.figure(figsize=(6,5))
sns.heatmap(mejor_cm_pct, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=["HAM", "SPAM"],
            yticklabels=["HAM", "SPAM"])
plt.title(f"Matriz de Confusión en % (Mejor modelo - Iter {mejor_iter+1})")
plt.xlabel("Etiqueta Predicción")
plt.ylabel("Etiqueta Real")
plt.show()
```

<img width="869" height="741" alt="image" src="https://github.com/user-attachments/assets/135ef080-2960-47e3-b41a-775114a5d33a" />

Esto indica que en un promedio de 50 iteraciones, el modelo tiene un exactitud de aproximadamente 78%-83%. Esto debido a que cada entrenamiento es diferente debido a la variabilidad.
