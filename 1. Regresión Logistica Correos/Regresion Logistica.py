import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1️ Cargar dataset
df = pd.read_csv("Regresión Logistica/dataset_correos_final.csv")
df = df.sample(frac=1).reset_index(drop=True)


# X = variables independientes, y = variable dependiente
X = df.drop(columns=["TipoCorreo"])
y = df["TipoCorreo"]

# 2️ Función para entrenar y evaluar 
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

# 3️ Repetir varias veces el entrenamiento (variabilidad) 
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

# 4️ Promedio y variabilidad (Seleccionar la mejor iteración) 
mejor_iter = np.argmax([(a + f) for a, f in zip(resultados_acc, resultados_f1)])
mejor_acc = resultados_acc[mejor_iter]
mejor_f1 = resultados_f1[mejor_iter]
mejor_cm = matrices[mejor_iter]

# Graficar Accuracy y F1 Score por iteración
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

print("\n Mejor modelo encontrado:")
print(f"Iteración {mejor_iter+1} con Accuracy = {mejor_acc*100:.2f}%, F1 Score = {mejor_f1:.2f}")

# Graficar matriz de confusión del mejor modelo
mejor_cm_pct = mejor_cm.astype('float') / mejor_cm.sum(axis=1)[:, None] * 100

plt.figure(figsize=(6,5))
sns.heatmap(mejor_cm_pct, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=["HAM", "SPAM"],
            yticklabels=["HAM", "SPAM"])
plt.title(f"Matriz de Confusión en % (Mejor modelo - Iter {mejor_iter+1})")
plt.xlabel("Etiqueta Predicción")
plt.ylabel("Etiqueta Real")
plt.show()