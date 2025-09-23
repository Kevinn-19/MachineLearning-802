import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.tree import export_text

# Cargar dataset
df = pd.read_csv("3. Arbol de Decisión Correos/dataset_correos_final.csv")
df = df.sample(frac=1).reset_index(drop=True)  

# Variables independientes (X) y dependiente (y)
X = df.drop(columns=["TipoCorreo"])
y = df["TipoCorreo"]

# Función para entrenar y evaluar con Árbol de Decisión
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

# Calcular estadísticas y Z-score
acc_media = np.mean(resultados_acc)
f1_media = np.mean(resultados_f1)
acc_desvest = np.std(resultados_acc)
f1_desvest = np.std(resultados_f1)

# Z-scores Accuracy
if acc_desvest == 0:
    z_scores_acc = [0.0] * len(resultados_acc)
else:
    z_scores_acc = [(a - acc_media) / acc_desvest for a in resultados_acc]

# Z-scores F1 
if f1_desvest == 0:
    z_scores_f1 = [0.0] * len(resultados_f1)
else:
    z_scores_f1 = [(f - f1_media) / f1_desvest for f in resultados_f1]

print("\nEstadísticos generales:")
print(f"Accuracy medio: {acc_media*100:.2f}% | Desviación estándar: {acc_desvest*100:.2f}%")
print(f"F1-Score medio: {f1_media*100:.2f}% | Desviación estándar: {f1_desvest*100:.2f}%")

# Seleccionar mejor iteración
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

# Gráfica Accuracy y F1 Score
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

# Gráfica Z-Scores
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

# Graficar el mejor árbol de decisión

plt.figure(figsize=(30,15))
plot_tree(mejor_modelo, filled=True, feature_names=X.columns, class_names=["HAM", "SPAM"], rounded=True, fontsize=8)
plt.title(f"Árbol de Decisión - Mejor Modelo (Iter {mejor_iter+1})")
plt.show()

print("Reglas del mejor modelo:")
reglas = export_text(mejor_modelo, feature_names=list(X.columns))
print(reglas)