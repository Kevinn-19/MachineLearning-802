import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------
# 1) Datos y modelo
# -----------------------
iris = load_iris()
X = iris.data[:, 2:4]   # petal length, petal width
y = iris.target
class_names = ["Setosa", "Versicolor", "Virginica"]

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
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]  # porcentajes por fila

# -----------------------
# 2) Rectas de regresión
# -----------------------
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

# -----------------------
# 3) Frontera de decisión
# -----------------------
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

# -----------------------
# 4) Matriz de confusión (solo %)
# -----------------------
plt.figure(figsize=(6,5))
im = plt.imshow(cm_norm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, fraction=0.046, pad=0.04)

plt.title("Matriz de confusión (%)")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.xticks(np.arange(len(class_names)), class_names, rotation=45, ha='right')
plt.yticks(np.arange(len(class_names)), class_names)

# Mostrar % con 1 decimal
for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        text = f"{cm_norm[i, j]*100:.1f}%"
        color = "white" if cm_norm[i, j] > 0.5 else "black"
        plt.text(j, i, text, ha="center", va="center", color=color)

plt.tight_layout()
plt.show()

# -----------------------
# 5) Clasificación de TODO el dataset
# -----------------------
y_all_pred_cont = model.predict(X)
y_all_pred = np.clip(np.rint(y_all_pred_cont), 0, 2).astype(int)

plt.figure(figsize=(7,6))
plt.scatter(X[:, 0], X[:, 1], c=y_all_pred, cmap="viridis", edgecolor="k", s=60)

plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("Clasificación de TODO el dataset (Iris)")
plt.show()

