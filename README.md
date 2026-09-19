# Activity 7 - Clustering
# 🔬 Clustering with K-Means, DBSCAN & Spectral Clustering

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange?logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/Clustering-Unsupervised%20Learning-purple" alt="Clustering">
</p>

<p align="center">
  <b>Comparación de algoritmos de aprendizaje no supervisado para identificar grupos dentro de los datos se trabajaron con los datasets de **iris, make_moons, make_circles**.</b>
</p>

---

## 📌 Descripción

Este proyecto presenta la aplicación y comparación de tres algoritmos de **Clustering** pertenecientes al aprendizaje no supervisado:

* 🎯 **K-Means**
* 🌐 **DBSCAN**
* 🧩 **Spectral Clustering**

# 🧠 Metodología

El trabajo se desarrolló siguiendo las siguientes etapas:

```text
              📊 DATOS
                 │
                 ▼
        ┌─────────────────┐
        │ Preprocesamiento│
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Escalamiento    │
        └────────┬────────┘
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
    🎯 K-Means  🌐 DBSCAN  🧩 Spectral
        │        │        │
        └────────┼────────┘
                 ▼
        📈 Evaluación
                 │
        ┌────────┼─────────┐
        ▼        ▼         ▼
    Silhouette  Calinski  Davies-
                Harabasz  Bouldin
                 │
                 ▼
          📊 Comparación
```

---

# ⚙️ 1. Preprocesamiento

Antes de aplicar los algoritmos, los datos fueron preparados para evitar que las diferencias de escala entre variables afectaran el proceso de clustering.

Se utilizó:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_escalado = scaler.fit_transform(X)
```

El escalamiento transforma las variables para que tengan aproximadamente:

```text
Media = 0
Desviación estándar = 1
```

Esto es especialmente importante para algoritmos basados en distancias, como **K-Means** y **DBSCAN**.

---

# 🎯 2. K-Means

K-Means divide los datos en un número determinado de clusters.

El algoritmo funciona de manera iterativa:

```text
1. Seleccionar centroides iniciales
          ↓
2. Asignar cada punto al centroide más cercano
          ↓
3. Recalcular los centroides
          ↓
4. Repetir hasta convergencia
```

Se implementó utilizando:

```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

labels_kmeans = kmeans.fit_predict(X_escalado)
```

### 🔎 Selección de K

Para analizar diferentes valores de `K`, se utilizó el método del **codo (Elbow Method)** mediante la inercia:

```python
inertias = []

for k in range(1, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_escalado)

    inertias.append(kmeans.inertia_)
```

La inercia mide qué tan cerca están los puntos de sus respectivos centroides.

---

# 🌐 3. DBSCAN

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** es un algoritmo basado en densidad.

A diferencia de K-Means, no requiere indicar previamente el número de clusters.

Sus principales parámetros son:

```python
DBSCAN(
    eps=...,
    min_samples=...
)
```

### `eps`

Define la distancia máxima que se considera para determinar si dos puntos son vecinos.

### `min_samples`

Define el número mínimo de puntos necesarios para considerar que una región tiene suficiente densidad.

Los puntos que no pertenecen a ningún grupo pueden ser identificados como **ruido**:

```text
Etiqueta = -1
```

Implementación:

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(
    eps=0.5,
    min_samples=5
)

labels_dbscan = dbscan.fit_predict(X_escalado)
```

Una ventaja importante de DBSCAN es su capacidad para detectar **clusters de formas arbitrarias** y separar puntos considerados ruido.

---

# 🧩 4. Spectral Clustering

Spectral Clustering utiliza una rep
