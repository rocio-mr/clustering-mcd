# Activity 7 - Clustering
# 🔬 Clustering with K-Means, DBSCAN & Spectral Clustering

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange?logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/Clustering-Unsupervised%20Learning-purple" alt="Clustering">
</p>

<p align="center">
  <b>Comparación de algoritmos de aprendizaje no supervisado para identificar grupos dentro de los datos se trabajaron con tres datasets: iris, make_moons, make_circles.</b>
</p>

---

## 📌 Descripción

Este proyecto presenta la aplicación y comparación de tres algoritmos de **Clustering** que pertenecen al aprendizaje no supervisado:

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
        📈 Evaluación externa e interna
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

Spectral Clustering utiliza una representación basada en un grafo de similitud entre los datos.

Conceptualmente:

Datos
  │
  ▼
Grafo de similitud
  │
  ▼
Matriz de similitud
  │
  ▼
Representación espectral
  │
  ▼
Clustering

Implementación:

from sklearn.cluster import SpectralClustering

spectral = SpectralClustering(
    n_clusters=3,
    affinity="nearest_neighbors",
    random_state=42
)

labels_spectral = spectral.fit_predict(X_escalado)

Cuando se utiliza:

affinity="nearest_neighbors"

el algoritmo construye las relaciones entre puntos utilizando sus vecinos más cercanos.

Esto permite identificar estructuras que pueden ser difíciles de encontrar mediante métodos basados únicamente en centroides.

### 5. 📚 Resolución de interrogantes

**Determine qué modelo o modelos son de naturaleza lineal o no lineal, así como los hiperparámetros críticos para el entrenamiento exitoso.**

El modelo que posee una naturaleza lineal es KMeans, debido a que divide a los datos alrededor de centroides utilizando la distancia entre los puntos y dichos centroides. Los clústers que genera tienden a ser convexos o esféricos, por ello pueden presentar dificultades cuando los datos presentan formas complejas o no separables mediante este tipo de agrupamiento. 
En cuanto a los modelos DBSCAN y Spectral Clustering, ambos son no lineales. DBSCAN tiene la capacidad de identificar clústers de formas arbitrarias basándose en la densidad de los datos, además de identificar puntos considerados como ruido. Por su parte, Spectral Clustering utiliza un grafo de similitud entre los datos para identificar estructuras complejas y relaciones entre los puntos que no necesariamente pueden ser detectados mediante una separación basada únicamente en distancias a centroides. 

Estas diferencias resultaron vitales al trabajar con nuestros datasets ya que K-Means se adecuó un poco mejor al dataset de Iris que presentaba una estructura aproximadamente compacta y convexa, DBSCAN y Spectral Clustering por su parte se adaptaron mejor a datasets con estructuras no convencionales como lo fueron moons y circles. 

En lo que respecta a los **hiperparámetros** para KMeans el más importante resultó ser el número de clusters, que se pudo calcular mediante el uso del Método del Codo, aparte de conocer de antemano la cantidad de clases también ya que se trabajó con un dataset eitquetado. En cuanto a DBSCAN, el hiperparámetro que resultó muy importante porque si se configuraba un valor pequeño varios valores se convertían en ruido mientras que si por el contrario se colocaban valores más altos se corría el riesgo de que los clústers terminen uniéndose. Finalmente, Spectral Clustering tuvo dos hiperparámetros críticos el número de clusters y affinity, ya que si colocamos nearest_neigbors y probamos con varios valores esto mejora o empeora el desempeño del modelo. 

**🌸 Las métricas de validación externa e interna obtenidas en el caso Iris ¿qué significan?**

Las métricas de validación externa comparan los clusters que fueron obtenidos por cada algoritmo con las clases reales de Iris. Para KMeans, el ARI fue de 0.4328 y un V-measure de 0.5895, lo que indica que existe una correspondencia moderada entre los clusters generados y las clases reales del dataset. 
En DBSCAN, se obtuvo un ARI de 0.5517 y un V-measure de 0.6899. Estos valores indican que existió una mayor correspondencia de clases en comparación con KMeans. 
Por otro lado, Spectral Clustering obtuvo un ARI de 0.6464 y un V-measure de 0.6837. El valor de ARI más alto de los tres modelos, lo que indica que su asignación de clusters es más exacto con las clases reales de Iris. Su V-measure también presenta un valor alto y cercano al obtenido por DBSCAN.
En lo referente a la validación interna, quien tuvo un mejor desenvolvimiento para el Silhouette Score fue el DBSCAN, obtuvo el valor más alto con un 0.597941 lo cual indica que generó clusters más compactos y mejor separados. En la métrica Davies-Bouldin Index, mientras mas bajo sea un valor es mejor, por lo tanto DBSCAN obtuvo el valor más bajo con un 0.568802 lo que significa que sus clusteres presentan una mejor relación entre compactación y separación. La última métrica fue Calinski-Harabasz Index, aquí mientras más alto sea el valor es mejor. Y también el DBSCAN obtuvo el valor más alto con un 277.650960. 

**🌙 Las métricas de validación externa e interna obtenidas en el caso moons ¿qué significan?**

En cuanto al conjunto de datos Moons, las métricas de validación externas muestran una clara diferencia entre KMeans y los métodos no lineales, ya que KMeans obtuvo un ARI de 0.4451 y un V-measure de 0.3516, lo que indica una correspondencia limitada entre los clústeres que fueron generados y las clases reales. En cambio, DBSCAN y Spectral Clustering obtuvieron un ARI y V-measure del 1.0000, lo que significa una correspondencia perfecta con las etiquetas del conjunto de datos. 
Por otro lado, las métricas de validación interna, K-Means obtuvo un Silhouette Score de 0.496150, un Davies-Bouldin Index de 0.810714 y un índice de Kalinski-Harabasz de 699.075526. DBSCAN y Spectral Clustering obtuvieron un Silhouette Score de 0.389338, un Davies-Bouldin Index de 1.017883 y un índice de Calinski-Harabasz de 438.387614. Por ello, se deduce que las métricas que resultaron con lo valores más favorales fuer KMeans.


**⭕ Las métricas de validación externa e interna obtenidas en el caso circles ¿qué significan?**

En el conjunto de datos de Circles, las métricas de validación externa nos muestran las diferencias significativas entre los modelos KMeans que obtuvo un ARI de -0.0024 y un V-measure de 0.0072, lo que muestra una correspondencía casi nula entre los clusteres generados y los reales. Por otro lado, tanto DBSCAN y Spectral Clustering obtuvieron un ARI de 1.000000 y un V-measure de 1.000000, lo que indica la correspondencia perfecta.
Respecto a las métricas de validación interna, K-Means obtuvo un Silhouette Score de 0.354496, un Davies-Bouldin Index de 1.175980 y un índice de Calinski-Harabasz de 438.387614. DBSCAN y Spectral Clustering obtuvieron un Silhouette Score de 0.110755, un Davies-Bouldin Index de 331.956363 y un índice de Calinski-Harabasz de 438.387614. Por lo tanto, las métricas internas presentan valores más favorables para K-Means en Silhouette y Davies-Bouldin, mientras que los tres modelos presentan el mismo valor en Calinski-Harabasz.

