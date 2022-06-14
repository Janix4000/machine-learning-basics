# %%
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from cmath import sqrt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from mpmath import nstr, mpf


df = joblib.load('df.joblib')

xy = df[:, [0, 1]]
cs = df[:, -1]

ds = (xy, cs)

h = 0.02  # step size in the mesh

# %
fig, ax = plt.subplots(figsize=(4, 4))
cm_bright = ListedColormap(
    ['#000000', "#FF0000", "#0000FF", '#00FF00',
        '#FFFF00', '#FF00FF', '#00FFFF', '#FFFFFF']
)
X, y = ds
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors="k")
# ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
ax.set_xticks(())
ax.set_yticks(())

# %%

# %%

model = KMeans(n_clusters=8, random_state=1).fit(X)
labels = model.labels_
ax.scatter(X[:, 0], X[:, 1], c=labels, cmap=cm_bright, edgecolors="k")
# ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
ax.set_xticks(())
ax.set_yticks(())
