import umap
from sklearn.datasets import fetch_openml
import joblib

sns.set(context="paper", style="white")

mnist = fetch_openml("mnist_784", version=1)

reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(mnist.data)

fig, ax = plt.subplots(figsize=(12, 10))
color = mnist.target.astype(int)
plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=0.1)
plt.setp(ax, xticks=[], yticks=[])
plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)

plt.show()
# %%
mnist_xy = np.c_[embedding[:, 0], embedding[:, 1]]
mnist_xy, _, mnist_labels, _ = train_test_split(mnist_xy, color, test_size=0.9)
# %%
joblib.dump((mnist_xy, mnist_labels), 'df.joblib')