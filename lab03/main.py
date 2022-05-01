from cmath import sqrt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

df = joblib.load('df.joblib')

xy = df[:, [0, 1]]
cs = df[:, -1]

ds = (xy, cs)

h = 0.02  # step size in the mesh

names = [
    "Linear SVM",
    "First coord SVM",
    "RBF SVM",
    "Cube SVM",
    "Partially linear SVM",
    "Quasi cosine SVM",
]


def part_linear(l, r, R):
    m = cdist(l, r)
    res = (R - m) / R
    res[res < 0] = 0
    return res


def quasi_cosine(l, r):
    centroid = np.mean(l, axis=0)
    l = l - centroid
    r = r - centroid
    dot = l.dot(r.T)
    norm = np.linalg.norm(l) * np.linalg.norm(r).reshape(-1, 1)
    return dot / norm


classifiers = [
    LinearSVC(C=0.025),
    SVC(kernel=lambda l, r: l[:, 0].reshape(-1, 1) * r[:, 0].reshape(1, -1)),
    SVC(kernel='rbf', gamma=4, C=1),
    SVC(kernel=lambda l, r: cdist(l, r) < 1),
    SVC(kernel=lambda l, r: part_linear(l, r, R=1)),
    SVC(kernel=quasi_cosine),
]


# plt.imshow(quasi_cosine(xy, xy))
# plt.show()

figure = plt.figure(figsize=(27, 7))
i = 1
# iterate over datasets
# preprocess dataset, split into training and test part
X, y = ds
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])
ax = plt.subplot(1, len(classifiers) + 1, i)
ax.set_title("Input data")
# Plot the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
           cmap=cm_bright, edgecolors="k")
# Plot the testing points
ax.scatter(
    X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())
i += 1

# iterate over classifiers
for name, clf in zip(names, classifiers):
    ax = plt.subplot(1, len(classifiers) + 1, i)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

    # Plot the training points
    ax.scatter(
        X_train[:, 0], X_train[:,
                               1], c=y_train, cmap=cm_bright, edgecolors="k"
    )
    # Plot the testing points
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap=cm_bright,
        edgecolors="k",
        alpha=0.6,
    )

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    ax.text(
        xx.max() - 0.3,
        yy.min() + 0.3,
        ("%.2f" % score).lstrip("0"),
        size=15,
        horizontalalignment="right",
    )
    i += 1

plt.tight_layout()
plt.show()
