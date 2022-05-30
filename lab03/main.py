# %%
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


def ff(x: float, n: int) -> str:
    return nstr(mpf(x), n)


df = joblib.load('df.joblib')

xy = df[:, [0, 1]]
cs = df[:, -1]

ds = (xy, cs)

h = 0.02  # step size in the mesh


def first_coord(l, r, gamma=None):
    return l[:, 0].reshape(-1, 1) * r[:, 0].reshape(1, -1)


def angular(l, r, gamma):
    return cdist(l, r) < gamma


def part_linear(l, r, gamma):
    m = cdist(l, r)
    res = (gamma - m) / gamma
    res[res < 0] = 0
    return res


def quasi_cosine(l, r, gamma=None):
    centroid = np.mean(l, axis=0)
    l = l - centroid
    r = r - centroid
    dot = l.dot(r.T)
    norm = np.linalg.norm(l) * np.linalg.norm(r).reshape(-1, 1)
    return dot / norm


def best_shape(n: int) -> tuple[int, int]:
    a, b = 1, n
    x = a
    while x**2 <= n:
        if n % x == 0:
            a, b = x, n // x
        x += 1
    if a % 2 == 0 and b % 2 == 1:
        a, b = b, a
    return a, b


def show_kernels(ds, classifiers, names, h=h, plot_input=True, plot_contour=False):
    i = 1
    # iterate over datasets
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    n_plots = len(classifiers)
    if plot_contour:
        n_plots *= 2
    if plot_input:
        n_plots += 1
    fig_shape = best_shape(n_plots)
    fig = plt.figure(figsize=(fig_shape[1] * 4, fig_shape[0] * 4))
    ax = plt.subplot(*fig_shape, i)
    ax.set_title("Input data")
    # Plot the training points
    if plot_input:
        ax.scatter(X[:, 0], X[:, 1], c=y,
                   cmap=cm_bright, edgecolors="k")
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(*fig_shape, i)
        i += 1
        if plot_contour:
            axr = plt.subplot(*fig_shape, i)
            i += 1
        clf.fit(X, y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8, levels=128)

        if plot_contour:
            axr.contourf(xx, yy, np.exp(-Z**2), cmap='binary', levels=128)

        # Plot the training points
        ax.scatter(
            X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors="k"
        )
        # Plot the testing points

        svs = clf.support_vectors_
        if svs.shape != (0, 0):
            sc = ax.scatter(
                svs[:, 0],
                svs[:, 1],
                edgecolors="k",
                zorder=2,
                s=10**2,
                marker='*',
            )
            sc.set_facecolor("none")
        ax.scatter(
            X[:, 0],
            X[:, 1],
            c=y,
            cmap=cm_bright,
            edgecolors="k",
            # alpha=0.6,
            zorder=3,
        )
        ax.set_title(name)
        axr.set_title('Decision boundary')
        axs = (ax, axr) if plot_contour else (ax, )
        for ax in axs:
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())

    return fig, axs[0]


# %
fig, ax = plt.subplots(figsize=(4, 4))
cm_bright = ListedColormap(["#FF0000", "#0000FF"])
X, y = ds
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors="k")
ax.set_xticks(())
ax.set_yticks(())
fig.savefig('pts.png', dpi=300)
# %%
names = [
    "Linear SVM",
    "First coord SVM",
    "RBF SVM",
    "Angular SVM",
    "Partially linear SVM",
    "Quasi cosine SVM",
]

classifiers = [
    SVC(kernel='linear', C=0.025),
    SVC(kernel=first_coord),
    SVC(kernel='rbf', gamma=4, C=1),
    SVC(kernel=lambda l, r: angular(l, r, gamma=1.7)),
    SVC(kernel=lambda l, r: part_linear(l, r, gamma=1)),
    SVC(kernel=quasi_cosine),
]
fig, ax = show_kernels(ds, classifiers, names,
                       plot_input=False, plot_contour=True)

fig.savefig('all.png', dpi=300)
# plt.show()

# %%

cs = np.logspace(2, -5, 8, endpoint=True, base=10)

names = [
    f"Linear SVM $C$={ff(c, 2)}" for c in cs
]

classifiers = [
    SVC(kernel='linear', C=c) for c in cs
]

fig, ax = show_kernels(ds, classifiers, names,
                       plot_input=False, plot_contour=True)
fig.savefig('lin.png', dpi=300)
# %%
cs = np.logspace(2, -5, 8, endpoint=True, base=10)

names = [
    f"First coord SVM C={ff(c, 2)}" for c in cs
]

classifiers = [
    SVC(kernel=first_coord, C=c) for c in cs
]

fig, ax = show_kernels(ds, classifiers, names,
                       plot_input=False, plot_contour=True)
fig.savefig('x.png', dpi=300)
# %%
gammas = [20, 10, 5, 2, 1, 0.5, 0.1, 0.001]

names = [
    f"RBF SVM $\gamma$={ff(c, 2)}" for c in gammas
]

classifiers = [
    SVC(kernel='rbf', gamma=c) for c in gammas
]

fig, ax = show_kernels(ds, classifiers, names,
                       plot_input=False, plot_contour=True)
fig.savefig('rbf.png', dpi=300)
# %%
gammas = [10, 5, 3.5, 2, 1.5, 1, 0.5, 0.1]

names = [
    f"Angular SVM $R$={ff(c, 2)}" for c in gammas
]

classifiers = [
    SVC(kernel=lambda l, r: angular(l, r, gamma=gammas[0])),
    SVC(kernel=lambda l, r: angular(l, r, gamma=gammas[1])),
    SVC(kernel=lambda l, r: angular(l, r, gamma=gammas[2])),
    SVC(kernel=lambda l, r: angular(l, r, gamma=gammas[3])),
    SVC(kernel=lambda l, r: angular(l, r, gamma=gammas[4])),
    SVC(kernel=lambda l, r: angular(l, r, gamma=gammas[5])),
    SVC(kernel=lambda l, r: angular(l, r, gamma=gammas[6])),
    SVC(kernel=lambda l, r: angular(l, r, gamma=gammas[7])),
]


fig, ax = show_kernels(ds, classifiers, names,
                       plot_input=False, plot_contour=True)
fig.savefig('kanc.png', dpi=300)
# %%
gammas = [100, 10, 3.5, 2, 1.5, 1, 0.5, 0.1]

names = [
    f"Partially linear SVM $R$={ff(c, 2)}" for c in gammas
]

classifiers = [
    SVC(kernel=lambda l, r: part_linear(l, r, gamma=gammas[0])),
    SVC(kernel=lambda l, r: part_linear(l, r, gamma=gammas[1])),
    SVC(kernel=lambda l, r: part_linear(l, r, gamma=gammas[2])),
    SVC(kernel=lambda l, r: part_linear(l, r, gamma=gammas[3])),
    SVC(kernel=lambda l, r: part_linear(l, r, gamma=gammas[4])),
    SVC(kernel=lambda l, r: part_linear(l, r, gamma=gammas[5])),
    SVC(kernel=lambda l, r: part_linear(l, r, gamma=gammas[6])),
    SVC(kernel=lambda l, r: part_linear(l, r, gamma=gammas[7])),
]

fig, ax = show_kernels(ds, classifiers, names,
                       plot_input=False, plot_contour=True)
fig.savefig('part_lin.png', dpi=300)
# %%
gammas = [20, 10, 5, 2, 1, 0.5, 0.1, 0.001]

names = [
    f"Quasi cosine SVM $C$={ff(c, 2)}" for c in gammas
]

classifiers = [
    SVC(kernel=quasi_cosine, C=c) for c in gammas
]

fig, ax = show_kernels(ds, classifiers, names,
                       plot_input=False, plot_contour=True)
fig.savefig('cos.png', dpi=300)
# %%
