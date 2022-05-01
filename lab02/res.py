# %%
from itertools import chain
from joblib import load
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
from math import sqrt, ceil
from matplotlib import pyplot as plt
import numpy as np


data = load('./data.joblib')
pca = load('./pca.joblib')
mean = np.mean(data, axis=0)


def best_divs(n: int):
    a, b = 1, n
    for i in range(2, int(ceil(sqrt(n)))):
        if n % i == 0:
            a, b = i, n // i
    return a, b


def show_images(images, *args, **kwargs):
    n = len(images)
    a, b = best_divs(n)
    fig, axss = plt.subplots(a, b, figsize=(b * 4, a * 4))
    for im, ax in zip(images, axss.flatten()):
        ax.imshow(im, *args, **kwargs)

    return fig, axss

# %%


def create_reduced(pca, data, dim: int):

    data_r = pca.transform(data)

    data_r[:, dim:] = 0
    transformed = data_r @ pca.components_ + mean

    fig, axs = show_images(
        list(
            map(
                lambda x: x.reshape(64, 64),
                pca.components_[:dim]
            )
        ),
        cmap='gray'
    )
    fig.savefig(f'res/comps_{dim}.png')

    for i, race in enumerate(['elves', 'dwarfs', 'orcs']):
        fig, axs = show_images(
            list(
                map(
                    lambda x: x.reshape(64, 64),
                    chain(
                        *zip(data[20 * i:20 * (i + 1)], transformed[20 * i:20 * (i + 1)])
                    )
                )
            ),
            cmap='gray', vmin=0, vmax=255
        )
        fig.savefig(f'./res/{race}_{dim}.png')


# %%

create_reduced(pca, data, dim=3)

# %%
# fig, axs = plt.subplots(4, 10)
# for index, (ax0, ax1) in zip(range(0, 20), axs.reshape(20, 2)):
#     ax0.imshow(data[index].reshape(64, 64), cmap='gray', vmin=0, vmax=255)
#     ax1.imshow(transformed[index].reshape(
#         64, 64), cmap='gray', vmin=0, vmax=255)
# plt.show()

# fig, axs = plt.subplots(4, 10)
# for index, (ax0, ax1) in zip(range(20, 40), axs.reshape(20, 2)):
#     ax0.imshow(data[index].reshape(64, 64), cmap='gray', vmin=0, vmax=255)
#     ax1.imshow(transformed[index].reshape(
#         64, 64), cmap='gray', vmin=0, vmax=255)
# plt.show()

# fig, axs = plt.subplots(4, 10)
# for index, (ax0, ax1) in zip(range(40, 60), axs.reshape(20, 2)):
#     ax0.imshow(data[index].reshape(64, 64), cmap='gray', vmin=0, vmax=255)
#     ax1.imshow(transformed[index].reshape(
#         64, 64), cmap='gray', vmin=0, vmax=255)
# plt.show()


# var_or = np.cov(data.T)
# var_r = np.cov(data_r.T)

# fig, ax = plt.subplots()
# ax.imshow((mean).reshape(64, 64), cmap='gray', vmin=0, vmax=255)
# plt.show()

# fig, (ax0, ax1) = plt.subplots(1, 2)
# ax0.imshow(var_or, cmap='gray', vmin=0, vmax=255)
# ax1.imshow(var_r, cmap='gray', vmin=0, vmax=255)
# plt.show()

# fig, (ax0, ax1) = plt.subplots(1, 2)
# ax0.hist(var_or.diagonal())
# ax1.hist(var_r.diagonal())
# plt.show()

# fig, axs = plt.subplots(6, 10, figsize=(20, 20))
# for ax, vec in zip(axs.flatten(), pca.components_):
#     ax.imshow(vec.reshape(64, 64), cmap='gray')
# plt.show()


# ax = plt.axes(projection='3d')
# colors = ["navy", "turquoise", "darkorange"]
# lw = 2

# target_names = ['elves', 'dwarfs', 'orcs']

# for color, i, target_name in zip(colors, [0, 1, 2], target_names):
#     xs = data_r[20 * i:20 * (i + 1), 0]
#     ys = data_r[20 * i:20 * (i + 1), 1]
#     zs = data_r[20 * i:20 * (i + 1), 2]
#     ax.scatter3D(
#         xs, ys, zs, color=color, alpha=0.8, lw=lw, label=target_name
#     )

# plt.legend()
# plt.show()

# fig, ax = plt.subplots()
# colors = ["navy", "turquoise", "darkorange"]
# lw = 2

# target_names = ['elves', 'dwarfs', 'orcs']

# for color, i, target_name in zip(colors, [0, 1, 2], target_names):
#     xs = data_r[20 * i:20 * (i + 1), 0]
#     ys = data_r[20 * i:20 * (i + 1), 1]
#     ax.scatter(
#         xs, ys, color=color, alpha=0.8, lw=lw, label=target_name
#     )


# def getImage(image, zoom=1):
#     return OffsetImage(image, zoom=zoom)


# for x0, y0, image in zip(data_r[:, 0], data_r[:, 1], data.reshape(-1, 64, 64)):
#     ab = AnnotationBbox(getImage(image, zoom=0.8), (x0, y0), frameon=False)
#     ax.add_artist(ab)

# plt.legend()
# plt.show()
# %%
