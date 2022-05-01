from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import numpy as np
from joblib import dump
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA


def preprocess_image(image):
    width, height = image.size

    # Setting the points for cropped image
    size = min(width, height)
    left = (width - size) / 2
    top = (height - size) / 2
    right = width - (width - size) / 2
    bottom = height - (height - size) / 2

    # Cropped image of above dimension
    # (It will not change original image)
    image = image.crop((left, top, right, bottom))
    image = ImageOps.grayscale(image)
    image = image.resize((64, 64))
    # Shows the image in image viewer
    return image


def prepare_data(n: int):
    path = './faces'
    races = ['elf', 'dwarf', 'orc']

    data = []

    for race in races:
        for i in range(n):
            filename = f'{path}/{race}/{race}_{i}.png'
            image = Image.open(filename)
            image = preprocess_image(image)
            image = np.array(image)
            data.append(image.flatten())
    return np.array(data)


data = prepare_data(n=20)
mean = np.mean(data, axis=0)

pca = PCA()
data_r = pca.fit_transform(data)

dump(pca, 'pca.joblib')
dump(data, 'data.joblib')

# data_r[:, 27:] = 0

# transformed = data_r @ pca.components_


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
