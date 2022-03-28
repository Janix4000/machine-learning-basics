from PIL import Image, ImageOps
import numpy as np


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
    image = image.resize((100, 100))
    # Shows the image in image viewer
    return image


def prepare_data():
    path = './faces'
    races = ['elf', 'dwarf', 'man']
    n = 3

    data = []

    for race in races:
        for i in range(n):
            filename = f'{path}/{race}/{race}_{i}.png'
            image = Image.open(filename)
            image = preprocess_image(image)
            image = np.array(image)
            data.append(image)
    return np.array(data)


data = prepare_data()

print(data.shape)
