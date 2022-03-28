import requests
from numpy import random
# import shutil
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

rng = random.default_rng()

endpoint = 'https://campaignwiki.org/face/render/alex'

races = {
    'dwarf': 'eyes_dwarf_{}.png,mouth_all_{}.png,chin_dwarf_{}.png,ears_dwarf_{}.png,nose_man_woman_dwarf_{}.png,hair_dwarf_{}.png',
    'elf': 'eyes_elf_{}.png,mouth_elf_{}.png_,chin_elf_{}.png_,ears_elf_{}.png,nose_woman_elf_{}.png,hair_elf_{}.png_',
    'man': 'eyes_all_{}.png,mouth_all_{}.png,chin_man_{}.png_,ears_all_{}.png_,nose_man_woman_dwarf_{}.png,hair_man_{}.png'
}


def generate_race(race: str):
    url_format: str = endpoint + '/' + races[race]

    m = url_format.count('{')
    while True:
        params = random.randint(1, 30, size=(m,))
        url = url_format.format(*params)
        yield requests.get(url)


def main():

    race = 'man'
    i = 0
    k = 0
    n = 40

    for response in generate_race(race):
        if i >= n or k > 40 * n:
            break
        if response.status_code == 200:
            path = f'faces/{race}/{race}_{i}.png'
            image = Image.open(BytesIO(response.content))
            image.save(path)
            i += 1
            print('Yay!')
        else:
            print('fail')
        k += 1


if __name__ == '__main__':
    main()
