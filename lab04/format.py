from PIL import Image
import numpy as np
import joblib
import matplotlib.pyplot as plt


def main():
    with Image.open('clusters.png') as img:

        n = 8
        img = np.array(img)

        cs_xy = [
            np.c_[np.where(img == c)] for c in range(1, n + 1)
        ]
        cs_c = [np.ones(shape=xy.shape[0]) * c for c, xy in enumerate(cs_xy)]

        rng = np.random.RandomState(3)

        xy = np.concatenate(cs_xy, axis=0) / 64
        xy += rng.uniform(size=xy.shape) / 32
        cs = np.concatenate(cs_c, axis=0).reshape(-1, 1)
        df = np.concatenate((xy, cs), axis=1)
        rng.shuffle(df)

        joblib.dump(df, 'df.joblib')


if __name__ == '__main__':
    main()
