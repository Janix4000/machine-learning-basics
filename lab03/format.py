from PIL import Image
import numpy as np
import joblib
import matplotlib.pyplot as plt


def main():
    with Image.open('dataimg.png') as img:
        img = np.array(img)
        a_xy = np.where(img == 1)
        a_xy = np.c_[a_xy]
        a_c = np.zeros(shape=a_xy.shape[0])

        b_xy = np.where(img == 2)
        b_xy = np.c_[b_xy]
        b_c = np.ones(shape=b_xy.shape[0])

        rng = np.random.RandomState(2)

        xy = np.concatenate((a_xy, b_xy), axis=0) / 100
        xy += rng.uniform(size=xy.shape) / 30
        cs = np.concatenate((a_c, b_c), axis=0).reshape(-1, 1)
        df = np.concatenate((xy, cs), axis=1)
        rng.shuffle(df)

        joblib.dump(df, 'df.joblib')


if __name__ == '__main__':
    main()
