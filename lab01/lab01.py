import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.ticker as ticker

rng = np.random.RandomState(123)


def unit_vector(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def get_angle(a1, a2, b1, b2):
    v1 = unit_vector(a1 - a2)
    v2 = unit_vector(b1 - b2)
    dot = np.sum(v1 * v2, axis=1)
    return np.arccos(dot)


def generat_angles(dim, n):
    a1, a2, b1, b2 = rng.uniform(-1, 1, size=(4, n, dim))
    return get_angle(a1, a2, b1, b2) / np.pi * 2


def generate_radius_distances(dim, n):
    pts = rng.uniform(-1, 1, size=(2 * n, dim))
    res = []
    for i in range(n):
        a = pts[2 * i]
        b = pts[2 * i + 1]
        d = np.linalg.norm(a - b, ord=2)
        ds = np.linalg.norm(pts - a, axis=1, ord=2)
        count = (np.sum(ds < d) - 1) / (2 * n)
        res.append(count)
    return np.array(res)


def generate_triples(dim, n):
    a, b, c = rng.uniform(-1, 1, size=(3, n, dim))
    d_ab = np.linalg.norm(a - b, axis=1)
    d_ac = np.linalg.norm(a - c, axis=1)
    res = np.abs(d_ab - d_ac) / (d_ab + d_ac) * 2
    return res


def create_hist(xss: list, title: str, format: str = '{}', x_label: str = 'values'):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
    axs = axs.ravel()
    stacked_ax = axs[-1]
    # x_format = r'%.2f$\cdot\frac{\pi}{2}$'

    stacked_ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(format))
    cs = np.eye(3, 3)
    for ax, c, xs in zip(axs[:3], cs, xss):
        mu = np.mean(xs)
        sigma = np.std(xs)
        label = fr'$\mu=$' + \
            format.format(x=mu) + '\n' + fr'$\sigma=$' + format.format(x=sigma)
        # ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(format))
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter(format))
        ax.hist(xs, density=True, fc=c, label=label)
        stacked_ax.hist(xs, density=True, fc=(*c, 0.33))
        ax.set_xlabel(x_label)
        ax.set_ylabel("Probability density")
        ax.legend()

    fig.suptitle(title)


def main():
    xss = [generat_angles(dim, int(1e5)) for dim in (10, 100, 1000)]
    title = 'Distributions of the angles between random vectors in the hypercubes.'
    format = r'{x:.2f}$\cdot\frac{{\pi}}{{2}}$'
    create_hist(xss, title=title, format=format, x_label='radians')
    # plt.show()

    xss = [generate_radius_distances(dim, int(1e4)) for dim in (10, 100, 1000)]
    title = 'Distributions of the percanteges of the points from the hypercube inside the radius of the hyperspheres.'
    format = r'{x*100:.2f}%'
    create_hist(xss, title=title, format=format, x_label='percentage')
    plt.show()

    xss = [generate_triples(dim, int(1e5)) for dim in (10, 100, 1000)]
    title = 'Distributions of the proportions between difference and mean distances between triples of points in the hypercubes.'
    format = r'{x:.2f}'
    create_hist(xss, title=title, format=format, x_label='radians')
    plt.show()

    # fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
    # axs = axs.ravel()
    # stacked_ax = axs[-1]
    # x_format = r'%.1f$\cdot\frac{\pi}{2}$'
    # stacked_ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(x_format))
    # xss = [angles_10, angles_100, angles_1000]
    # cs = np.eye(3, 3)
    # for ax, c, xs in zip(axs[:3], cs, xss):
    #     mu = np.mean(xs)
    #     sigma = np.std(xs)
    #     title = fr'$\mu={mu:.2f}\cdot\frac{{\pi}}{{2}}$' + \
    #         '\n' + fr'$\sigma={sigma:.2f}\cdot\frac{{\pi}}{{2}}$'
    #     ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(x_format))
    #     ax.hist(xs, density=True, fc=c, label=title)
    #     stacked_ax.hist(xs, density=True, fc=(*c, 0.33))
    #     ax.set_xlabel("Radians")
    #     ax.set_ylabel("Probability density")
    #     ax.legend()

    # fig.suptitle(
    #     'Distributions of the angles between random vectors in the hypercubes.')
    # # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()
