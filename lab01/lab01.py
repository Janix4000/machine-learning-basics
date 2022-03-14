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
    return get_angle(a1, a2, b1, b2)

def main():
    angles_10 = generat_angles(10, n=10000) / np.pi
    angles_100 = generat_angles(100, n=10000) / np.pi
    angles_1000 = generat_angles(1000, n=10000) / np.pi
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
    axs = axs.ravel()
    stacked_ax = axs[-1]
    stacked_ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f$\pi$'))
    xss = [angles_10, angles_100, angles_1000]
    cs = np.eye(3, 3)
    for ax, c, xs in zip(axs[:3], cs, xss):
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f$\pi$'))
        ax.hist(xs, density=True, fc=c)
        stacked_ax.hist(xs, density=True, fc=(*c, 0.33))
        
        
    
    plt.show()
    

if __name__ == '__main__':
    main()