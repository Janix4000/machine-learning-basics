from matplotlib import pyplot as plt
import numpy as np
from agent_log import AgentLog, create_episodes_moves_mean, create_episodes_score_mean, create_heat_map, create_moves_win, create_res_counter


def show_results(agents_res: list[AgentLog], board):
    n_axs = 5
    fig, axs = plt.subplots(n_axs, 1, figsize=(4, 4 * n_axs))

    places_heat_map = create_heat_map(agents_res, board)
    axs[0].imshow(places_heat_map, cmap='hot', interpolation='nearest')

    res, counter = create_res_counter(agents_res)
    axs[1].pie(counter, labels=res)

    moves, saved = create_moves_win(agents_res)
    axs[2].scatter(moves, saved)
    axs[2].plot(moves, saved, '--')

    n_moves, cum_mean = create_episodes_moves_mean(agents_res)
    axs[3].plot(n_moves, cum_mean)

    n_moves, cum_mean = create_episodes_score_mean(agents_res)
    axs[4].plot(n_moves, cum_mean)

    plt.show()


def show_paths(agents_res, board, n=20):
    agents = agents_res[-n:]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    places_heat_map = create_heat_map(agents, board)
    ax.imshow(places_heat_map, cmap='hot', interpolation='nearest')

    ax.set_title(
        f'Heatmap of the most frequent visited places on the board\nLast {n} lemings.')

    return fig, ax


def show_res_stats(agents_res):
    res, counter = create_res_counter(agents_res)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.pie(counter, labels=res, autopct='%1.1f%%')
    ax.set_facecolor((1.0, 1.0, 1.0))
    ax.set_title(f'Outcomes percentage')

    return fig, ax


def show_moves_wins(agentss: list[np.ndarray], n_max_moves=15):
    movess, saveds = zip(*[create_moves_win(agents_res, n_max_moves=n_max_moves)
                         for agents_res in agentss])

    saveds = np.stack(saveds)
    saved_mean = np.mean(saveds, axis=0)
    saved_std = np.std(saveds, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.bar(movess[0], saved_mean, yerr=saved_std)
    ax.set_title(
        f'Number of wins for number of steps\nFor {agentss.shape[0]} experiments')
    ax.set_xlabel(f'Number of steps in episode')
    ax.set_ylabel(f'Number of saved agents')

    return fig, ax


def show_moves_wins(agentss: list[np.ndarray], n_max_moves=15):
    movess, saveds = zip(*[create_moves_win(agents_res, n_max_moves=n_max_moves)
                         for agents_res in agentss])

    saveds = np.stack(saveds)
    saved_mean = np.mean(saveds, axis=0)
    saved_std = np.std(saveds, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.bar(movess[0], saved_mean, yerr=saved_std)
    ax.set_title(
        f'Number of wins for number of actions\nFor {agentss.shape[0]} experiments')
    ax.set_xlabel(f'Number of actions in episode')
    ax.set_ylabel(f'Number of saved lemings')

    return fig, ax


def show_moves_mean(agentss: list[np.ndarray], n=20):
    movess, moves_mean_roll = zip(*[create_episodes_moves_mean(agents_res, n=n)
                                    for agents_res in agentss])

    moves_mean_roll = np.stack(moves_mean_roll)

    saved_mean = np.mean(moves_mean_roll, axis=0)
    saved_std = np.std(moves_mean_roll, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.fill_between(movess[0], saved_mean + saved_std,
                    saved_mean - saved_std, alpha=0.3)
    ax.plot(movess[0], saved_mean, linewidth=3)
    ax.set_title(
        f'Rolling mean of number of actions for episodes\nFor {agentss.shape[0]} experiments\nRolling={n}')
    ax.set_xlabel(f'Number of episode')
    ax.set_ylabel(f'Average number of actions')

    return fig, ax


def show_score_mean(agentss: list[np.ndarray], n=20):
    movess, moves_mean_roll = zip(*[create_episodes_score_mean(agents_res, n=n)
                                    for agents_res in agentss])

    moves_mean_roll = np.stack(moves_mean_roll)

    saved_mean = np.mean(moves_mean_roll, axis=0)
    saved_std = np.std(moves_mean_roll, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.fill_between(movess[0], saved_mean + saved_std,
                    saved_mean - saved_std, alpha=0.3)
    ax.plot(movess[0], saved_mean, linewidth=3)
    ax.set_title(
        f'Rolling mean of score for episodes\nFor {agentss.shape[0]} experiments\nRolling={n}')
    ax.set_xlabel(f'Number of episode')
    ax.set_ylabel(f'Average score')

    return fig, ax
