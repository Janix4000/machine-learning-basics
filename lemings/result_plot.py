from matplotlib import pyplot as plt
import numpy as np
from agent_log import AgentLog, create_episodes_moves_mean, create_episodes_score_mean, create_heat_map, create_moves_win, create_res_counter


from sklearn.preprocessing import OrdinalEncoder as OE
import seaborn as sns


def show_results(agentss: list[AgentLog], board, n_max_moves):
    fig, axs = plt.subplots(3, 2, figsize=(12, 18))
    axs = axs.ravel()
    show_board(board, ax=axs[0])
    show_paths(agentss, board, n=20, ax=axs[1])
    show_res_stats(agentss, ax=axs[2])
    show_moves_wins(agentss, n_max_moves=n_max_moves, ax=axs[3])
    show_moves_mean(agentss, ax=axs[4])
    show_score_mean(agentss, ax=axs[5])

    return fig, axs


def show_board(board, ax=None):
    oe = OE().fit(np.array(list('.#*0123456789')).reshape(-1, 1))
    b = [oe.transform(np.array(list(r)).reshape(-1, 1)) for r in board]
    b = np.array(b).reshape(len(b), len(b[0]))
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = None
    labels = [list(r) for r in board]

    ax = sns.heatmap(b, cmap="inferno", annot=labels,
                     annot_kws={'fontsize': 16}, fmt='s', ax=ax)
    return fig, ax


def show_paths(agentss, board, n=20, ax=None):
    agentss = agentss[:, -n:]
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = None
    places_heat_map = np.sum(create_heat_map(agents, board)
                             for agents in agentss)
    ax.imshow(places_heat_map, cmap='hot', interpolation='nearest')

    ax.set_title(
        f'Heatmap of the most frequent visited places on the board\nLast {n} lemings.')

    return fig, ax


def show_res_stats(agentss, n=20, ax=None):
    agentss = agentss[:, -n:]
    ress, counters = zip(*(create_res_counter(agents) for agents in agentss))
    res = ress[0]
    counter = np.mean(np.stack(counters), axis=0)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = None
    ax.pie(counter, labels=res, autopct='%1.1f%%')
    ax.set_facecolor((1.0, 1.0, 1.0))
    ax.set_title(f'Outcomes percentage\nLast {n} lemmings.')

    return fig, ax


def show_moves_wins(agentss: list[np.ndarray], n_max_moves=15, ax=None):
    movess, saveds = zip(*[create_moves_win(agents_res, n_max_moves=n_max_moves)
                         for agents_res in agentss])

    saveds = np.stack(saveds)
    saved_mean = np.mean(saveds, axis=0)
    saved_std = np.std(saveds, axis=0)

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = None

    ax.bar(movess[0], saved_mean, yerr=saved_std)
    ax.set_title(
        f'Number of wins for number of steps\nFor {agentss.shape[0]} experiments')
    ax.set_xlabel(f'Number of steps in episode')
    ax.set_ylabel(f'Number of saved agents')

    return fig, ax


def show_moves_wins(agentss: list[np.ndarray], n_max_moves=15, ax=None):
    movess, saveds = zip(*[create_moves_win(agents_res, n_max_moves=n_max_moves)
                         for agents_res in agentss])

    saveds = np.stack(saveds)
    saved_mean = np.mean(saveds, axis=0)
    saved_std = np.std(saveds, axis=0)

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = None

    ax.bar(movess[0], saved_mean, yerr=saved_std)
    ax.set_title(
        f'Number of wins for number of actions\nFor {agentss.shape[0]} experiments')
    ax.set_xlabel(f'Number of actions in episode')
    ax.set_ylabel(f'Number of saved lemings')

    return fig, ax


def show_moves_mean(agentss: list[np.ndarray], n=20, ax=None):
    movess, moves_mean_roll = zip(*[create_episodes_moves_mean(agents_res, n=n)
                                    for agents_res in agentss])

    moves_mean_roll = np.stack(moves_mean_roll)

    saved_mean = np.mean(moves_mean_roll, axis=0)
    saved_std = np.std(moves_mean_roll, axis=0)

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = None

    ax.fill_between(movess[0], saved_mean + saved_std,
                    saved_mean - saved_std, alpha=0.3)
    ax.plot(movess[0], saved_mean, linewidth=3)
    ax.set_title(
        f'Rolling mean of number of actions for episodes\nFor {agentss.shape[0]} experiments\nRolling={n}')
    ax.set_xlabel(f'Number of episode')
    ax.set_ylabel(f'Average number of actions')

    return fig, ax


def show_score_mean(agentss: list[np.ndarray], n=20, ax=None):
    movess, moves_mean_roll = zip(*[create_episodes_score_mean(agents_res, n=n)
                                    for agents_res in agentss])

    moves_mean_roll = np.stack(moves_mean_roll)

    saved_mean = np.mean(moves_mean_roll, axis=0)
    saved_std = np.std(moves_mean_roll, axis=0)

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = None

    ax.fill_between(movess[0], saved_mean + saved_std,
                    saved_mean - saved_std, alpha=0.3)
    ax.plot(movess[0], saved_mean, linewidth=3)
    ax.set_title(
        f'Rolling mean of score for episodes\nFor {agentss.shape[0]} experiments\nRolling={n}')
    ax.set_xlabel(f'Number of episode')
    ax.set_ylabel(f'Average score')

    return fig, ax
