

from collections import Counter
import numpy as np


class AgentLog:
    def __init__(self):
        self.path: list = []
        self.res: str = None
        self.scores: list[float] = []


def create_heat_map(agents: list[AgentLog], board) -> np.ndarray:
    places_heat_map = np.zeros((len(board), len(board[0])), dtype=int)
    for agent in agents:
        for (row, col) in agent.path:
            places_heat_map[row, col] += 1

    return places_heat_map


def create_moves_win(agents: list[AgentLog], n_max_moves=15):
    win_n_moves = Counter(len(agent.path)
                          for agent in agents if agent.res == 'win')
    moves, wins = np.array(list(win_n_moves.keys())), np.array(
        list(win_n_moves.values()))
    max_moves = np.max(moves)
    wins_zeros = np.zeros(shape=(n_max_moves + 1,))
    wins_zeros[moves] = wins
    return np.arange(0, n_max_moves + 1), wins_zeros


def create_res_counter(agents: list[AgentLog]):
    counter = Counter(agent.res for agent in agents)
    return np.array(list(counter.keys())), np.array(list(counter.values()))


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def create_episodes_moves_mean(agents: list[AgentLog], n=20):
    moves = [len(agent.path) for agent in agents]
    move_mean = moving_average(moves, n=n)
    n_moves = np.arange(n, move_mean.shape[0] + n)
    return n_moves, move_mean


def create_episodes_score_mean(agents: list[AgentLog], n=20):
    moves = [np.mean(agent.scores) for agent in agents]
    move_mean = moving_average(moves, n=n)
    n_moves = np.arange(n, move_mean.shape[0] + n)
    return n_moves, move_mean
