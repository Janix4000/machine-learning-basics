

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


def create_moves_win(agents: list[AgentLog]):
    win_n_moves = Counter(len(agent.path)
                          for agent in agents if agent.res == 'win')
    moves, wins = np.array(list(win_n_moves.keys())), np.array(
        list(win_n_moves.values()))
    max_moves = np.max(moves)
    wins_zeros = np.zeros(shape=(max_moves + 1,))
    wins_zeros[moves] = wins
    return np.arange(0, max_moves + 1), wins_zeros


def create_res_counter(agents: list[AgentLog]):
    counter = Counter(agent.res for agent in agents)
    return np.array(list(counter.keys())), np.array(list(counter.values()))
