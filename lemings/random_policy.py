# %%
import numpy as np
from agent_log import create_episodes_moves_mean, create_episodes_score_mean, create_heat_map, create_moves_win, create_res_counter
from game import GameState
import game
from policy import Policy
from random import choice
import matplotlib.pyplot as plt
from collections import Counter


class RandomPolicy(Policy):
    def next_state_score(self, game_state: GameState):
        mv = choice(['l', 'r'])
        return game.next_game_move(game_state, mv), 0

    def init_agent(game_state: GameState) -> None:
        pass


# %%

# if __name__ == '__main__':


# %%
if __name__ == '__main__':

    policy = RandomPolicy()

    board = np.array([
        '##########',
        '#........#',
        '#........#',
        '#..###...#',
        '#***..4..#',
        '##########',
    ])

    agents_res = policy.run_policy(board, 500, 15)

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

# %%
