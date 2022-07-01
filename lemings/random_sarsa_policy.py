# %%
import numpy as np
from agent_log import create_episodes_moves_mean, create_episodes_score_mean, create_heat_map, create_moves_win, create_res_counter
from game import GameState
import game
import matplotlib.pyplot as plt

from sarsa_policy import SarsaPolicy


class RandomSarsaPolicy(SarsaPolicy):

    def __init__(self, board, n_max_moves, lr=0.5, df=0.95, er=0.05, mvs=None):
        super().__init__(board, n_max_moves, lr, df, er, mvs)

    def next_state_score(self, game_state: GameState):
        pos = game_state.pos
        action = self.choose_action(pos[0], pos[1])
        mv = self.mvs[action]

        new_game_state = game.next_game_move(game_state, mv, rand=1)
        self.n_moves += 1
        score = self.score_state(new_game_state)

        next_score = np.max(self.q[new_game_state.pos[0],
                            new_game_state.pos[1]])

        self.q[pos[0], pos[1], action] = (
            1 - self.lr) * self.q[pos[0], pos[1], action] + self.lr * (score + self.df * next_score)

        return new_game_state, score


# %%

# if __name__ == '__main__':


# %%
if __name__ == '__main__':

    policy = RandomSarsaPolicy()

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
