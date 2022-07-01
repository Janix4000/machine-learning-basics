# %%
import numpy as np
from agent_log import create_episodes_moves_mean, create_episodes_score_mean, create_heat_map, create_moves_win, create_res_counter
from game import GameState
import game
from policy import Policy
import matplotlib.pyplot as plt


class SarsaPolicy(Policy):
    def __init__(self, board, n_max_moves, lr=0.5, df=0.95, er=0.05, mvs=None):
        self.mvs = mvs or ['l', 'r']

        self.lr = lr
        self.df = df
        self.er = er

        self.q = np.random.rand(len(board), len(board[0]), len(self.mvs))

        self.n_max_moves = n_max_moves
        self.n_moves = 0

    def choose_action(self, row, col):
        t = self.q[row, col]
        res = np.argmax(t)
        if np.random.rand() < self.er:
            new_res = res
            while new_res == res:
                new_res = np.random.randint(len(t))
            res = new_res
        return res

    def score_state(self, game_state):
        score = -self.n_moves
        if game_state.state == 'dead':
            score -= self.n_max_moves**2
        elif game_state.state == 'win':
            score += self.n_max_moves**2
        elif self.n_moves == self.n_max_moves - 1:
            score -= self.n_max_moves**2
        return score

    def next_state_score(self, game_state: GameState):
        pos = game_state.pos
        action = self.choose_action(pos[0], pos[1])
        mv = self.mvs[action]

        new_game_state = game.next_game_move(game_state, mv)
        self.n_moves += 1
        score = self.score_state(new_game_state)

        next_action = self.choose_action(
            new_game_state.pos[0], new_game_state.pos[1])
        next_score = self.q[new_game_state.pos[0],
                            new_game_state.pos[1], next_action]

        self.q[pos[0], pos[1], action] = (
            1 - self.lr) * self.q[pos[0], pos[1], action] + self.lr * (score + self.df * next_score)

        return new_game_state, score

    def init_agent(self, game_state: GameState) -> None:
        self.n_moves = 0


# %%

# if __name__ == '__main__':


# %%
if __name__ == '__main__':

    policy = SarsaPolicy()

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