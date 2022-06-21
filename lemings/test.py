# %%

from q_policy import Q_Policy
from random_policy import RandomPolicy
from result_plot import *
import numpy as np
from sarsa_policy import SarsaPolicy


# %%
name = 'Q'
map = 'hard'
n_max_moves = 40
n_agents = 200

boards = {
    'simple': np.array([
        '##########',
        '#........#',
        '#........#',
        '#..###...#',
        '#***..4..#',
        '##########',
    ]),
    'example': np.array([
        '##############',
        '#............#',
        '#.......###..#',
        '####.....*...#',
        '#...4447*..**#',
        '#.........####',
        '#***.........#',
        '#######.4*...#',
        '#............#',
        '#******......#',
        '##############',
    ]),
    'multi': np.array([
        '##############',
        '#............#',
        '######.#######',
        '#............#',
        '#..#####..####',
        '#*.......*...#',
        '#..#####..####',
        '#*.......*...#',
        '#..#####..####',
        '#*.......*...#',
        '#..#####..####',
        '#............#',
        '#.############',
        '#............#',
        '##############',
    ]),
    'jump': np.array([
        '##############',
        '#..*.........#',
        '##.*****.###.#',
        '#.......3....#',
        '#..########..#',
        '#.......#*...#',
        '######..#*...#',
        '#......#*....#',
        '#..#####.....#',
        '#............#',
        '##############',
    ]),
    'hard': np.array([
        '##############',
        '#............#',
        '####..######.#',
        '#....3.......#',
        '#..#####.##..#',
        '#......#..*..#',
        '######..#.*4.#',
        '#......#*...*#',
        '#..#####*....#',
        '#............#',
        '##############',
    ])
}

board = boards[map]

# %%
policy = {'Sarsa': SarsaPolicy, 'Q': Q_Policy}[name]
policies = [policy(board, n_max_moves) for _ in range(10)]
agentss = [policy.run_policy(board, n_agents, n_max_moves)
           for policy in policies]
agentss = np.array(agentss)

title = f'res/{name}_{map}_{n_max_moves}_{n_agents}'
# %%
fig, ax = show_board(board)
fig.savefig(f'res/{map}_board.png', dpi=300)
# %%
fig, axs = show_results(agentss, board, n_max_moves=n_max_moves)
fig.savefig(f'{title}')

# %%
