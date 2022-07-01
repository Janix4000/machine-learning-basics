# %%

from more_moves_sarsa_policy import MoreMovesSarsaPolicy
from q_policy import Q_Policy
from random_policy import RandomPolicy
from random_sarsa_policy import RandomSarsaPolicy
from result_plot import *
import numpy as np
from sarsa_policy import SarsaPolicy


# %%
name = 'Random'
n_max_moves = 40
n_agents = 200

policy = {'Sarsa': SarsaPolicy, 'Q': Q_Policy, 'RandomSarsa': RandomSarsaPolicy,
          'MoreMoves': MoreMovesSarsaPolicy, 'Random': RandomPolicy}[name]


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

# %%

for board_name, board in boards.items():
    policies = [policy(board, n_max_moves) for _ in range(10)]
    agentss = [policy.run_policy(board, n_agents, n_max_moves)
               for policy in policies]
    agentss = np.array(agentss)

    title = f'res/{name}_{board_name}_{n_max_moves}_{n_agents}'
    fig, ax = show_board(board)
    fig.savefig(f'res/{board_name}_board.png', dpi=300)
    fig, axs = show_results(agentss, board, n_max_moves=n_max_moves)
    fig.savefig(f'{title}')

# %%
