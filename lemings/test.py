# %%

from random_policy import RandomPolicy
from result_plot import *
import numpy as np

from sarsa_policy import SarsaPolicy
# %%
board = np.array([
    '##########',
    '#........#',
    '#........#',
    '#..###...#',
    '#***..4..#',
    '##########',
])


# %%
policies = [SarsaPolicy(board, 15) for _ in range(10)]
agentss = [policy.run_policy(board, 500, 15)
           for policy in policies]
agentss = np.array(agentss)

# %%
fig, ax = show_paths(agentss[0], board, n=20)
# %%
fig, ax = show_res_stats(agentss[0])
# %%
fig, axs = show_moves_wins(agentss, n_max_moves=15)
# %%
fig, ax = show_moves_mean(agentss)

# %%
fig, ax = show_score_mean(agentss)

# %%
