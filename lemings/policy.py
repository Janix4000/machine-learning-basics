from abc import ABC, abstractmethod
from agent_log import AgentLog
from game import GameState

import numpy as np
import game


class Policy(ABC):
    @abstractmethod
    def next_move(self, game_state: GameState) -> tuple[str, float]:
        pass

    @abstractmethod
    def end_agent(game_state: GameState) -> None:
        pass

    def run_policy(self, board: game.Board, n_agents: int, n_max_moves: int):
        agents_res = []
        for _ in range(n_agents):
            game_state = game.GameState()
            n_moves = 0
            agent = AgentLog()
            while game_state.state == 'living' and n_moves < n_max_moves:
                mv, score = self.next_move(game_state)
                game_state = game.next_game_move(game_state, board, mv)
                agent.path.append(game_state.pos)
                agent.scores.append(score)
                n_moves += 1

            if game_state.state == 'living':
                agent.res = 'bored'
            else:
                agent.res = game_state.state

            agents_res.append(agent)

        return agents_res
