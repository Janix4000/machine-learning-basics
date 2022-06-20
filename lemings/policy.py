from abc import ABC, abstractmethod
from agent_log import AgentLog
from game import GameState

import numpy as np
import game


class Policy(ABC):
    @abstractmethod
    def next_state_score(self, game_state: GameState) -> tuple[str, float]:
        pass

    @abstractmethod
    def init_agent(game_state: GameState) -> None:
        pass

    def run_policy(self, board: game.Board, n_agents: int, n_max_moves: int):
        agents_res = []
        for _ in range(n_agents):
            game_state = game.GameState()
            game_state.board = board
            n_moves = 0
            agent = AgentLog()
            self.init_agent(game_state)
            while game_state.state == 'living' and n_moves < n_max_moves:
                game_state, score = self.next_state_score(game_state)
                agent.path.append(game_state.pos)
                agent.scores.append(score)
                n_moves += 1

            if game_state.state == 'living':
                agent.res = 'bored'
            else:
                agent.res = game_state.state

            agents_res.append(agent)

        return np.array(agents_res)
