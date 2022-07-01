import numpy as np

Board = list[str]
Position = np.ndarray


class GameState:
    def __init__(self):
        self.state = 'living'
        self.pos = np.array([1, 1])
        self.board = None


next_move = {
    's': [+0, +0],
    'l': [+0, -1],
    'r': [+0, +1],
    'u': [-1, +0],
    'd': [+1, +0],
    'ld': [+1, +1],
    'rd': [-1, +1],
    'p': [+0, +0],
}


def next_game_move(game_state: GameState, move: str, rand=0):
    board = game_state.board
    state = move
    final_pos = game_state.pos
    fall = 1
    next_game_state = GameState()

    while state != 's':
        next_pos = final_pos + next_move[state]
        next_obstacle = board[next_pos[0]][next_pos[1]]
        state = 's'
        if next_obstacle == '#':
            state = 's'
        else:
            final_pos = next_pos
            if next_obstacle.isdigit():
                state = 'u'
                shift = max(int(next_obstacle) +
                            np.random.randint(-rand, rand + 1), 0)
                fall -= shift
            elif next_obstacle == '*':
                state = 's'
                next_game_state.state = 'dead'
            else:
                state = 's'
        if next_game_state.state != 'dead' and fall != 0:
            state = 'd' if fall > 0 else 'u'
            fall -= abs(fall) // fall

    next_game_state.pos = final_pos
    next_game_state.board = board

    if next_game_state.state != 'dead' and len(board) - 2 == final_pos[0] and len(board[0]) - 2 == final_pos[1]:
        next_game_state.state = 'win'
    return next_game_state


if __name__ == '__main__':
    board = np.array([
        '##########',
        '#........#',
        '#........#',
        '#..###...#',
        '#***..4..#',
        '##########',
    ])

    board = np.array([
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
    ])
    board = np.array([
        '##############',
        '#............#',
        '########.#####',
        '#............#',
        '#..######.####',
        '#*........*..#',
        '#..######.####',
        '#.........*..#',
        '#*.######.####',
        '#.........*..#',
        '#*.######.####',
        '#.........*..#',
        '#.############',
        '#............#',
        '##############',
    ])

    game_state = GameState()
    game_state.board = board

    def print_board(board, pos):
        for idx, row in enumerate(board):
            if idx == pos[0]:
                print(row[:pos[1]], end='')
                print('@', end='')
                print(row[pos[1]+1:])
            else:
                print(row)

    print_board(board, game_state.pos)
    while game_state.state == 'living':
        mv = input('Move: ')
        game_state = next_game_move(game_state, mv)
        print_board(board, game_state.pos)
    print(game_state.state)
