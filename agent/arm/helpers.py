import numpy as np

def compute_danger_map(bombs, b_strength, b_life, walls, crates):
    size = len(bombs)
    high = np.zeros((size, size))
    med = np.zeros((size, size))
    low = np.zeros((size, size))
    obstacles = np.logical_or(walls, crates)

    for row_idx, row in enumerate(bombs):
        for col_idx, entry in enumerate(row):
            if entry != 1:
                continue
            life = b_life[row_idx][col_idx]
            strength = b_strength[row_idx][col_idx]
            exp = np.zeros((size, size))
            left = range(row_idx, max(-1, row_idx - int(strength)), -1)
            right = range(row_idx, min(size, row_idx + int(strength)))
            down = range(col_idx, max(-1, col_idx - int(strength)), -1)
            up = range(col_idx, min(size, col_idx + int(strength)))
            for idx in left:
                if obstacles[idx][col_idx]:
                    break
                exp[idx][col_idx] = 1
            for idx in right:
                if obstacles[idx][col_idx]:
                    break
                exp[idx][col_idx] = 1
            for idx in down:
                if obstacles[row_idx][idx]:
                    break
                exp[row_idx][idx] = 1
            for idx in up:
                if obstacles[row_idx][idx]:
                    break
                exp[row_idx][idx] = 1
            if life < 3:
                high = np.logical_or(high, exp)
            elif life < 6:
                med = np.logical_or(med, exp)
            else:
                low = np.logical_or(low, exp)
    return high.astype(int), med.astype(int), low.astype(int)


def featurize_obs(obs, save_board):
    board = np.array(obs['board'])
    player_pos = obs['position']
    teammate = obs['teammate'].value

    board = np.where(save_board == 1, np.ones((11, 11)), board)
    walls = np.where(board == 1, np.ones((11, 11)), np.zeros((11, 11)))
    save_board = np.logical_or(save_board, walls)

    large_board = np.full((21, 21), 1)
    large_board[5:5+board.shape[0], 5:5+board.shape[1]] = board

    b_strength = obs['bomb_blast_strength']
    b_life = obs['bomb_life']

    large_b_strength = np.full((21, 21), 0)
    large_b_strength[5:5+b_strength.shape[0],
                     5:5+b_strength.shape[1]] = b_strength

    large_b_life = np.full((21, 21), 0)
    large_b_life[5:5+b_life.shape[0], 5:5+b_life.shape[1]] = b_life

    player_x, player_y = player_pos
    board = large_board[player_x+1:player_x+10, player_y+1:player_y+10]
    b_strength = large_b_strength[player_x +
                                  1:player_x+10, player_y+1:player_y+10]
    b_life = large_b_life[player_x+1:player_x+10, player_y+1:player_y+10]

    teammate_board = np.where(
        board == teammate, np.ones((9, 9)), np.zeros((9, 9)))
    board = np.where(board == teammate, np.zeros((9, 9)), board)
    enemies = np.where(board >= 10, np.ones((9, 9)), np.zeros((9, 9)))
    walls = np.where(board == 1, np.ones((9, 9)), np.zeros((9, 9)))
    crates = np.where(board == 2, np.ones((9, 9)), np.zeros((9, 9)))
    flames = np.where(board == 4, np.ones((9, 9)), np.zeros((9, 9)))

    bombs = np.where(b_life != 0, np.ones((9, 9)), np.zeros((9, 9)))

    high, med, low = compute_danger_map(
        bombs, b_strength, b_life, walls, crates)

    images = [teammate_board, enemies, walls,
              crates, flames, high, med, low]  # fog, extra_bomb, extra_range, kick]

    scalar = [obs['ammo'], obs['blast_strength'], int(obs['can_kick'])]

    return images, scalar, save_board