import numpy as np


def featurize_obs(obs):
    # grab observations
    board = np.array(obs['board'])
    player_pos = obs['position']
    teammate = obs['teammate'].value
    b_life = obs['bomb_life']
    b_strength = obs['bomb_blast_strength']
    b_movement = obs['bomb_moving_direction']
    flames = obs['flame_life']

    board = np.pad(board, 1, 'constant', constant_values=1)
    b_life = np.pad(b_life, 1, 'constant', constant_values=0)
    b_strength = np.pad(b_strength, 1, 'constant', constant_values=0)
    b_movement = np.pad(b_movement, 1, 'constant', constant_values=0)
    flames = np.pad(flames, 1, 'constant', constant_values=0)
    player_pos = tuple(np.array(player_pos) + 1)

    # remove player
    board[player_pos] = 0

    teammate_board = (board == teammate).astype(int)
    board[np.nonzero(teammate_board)] = 0
    p_board = np.zeros_like(board)
    p_board[player_pos] = 1
    enemies = (board >= 10).astype(int)
    walls = (board == 1).astype(int)
    crates = (board == 2).astype(int)
    flames = flames / 3

    # expand bomb explosions using b_strength and b_life
    explosions = compute_explosions(walls, crates, b_life, b_strength)
    # one hot encode bomb movement
    b_movement = [(b_movement == action).astype(int) for action in range(1, 5)]

    # b_strength = [(b_strength == tick).astype(int) for tick in range(2, 10)]
    extra_bomb = (board == 6).astype(int)
    extra_range = (board == 7).astype(int)
    kick = (board == 8).astype(int)

    images = np.array([p_board,
                       teammate_board,
                       walls,
                       crates,
                       flames,
                       enemies,
                       explosions,
                       *b_movement,
                       extra_bomb,
                       extra_range,
                       kick])

    scalar = [obs['ammo'], obs['blast_strength'], int(obs['can_kick'])]
    scalar += [0] * 8

    scalar = np.array(scalar)

    return images, scalar


def compute_explosions(walls, crates, b_life, b_strength):
    # invert bomb life: convert tick range from 10 - 1 (at 1 bomb explodes next step)
    # to 0.1 - 1 (at 1 bomb explodes)
    b_life = (11 * (b_life > 0).astype(int) - b_life) / 10
    base_exp = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    explosions = np.zeros_like(b_strength)
    bomb_coords = np.array(np.nonzero(b_strength)).T
    if not bomb_coords.size:
        return explosions
    life_arr = [b_life[tuple(coords)] for coords in bomb_coords]
    bomb_coords = bomb_coords[np.argsort(life_arr)[::-1]]
    for bomb_x, bomb_y in bomb_coords:
        strength = int(b_strength[(bomb_x, bomb_y)]) - 1
        # if bomb in radius of already exploded bomb, set
        # bomb life to life of exploded bomb
        life = max(b_life[(bomb_x, bomb_y)], explosions[(bomb_x, bomb_y)])
        max_idx = b_strength.shape[-2] - 1
        # explosion radius
        exp_x_min = max(-bomb_x + strength, 0)
        exp_x_max = min(max_idx - bomb_x + strength + 1, strength * 2 + 1)
        exp_y_min = max(-bomb_y + strength, 0)
        exp_y_max = min(max_idx - bomb_y + strength + 1, strength * 2 + 1)
        # explosion radius on map
        bomb_x1 = max(bomb_x - strength, 0)
        bomb_x2 = min(bomb_x + strength, max_idx) + 1
        bomb_y1 = max(bomb_y - strength, 0)
        bomb_y2 = min(bomb_y + strength, max_idx) + 1
        explosion = np.zeros_like(b_life, dtype=float)
        # expand explosion
        bomb_exp = np.pad(base_exp, (strength + 1) - 2, 'edge')[exp_x_min: exp_x_max,
                                                                exp_y_min: exp_y_max]
        # add explosion to empty map at coordinates
        explosion[bomb_x1:bomb_x2, bomb_y1:bomb_y2] = bomb_exp * life
        # compute binary wall map starting from bomb position
        exp_walls = np.zeros_like(walls)
        exp_walls[bomb_x::-1] += np.cumsum(walls[bomb_x::-1], axis=0)
        exp_walls[bomb_x:] += np.cumsum(walls[bomb_x:], axis=0)
        exp_walls[:, bomb_y::-1] += np.cumsum(walls[:, bomb_y::-1], axis=1)
        exp_walls[:, bomb_y:] += np.cumsum(walls[:, bomb_y:], axis=1)
        exp_walls = np.clip(exp_walls, 0, 1)
        # compute binary crate map starting from bomb position
        # crates closes to bomb have to disregarded, therefore
        # shift -1 and +1 in every direction
        exp_crates = np.zeros_like(crates)
        if bomb_x > 1:
            exp_crates[bomb_x-1::-1] += np.cumsum(
                crates[bomb_x:0:-1], axis=0)
        exp_crates[bomb_x+1:] += np.cumsum(
            crates[bomb_x:max_idx], axis=0)
        if bomb_y > 1:
            exp_crates[:, bomb_y-1::-1] += np.cumsum(
                crates[:, bomb_y:0:-1], axis=1)
        exp_crates[:, bomb_y+1:] += np.cumsum(
            crates[:, bomb_y:max_idx], axis=1)
        exp_crates = np.clip(exp_crates, 0, 1)
        # combine walls and crates
        obstacles = np.logical_or(exp_walls, exp_crates).astype(int)
        # confine explosion by obstacles
        explosion *= np.logical_not(obstacles).astype(int)
        # add explosion to explosion map
        explosions = np.maximum(explosions, explosion)
    return explosions


def center_boards(board, view_distance, player_pos):
    expanded_size = board.shape[-1] + 2 * view_distance

    large_board = np.full(
        (*board.shape[:-2], expanded_size, expanded_size), 0, dtype=float)
    large_board[..., view_distance:view_distance+board.shape[-2],
                view_distance:view_distance+board.shape[-1]] = board

    board_size = (view_distance * 2) + 1

    x = player_pos[0][0]
    y = player_pos[1][0]

    board = large_board[:, x:x+board_size, y:y+board_size] 

    return board
