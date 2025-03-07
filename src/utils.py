import numpy as np

def generate_key_positions(
    win_width=2048,
    win_height=1152,
    key_width=120,         
    key_height=120,
    horizontal_spacing=30,  
    vertical_spacing=200
):
    """
    Returns a dictionary with the approximate positions (x, y) of each key.
    For illustration, 3 rows of keys + one 'BOTAO_ACABAR' at the bottom.
    """
    top_row_y = win_height / 2 - vertical_spacing * 1.5
    middle_row_y = top_row_y - vertical_spacing
    bottom_row_y = middle_row_y - vertical_spacing

    row1_labels = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P']
    row2_labels = ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L']
    row3_labels = ['Z', 'X', 'C', 'V', 'B', 'N', 'M']

    def calc_start_x(num_keys):
        total_width = num_keys * key_width + (num_keys - 1) * horizontal_spacing
        return -total_width / 2 + key_width / 2

    def compute_positions(labels, start_x, y_pos):
        pos_list = []
        for i, label in enumerate(labels):
            x = start_x + i * (key_width + horizontal_spacing)
            pos_list.append((label.upper(), [x, y_pos]))
        return pos_list

    start_x_r1 = calc_start_x(len(row1_labels))
    start_x_r2 = calc_start_x(len(row2_labels))
    start_x_r3 = calc_start_x(len(row3_labels))

    key_positions = {}
    for label, pos in compute_positions(row1_labels, start_x_r1, top_row_y):
        key_positions[label] = pos
    for label, pos in compute_positions(row2_labels, start_x_r2, middle_row_y):
        key_positions[label] = pos
    for label, pos in compute_positions(row3_labels, start_x_r3, bottom_row_y):
        key_positions[label] = pos

    key_positions['BOTAO_ACABAR'] = [0, bottom_row_y - vertical_spacing]
    return key_positions

def pad_or_truncate(points_2d, num_points):
    """
    Ensures each (N,2) array has exactly num_points rows.
    If smaller, pad with zeros; if larger, truncate.
    """
    cur_len = points_2d.shape[0]
    if cur_len > num_points:
        return points_2d[:num_points]
    else:
        needed = num_points - cur_len
        pad_block = np.zeros((needed, 2))
        return np.vstack([points_2d, pad_block])

def compute_gaussian_probabilities(gaze_points, key_positions, sigma=85.0):
    """
    for each gaze point, compute a normalized Gaussian affinity for each key
    return shape: (num_points, num_keys)
    """
    gaze_points = np.array(gaze_points)  # (num_points, 2)
    keys = sorted(key_positions.keys())
    coords = np.array([key_positions[k] for k in keys])  # (num_keys, 2)

    prob_array = np.zeros((gaze_points.shape[0], len(keys)), dtype=np.float32)
    for i, gp in enumerate(gaze_points):
        dist_sq = np.sum((coords - gp) ** 2, axis=1)
        gaussians = np.exp(-dist_sq / (2 * (sigma ** 2)))
        norm = np.sum(gaussians) + 1e-8
        prob_array[i] = gaussians / norm
    return prob_array
