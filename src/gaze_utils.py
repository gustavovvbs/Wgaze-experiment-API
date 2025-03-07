import random
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

def generate_gaze_points(
    word,
    key_positions,
    num_points=None,
    fixation_points_per_key=10,
    gauss_std=15,
    gauss_std_center=20,
    saccade_points_min=1,
    saccade_points_max=2,
    wrong_key_probability=0.2,
    wrong_key_fixation_points=4
):
    """
    Generates a synthetic 'gaze path' for a given word based on approximate
    key positions. Includes random 'wrong keys' to simulate typical user errors.
    """
    gaze_points = []
    word = word.upper()

    # For centralizing extreme positions
    all_x = [pos[0] for pos in key_positions.values()]
    xmin, xmax = min(all_x), max(all_x)
    center_x = (xmax + xmin) / 2

    # What I consider distant from center is defined by this 0.3 magic value
    threshold = (xmax - xmin) * 0.3

    def adjust_extreme(pos):
        x, y = pos
        dist_center = x - center_x
        if abs(dist_center) > threshold:
            x -= 0.2 * dist_center  # shift slightly back toward center
        return (x, y)

    # Start from BOTAO_ACABAR
    start_key = 'BOTAO_ACABAR'
    if start_key not in key_positions:
        print(f"Key {start_key} not found, returning empty gaze list.")
        return gaze_points

    # 1. Fixation on Start (fixed number of fixations per key)
    start_pos = adjust_extreme(key_positions[start_key])
    for _ in range(fixation_points_per_key):
        x = start_pos[0] + np.random.normal(0, gauss_std_center)
        y = start_pos[1] + np.random.normal(0, gauss_std_center)
        gaze_points.append([x, y])

    prev = gaze_points[-1]

    # 2. for each char in the word
    for char in word:
        if char not in key_positions:
            continue  # unknown keys

        key_pos = adjust_extreme(key_positions[char])
        current_key = (
            key_pos[0] + np.random.normal(0, gauss_std_center),
            key_pos[1] + np.random.normal(0, gauss_std_center)
        )

        # might fixate on a wrong key before the right key
        if random.random() < wrong_key_probability:
            all_keys = list(key_positions.keys())
            all_keys.remove(char)
            wrong_key_choice = random.choice(all_keys)
            wrong_pos = adjust_extreme(key_positions[wrong_key_choice])
            wrong_fix = (
                wrong_pos[0] + np.random.normal(0, gauss_std_center),
                wrong_pos[1] + np.random.normal(0, gauss_std_center)
            )
            # fixed number of fixations in wrong key
            for _ in range(wrong_key_fixation_points):
                fx = np.random.normal(wrong_fix[0], gauss_std)
                fy = np.random.normal(wrong_fix[1], gauss_std)
                gaze_points.append([fx, fy])

        # saccade from previous point to next
        n_sacc = random.randint(saccade_points_min, saccade_points_max)
        for _ in range(n_sacc):
            t = random.uniform(0, 1)
            dx = current_key[0] - prev[0]
            dy = current_key[1] - prev[1]
            sx = prev[0] + t * dx + np.random.normal(0, gauss_std)
            sy = prev[1] + t * dy + np.random.normal(0, gauss_std)
            gaze_points.append([sx, sy])

        # actual fixation on correct point
        for _ in range(fixation_points_per_key):
            fx = np.random.normal(current_key[0], gauss_std)
            fy = np.random.normal(current_key[1], gauss_std)
            gaze_points.append([fx, fy])

        prev = gaze_points[-1]

    # 3. return to BOTAO_ACABAR
    end_pos = adjust_extreme(key_positions[start_key])
    end_key = (
        end_pos[0] + np.random.normal(0, gauss_std_center),
        end_pos[1] + np.random.normal(0, gauss_std_center)
    )
    n_sacc = random.randint(saccade_points_min, saccade_points_max)
    for _ in range(n_sacc):
        t = random.uniform(0, 1)
        dx = end_key[0] - prev[0]
        dy = end_key[1] - prev[1]
        sx = prev[0] + t * dx + np.random.normal(0, gauss_std)
        sy = prev[1] + t * dy + np.random.normal(0, gauss_std)
        gaze_points.append([sx, sy])

    for _ in range(fixation_points_per_key):
        fx = end_key[0] + np.random.normal(0, gauss_std)
        fy = end_key[1] + np.random.normal(0, gauss_std)
        gaze_points.append([fx, fy])

    # 4. if num_points specified, up/down-sample
    if num_points is not None:
        current_length = len(gaze_points)
        if current_length > num_points:
            idxs = np.round(np.linspace(0, current_length - 1, num_points)).astype(int)
            gaze_points = [gaze_points[i] for i in idxs]
        elif current_length < num_points:
            factor = num_points // current_length
            remainder = num_points % current_length
            gaze_points = gaze_points * factor + gaze_points[:remainder]

    return gaze_points