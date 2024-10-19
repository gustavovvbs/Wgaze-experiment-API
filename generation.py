import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
import random

def plot_last_n_gestures(db, collection_name, n=5, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    collection = db[collection_name]
    data = list(collection.find({'word': {'$exists': True}})
                         .sort([('_id', -1)])
                         .limit(n))

    if not data:
        print("No gesture data found in the collection.")
        return

    win_width = 2048
    win_height = 1152
    key_width = 160 * 0.75
    key_height = 160 * 0.75
    horizontal_spacing = key_width * 0.25
    vertical_spacing = 200

    top_row_y = win_height / 2 - vertical_spacing * 1.5
    middle_row_y = top_row_y - vertical_spacing
    bottom_row_y = middle_row_y - vertical_spacing

    row1_labels = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P']
    row2_labels = ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L']
    row3_labels = ['Z', 'X', 'C', 'V', 'B', 'N', 'M']

    def calculate_start_x(num_keys):
        total_width = num_keys * key_width + (num_keys - 1) * horizontal_spacing
        return -total_width / 2 + key_width / 2

    def compute_key_positions(labels, start_x, y_pos):
        positions = []
        for i, label in enumerate(labels):
            x = start_x + i * (key_width + horizontal_spacing)
            positions.append((label, [x, y_pos]))
        return positions

    start_x_row1 = calculate_start_x(len(row1_labels))
    start_x_row2 = calculate_start_x(len(row2_labels))
    start_x_row3 = calculate_start_x(len(row3_labels))

   
    key_positions = {}
    for label, pos in compute_key_positions(row1_labels, start_x_row1, top_row_y):
        key_positions[label.upper()] = pos
    for label, pos in compute_key_positions(row2_labels, start_x_row2, middle_row_y):
        key_positions[label.upper()] = pos
    for label, pos in compute_key_positions(row3_labels, start_x_row3, bottom_row_y):
        key_positions[label.upper()] = pos

    key_positions['BOTAO_ACABAR'] = [0, bottom_row_y - vertical_spacing]

    for i, gesture in enumerate(data):
        df = pd.DataFrame([gesture]).drop(columns=['_id', 'name'])
        gaze_data = np.array([np.array(x) for x in df['data'].values[0]])

        plt.figure(figsize=(15, 10))

        plt.scatter(gaze_data[:, 0], gaze_data[:, 1], s=50, label='Gaze Points')

        plt.plot(gaze_data[:, 0], gaze_data[:, 1], linestyle='-', color='blue', alpha=0.6)

        keys = []
        keys += compute_key_positions(row1_labels, start_x_row1, top_row_y)
        keys += compute_key_positions(row2_labels, start_x_row2, middle_row_y)
        keys += compute_key_positions(row3_labels, start_x_row3, bottom_row_y)
        keys.append(('BOTAO_ACABAR', [0, bottom_row_y - vertical_spacing]))

        for label, (x, y) in keys:
            rect = plt.Rectangle((x - key_width/2, y - key_height/2), key_width, key_height,
                                 linewidth=1, edgecolor='black', facecolor='lightgray')
            plt.gca().add_patch(rect)
            plt.text(x, y, label, fontsize=12, ha='center', va='center')

        plt.xlim(-win_width/2, win_width/2)
        plt.ylim(-win_height/2, win_height/2)
        plt.title(f'Gesture {i+1}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'gesture_{i+1}.png'))
        plt.close()


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
    gaze_points = []

    word = word.upper()

    all_x_positions = [pos[0] for pos in key_positions.values()]
    center_x = (max(all_x_positions) + min(all_x_positions)) / 2

    def adjust_extreme_positions(pos):
        x, y = pos
        threshold = (max(all_x_positions) - min(all_x_positions)) * 0.3
        distance_from_center = x - center_x
        if abs(distance_from_center) > threshold:
            adjustment_factor = 0.2
            x -= adjustment_factor * distance_from_center
        return (x, y)

    start_key = 'BOTAO_ACABAR'
    if start_key not in key_positions:
        print(f"Key '{start_key}' not found in key_positions.")
        return gaze_points

    start_pos = adjust_extreme_positions(key_positions[start_key])
    for _ in range(fixation_points_per_key):
        start_fixation_x = start_pos[0] + np.random.normal(0, gauss_std_center)
        start_fixation_y = start_pos[1] + np.random.normal(0, gauss_std_center)
        gaze_points.append([start_fixation_x, start_fixation_y])

    previous_key_pos = gaze_points[-1]  

    for char in word:
        if char not in key_positions:
            print(f"Letter '{char}' not found in key_positions.")
            continue

        key_pos = adjust_extreme_positions(key_positions[char])
        current_key_pos = (
            key_pos[0] + np.random.normal(0, gauss_std_center),
            key_pos[1] + np.random.normal(0, gauss_std_center)
        )

        if random.random() < wrong_key_probability:
            wrong_key = random.choice([
                k for k in key_positions if k != char and k in key_positions
            ])
            wrong_key_pos = adjust_extreme_positions(key_positions[wrong_key])
            wrong_key_fixation = (
                wrong_key_pos[0] + np.random.normal(0, gauss_std_center),
                wrong_key_pos[1] + np.random.normal(0, gauss_std_center)
            )

            for _ in range(wrong_key_fixation_points):
                fixation_x = np.random.normal(wrong_key_fixation[0], gauss_std)
                fixation_y = np.random.normal(wrong_key_fixation[1], gauss_std)
                gaze_points.append([fixation_x, fixation_y])

        num_saccades = random.randint(saccade_points_min, saccade_points_max)
        for _ in range(num_saccades):
            t = random.uniform(0, 1)
            dx = current_key_pos[0] - previous_key_pos[0]
            dy = current_key_pos[1] - previous_key_pos[1]

            saccade_x = previous_key_pos[0] + t * dx + np.random.normal(0, gauss_std)
            saccade_y = previous_key_pos[1] + t * dy + np.random.normal(0, gauss_std)
            gaze_points.append([saccade_x, saccade_y])

        for _ in range(fixation_points_per_key):
            fixation_x = np.random.normal(current_key_pos[0], gauss_std)
            fixation_y = np.random.normal(current_key_pos[1], gauss_std)
            gaze_points.append([fixation_x, fixation_y])

        previous_key_pos = gaze_points[-1] 

    end_pos = adjust_extreme_positions(key_positions[start_key])
    current_key_pos = (
        end_pos[0] + np.random.normal(0, gauss_std_center),
        end_pos[1] + np.random.normal(0, gauss_std_center)
    )

    num_saccades = random.randint(saccade_points_min, saccade_points_max)
    for _ in range(num_saccades):
        t = random.uniform(0, 1)
        dx = current_key_pos[0] - previous_key_pos[0]
        dy = current_key_pos[1] - previous_key_pos[1]

        saccade_x = previous_key_pos[0] + t * dx + np.random.normal(0, gauss_std)
        saccade_y = previous_key_pos[1] + t * dy + np.random.normal(0, gauss_std)
        gaze_points.append([saccade_x, saccade_y])

    for _ in range(fixation_points_per_key):
        end_fixation_x = current_key_pos[0] + np.random.normal(0, gauss_std)
        end_fixation_y = current_key_pos[1] + np.random.normal(0, gauss_std)
        gaze_points.append([end_fixation_x, end_fixation_y])

    if num_points is not None:
        current_length = len(gaze_points)
        if current_length > num_points:
            indices = np.round(np.linspace(0, current_length - 1, num_points)).astype(int)
            gaze_points = [gaze_points[i] for i in indices]
        elif current_length < num_points:
            factor = num_points // current_length
            remainder = num_points % current_length
            gaze_points = gaze_points * factor + gaze_points[:remainder]

    return gaze_points



def prepare_key_positions():
    win_width = 2048
    win_height = 1152
    key_width = 160 * 0.75
    key_height = 160 * 0.75
    horizontal_spacing = key_width * 0.25
    vertical_spacing = 200

    top_row_y = win_height / 2 - vertical_spacing * 1.5
    middle_row_y = top_row_y - vertical_spacing
    bottom_row_y = middle_row_y - vertical_spacing

    row1_labels = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P']
    row2_labels = ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L']
    row3_labels = ['Z', 'X', 'C', 'V', 'B', 'N', 'M']

    def calculate_start_x(num_keys):
        total_width = num_keys * key_width + (num_keys - 1) * horizontal_spacing
        return -total_width / 2 + key_width / 2

    def compute_key_positions(labels, start_x, y_pos):
        positions = []
        for i, label in enumerate(labels):
            x = start_x + i * (key_width + horizontal_spacing)
            positions.append((label, [x, y_pos]))
        return positions

    start_x_row1 = calculate_start_x(len(row1_labels))
    start_x_row2 = calculate_start_x(len(row2_labels))
    start_x_row3 = calculate_start_x(len(row3_labels))

    key_positions = {}
    for label, pos in compute_key_positions(row1_labels, start_x_row1, top_row_y):
        key_positions[label.upper()] = pos
    for label, pos in compute_key_positions(row2_labels, start_x_row2, middle_row_y):
        key_positions[label.upper()] = pos
    for label, pos in compute_key_positions(row3_labels, start_x_row3, bottom_row_y):
        key_positions[label.upper()] = pos
    key_positions['BOTAO_ACABAR'] = [0, bottom_row_y - vertical_spacing]

    return key_positions

def plot_generated_gaze_points(gaze_points, key_positions, word,
                               win_width=2048, win_height=1152, 
                               key_width=120, key_height=120):

    plt.figure(figsize=(15, 10))

    for label, (x, y) in key_positions.items():
        rect = plt.Rectangle((x - key_width/2, y - key_height/2), key_width, key_height,
                             linewidth=1, edgecolor='black', facecolor='lightgray', zorder=1)
        plt.gca().add_patch(rect)
        plt.text(x, y, label, fontsize=12, ha='center', va='center', zorder=2)

   
    if gaze_points:
        gaze_array = np.array(gaze_points)
        plt.scatter(gaze_array[:, 0], gaze_array[:, 1], s=50, color='red', label='Gaze Points', zorder=3)
        plt.plot(gaze_array[:, 0], gaze_array[:, 1], linestyle='-', color='blue', alpha=0.6, zorder=3)

    plt.xlim(-win_width/2, win_width/2)
    plt.ylim(-win_height/2, win_height/2)
    plt.title(f'Simulated Gaze Points for Word {word}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.show()



