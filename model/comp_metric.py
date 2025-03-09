import matplotlib.pyplot as plt
import numpy as np
from generation import *
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from utils import calculate_wasserstein_distance
load_dotenv()


def get_last_n_gestures(db, collection_name, n=5):
    collection = db[collection_name]
    last_n_gestures = list(collection.find().sort('_id', -1).limit(n))
    return last_n_gestures


def generated_n_gestures(data, key_positions, wrong_key_prob):
    generated_gestures = []
    for record in data:
        word = record.get('word', 'UNKNOWN')
        generated_gesture = generate_gaze_points(
            word, key_positions, fixation_points_per_key=10, 
            gauss_std=14,
            num_points=len(record['data']),
            saccade_points_min=1, saccade_points_max=2,
            wrong_key_probability=wrong_key_prob
        )
        generated_gestures.append(generated_gesture)
    return generated_gestures


def plot_keyboard_layout(key_positions):
    key_width = 160 * 0.75
    key_height = 160 * 0.75

    for key, (x, y) in key_positions.items():
        rect = plt.Rectangle((x - key_width / 2, y - key_height / 2), key_width, key_height,
                             linewidth=1, edgecolor='black', facecolor='none', alpha=0.6)
        plt.gca().add_patch(rect)
        plt.text(x, y, key, fontsize=14, ha='center', va='center')


def normalize_data(value, min_val, max_val):
    if max_val - min_val == 0:
        return 0.0  
    return 2 * (value - min_val) / (max_val - min_val) - 1

def normalize_data(data, min_val, max_val):
    return 2 * (data - min_val) / (max_val - min_val) - 1


def plot_generated_and_real_gestures(generated_gestures, data, key_positions, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    gesture_data = []

    all_key_positions_x = [pos[0] for pos in key_positions.values()]
    all_key_positions_y = [pos[1] for pos in key_positions.values()]

    words_generated = []
    words_real = []

    all_x = all_key_positions_x.copy()
    all_y = all_key_positions_y.copy()

    for i, gesture in enumerate(generated_gestures):
        real_gaze_data = np.array([(point[0], point[1]) for point in data[i]['data']])
        generated_gesture = np.array(gesture)
        word = data[i].get('word', 'UNKNOWN')

        words_generated.append(word)
        words_real.append(word)  

        all_x.extend(list(real_gaze_data[:, 0]) + list(generated_gesture[:, 0]))
        all_y.extend(list(real_gaze_data[:, 1]) + list(generated_gesture[:, 1]))

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    normalized_key_positions = {key: (normalize_data(pos[0], min_x, max_x), normalize_data(pos[1], min_y, max_y)) 
                                 for key, pos in key_positions.items()}

    distances_w = []
    for i, gesture in enumerate(generated_gestures):
        real_gaze_data = np.array([(point[0], point[1]) for point in data[i]['data']])
        word = data[i].get('word', 'UNKNOWN')

        normalized_real_gaze = np.array([[normalize_data(x, min_x, max_x), normalize_data(y, min_y, max_y)] 
                                         for x, y in real_gaze_data])
        normalized_generated_gesture = np.array([[normalize_data(x, min_x, max_x), normalize_data(y, min_y, max_y)] 
                                                 for x, y in gesture])

        gesture_data.append({
            'word': word,
            'real': normalized_real_gaze,
            'generated': normalized_generated_gesture
        })

       
        wasserstein_distance = calculate_wasserstein_distance(
            [normalized_generated_gesture], 
            [normalized_real_gaze], 
            word_labels_generated=[word], 
            word_labels_real=[word],  
            distance_metric='L2'
        )
        distances_w.append(wasserstein_distance)
        print(f'Wasserstein distance for word "{word}", {i}: {wasserstein_distance}')
        print(f'Average Wasserstein distance for all words: {np.mean(distances_w):.4f}')
        print(f'Standard deviation of Wasserstein distances for all words: {np.std(distances_w):.4f}')

        plt.figure(figsize=(15, 10))

        plot_keyboard_layout(normalized_key_positions)

        plt.scatter(normalized_real_gaze[:, 0], normalized_real_gaze[:, 1], s=50, color='red', label='Real Gaze Points')
        plt.plot(normalized_real_gaze[:, 0], normalized_real_gaze[:, 1], linestyle='-', color='red', alpha=0.6, label='Real Path')

        plt.scatter(normalized_generated_gesture[:, 0], normalized_generated_gesture[:, 1], s=50, color='blue', label='Generated Gaze Points')
        plt.plot(normalized_generated_gesture[:, 0], normalized_generated_gesture[:, 1], linestyle='-', color='blue', alpha=0.6, label='Generated Path')

        plt.title(f'Generated vs Real Gesture for Word: {word}')
        plt.xlabel('X Coordinate (normalized)')
        plt.ylabel('Y Coordinate (normalized)')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.legend()
        plt.grid()
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, f'gesture_comparison_{word}_{i}.png'))
        plt.close()

    return gesture_data, distances_w

