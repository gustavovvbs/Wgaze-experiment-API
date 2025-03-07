import os

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from fastdtw import fastdtw
from pymongo import MongoClient
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import euclidean

from gaze_utils import generate_gaze_points, generate_key_positions
from utils import compute_gaussian_probabilities, pad_or_truncate
from visualize import visualize_gaze_path

load_dotenv()

DB_NAME = "wgaze"
DB_COLLECTION = "batches"


def calculate_l2_distance_with_temporal(gesture1, gesture2):
    """
    Calculate L2 distance between two gesture trajectories.
    """
    return np.linalg.norm(gesture1 - gesture2)


def calculate_dtw_distance_with_temporal(gesture1, gesture2):
    """
    Calculate Dynamic Time Warping distance between two gesture trajectories.
    """
    distance, _ = fastdtw(gesture1, gesture2, dist=euclidean)
    return distance


def set_distances_between_words_to_infinity(distance_matrix, word_labels_generated, word_labels_real):
    """
    Set distances between different words to infinity to ensure only same-word gestures are matched.
    """
    word_labels_generated = np.array(word_labels_generated).reshape(-1, 1)  # (num_generated, 1)
    word_labels_real = np.array(word_labels_real).reshape(1, -1)            # (1, num_real)

    # create mask to set distances between different words to infinity
    mismatch_mask = word_labels_generated != word_labels_real  # Broadcasting to (num_generated, num_real)
    distance_matrix[mismatch_mask] = np.inf

    return distance_matrix


def calculate_pairwise_distances_with_temporal(generated_gestures, real_gestures, distance_metric='L2'):
    """
    Calculate pairwise distances between all generated and real gestures.
    """
    num_generated = len(generated_gestures)
    num_real = len(real_gestures)
    cost_matrix = np.zeros((num_generated, num_real))

    for i in range(num_generated):
        for j in range(num_real):
            if distance_metric == 'L2':
                cost_matrix[i, j] = calculate_l2_distance_with_temporal(generated_gestures[i], real_gestures[j])
            elif distance_metric == 'DTW':
                cost_matrix[i, j] = calculate_dtw_distance_with_temporal(generated_gestures[i], real_gestures[j])

    return cost_matrix


def calculate_wasserstein_distance(generated_gestures, real_gestures, word_labels_generated, word_labels_real, distance_metric='L2'):
    """
    Calculate Wasserstein distance using linear assignment with temporal information.
    
    Parameters:
        generated_gestures: List of np arrays containing generated gaze data
        real_gestures: List of np arrays containing real gaze data
        word_labels_generated: List of word labels for generated data
        word_labels_real: List of word labels for real data
        distance_metric: Distance metric to use ('L2' or 'DTW')
        
    Returns:
        Average optimal assignment distance
    """
    cost_matrix = calculate_pairwise_distances_with_temporal(generated_gestures, real_gestures, distance_metric) 
    cost_matrix = set_distances_between_words_to_infinity(cost_matrix, word_labels_generated, word_labels_real)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    selected_distances = cost_matrix[row_ind, col_ind]

    finite_distances = selected_distances[np.isfinite(selected_distances)]

    if len(finite_distances) == 0:
        return np.inf

    average_min_distance = finite_distances.mean()

    return average_min_distance


def get_last_n_gestures(db, collection_name, n=5):
    """
    Get the last n gestures from the database.
    
    Parameters:
        db: MongoDB database connection
        collection_name: Name of the collection
        n: Number of gestures to fetch
        
    Returns:
        List of gesture records
    """
    collection = db[DB_COLLECTION]
    last_n_gestures = list(collection.find().sort('_id', -1).limit(n))
    return last_n_gestures


def generate_n_gestures(data, key_positions, wrong_key_prob=0.2):
    """
    Generate synthetic gestures based on real gesture records.
    
    Parameters:
        data: List of real gesture records
        key_positions: Dictionary of key positions
        wrong_key_prob: Probability of fixating on wrong keys
        
    Returns:
        List of generated gesture paths
    """
    generated_gestures = []
    for record in data:
        word = record.get('word', 'UNKNOWN')
        generated_gesture = generate_gaze_points(
            word, 
            key_positions, 
            num_points=len(record['data']),
            fixation_points_per_key=10, 
            gauss_std=14,
            saccade_points_min=1, 
            saccade_points_max=2,
            wrong_key_probability=wrong_key_prob
        )
        generated_gestures.append(generated_gesture)
    return generated_gestures


def normalize_data_points(data, min_val, max_val):
    """
    Normalize data to range [-1, 1] using min-max scaling.
    
    Parameters:
        data: Value or array to normalize
        min_val: Minimum value in the dataset
        max_val: Maximum value in the dataset
        
    Returns:
        Normalized data
    """
    if isinstance(data, (int, float)):
        # Handle scalar case
        if max_val - min_val == 0:
            return 0.0
        return 2 * (data - min_val) / (max_val - min_val) - 1
    else:
        # Handle array case
        return 2 * (data - min_val) / (max_val - min_val) - 1


def plot_keyboard_layout(ax, key_positions, key_width=120, key_height=120):
    """
    Plot keyboard layout on the given axes.
    
    Parameters:
        ax: Matplotlib axes to draw on
        key_positions: Dictionary of key positions
        key_width: Width of key rectangles
        key_height: Height of key rectangles
    """
    key_width *= 0.75  # Scale down for better visualization
    key_height *= 0.75

    for key, (x, y) in key_positions.items():
        rect = plt.Rectangle((x - key_width/2, y - key_height/2), 
                             key_width, key_height,
                             linewidth=1, edgecolor='black', 
                             facecolor='none', alpha=0.6)
        ax.add_patch(rect)
        ax.text(x, y, key, fontsize=14, ha='center', va='center')


def plot_generated_and_real_gestures(generated_gestures, data, key_positions, output_dir="plots"):
    """
    Plot and compare real vs generated gesture paths.
    
    Parameters:
        generated_gestures: List of generated gesture paths
        data: List of real gesture records
        key_positions: Dictionary of key positions 
        output_dir: Directory to save plots
        
    Returns:
        Tuple of (gesture_data, wasserstein_distances)
    """
    os.makedirs(output_dir, exist_ok=True)
    gesture_data = []
    distances_w = []

    # Collect all coordinates for normalization
    all_x = [pos[0] for pos in key_positions.values()]
    all_y = [pos[1] for pos in key_positions.values()]

    for i, gesture in enumerate(generated_gestures):
        real_gaze_data = np.array([(point[0], point[1]) for point in data[i]['data']])
        all_x.extend(list(real_gaze_data[:, 0]) + [point[0] for point in gesture])
        all_y.extend(list(real_gaze_data[:, 1]) + [point[1] for point in gesture])

    # Compute normalization bounds
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Normalize key positions
    normalized_key_positions = {
        key: (normalize_data_points(pos[0], min_x, max_x), 
              normalize_data_points(pos[1], min_y, max_y)) 
        for key, pos in key_positions.items()
    }

    # Process each gesture pair
    for i, gesture in enumerate(generated_gestures):
        real_gaze_data = np.array([(point[0], point[1]) for point in data[i]['data']])
        word = data[i].get('word', 'UNKNOWN')

        # Normalize coordinate data
        normalized_real_gaze = np.array([
            [normalize_data_points(x, min_x, max_x), 
             normalize_data_points(y, min_y, max_y)] 
            for x, y in real_gaze_data
        ])
        
        normalized_generated_gesture = np.array([
            [normalize_data_points(x, min_x, max_x), 
             normalize_data_points(y, min_y, max_y)] 
            for x, y in gesture
        ])

        # Store normalized data
        gesture_data.append({
            'word': word,
            'real': normalized_real_gaze,
            'generated': normalized_generated_gesture
        })

        # Calculate Wasserstein distance
        wasserstein_distance = calculate_wasserstein_distance(
            [normalized_generated_gesture], 
            [normalized_real_gaze], 
            word_labels_generated=[word], 
            word_labels_real=[word],  
            distance_metric='L2'
        )
        distances_w.append(wasserstein_distance)
        
        print(f'Wasserstein distance for word "{word}" ({i}): {wasserstein_distance:.4f}')

        fig, ax = plt.subplots(figsize=(15, 10))
        
        plot_keyboard_layout(ax, normalized_key_positions)
        
        ax.scatter(normalized_real_gaze[:, 0], normalized_real_gaze[:, 1], 
                  s=50, color='red', label='Real Gaze Points')
        ax.plot(normalized_real_gaze[:, 0], normalized_real_gaze[:, 1], 
               linestyle='-', color='red', alpha=0.6, label='Real Path')
        
        ax.scatter(normalized_generated_gesture[:, 0], normalized_generated_gesture[:, 1], 
                  s=50, color='blue', label='Generated Gaze Points')
        ax.plot(normalized_generated_gesture[:, 0], normalized_generated_gesture[:, 1], 
               linestyle='-', color='blue', alpha=0.6, label='Generated Path')
        
        ax.set_title(f'Generated vs Real Gesture for Word: {word}')
        ax.set_xlabel('X Coordinate (normalized)')
        ax.set_ylabel('Y Coordinate (normalized)')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.legend()
        ax.grid()
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'gesture_comparison_{word}_{i}.png'))
        plt.close()

    print(f'Average Wasserstein distance: {np.mean(distances_w):.4f}')
    print(f'Standard deviation of Wasserstein distances: {np.std(distances_w):.4f}')

    return gesture_data, distances_w


def compare_real_and_synthetic_gestures(mongodb_uri, db_name, collection_name, n_samples=5, wrong_key_prob=0.2):
    """
    Main function to compare real and synthetic gestures.
    
    Parameters:
        mongodb_uri: MongoDB connection URI
        db_name: Name of the database
        collection_name: Name of the collection
        n_samples: Number of samples to analyze
        wrong_key_prob: Probability of wrong key fixations in synthetic data
        
    Returns:
        Tuple of (gesture_data, wasserstein_distances)
    """
    client = MongoClient(mongodb_uri)
    db = client[db_name]
    
    # Get key positions
    key_positions = generate_key_positions()
    
    # Get real gesture data
    real_gestures = get_last_n_gestures(db, collection_name, n=n_samples)
    
    # Generate synthetic gestures
    synthetic_gestures = generate_n_gestures(real_gestures, key_positions, wrong_key_prob)
    
    gesture_data, distances = plot_generated_and_real_gestures(
        synthetic_gestures, real_gestures, key_positions
    )
    
    return gesture_data, distances


def visualize_gaze_comparison(
    mongodb_uri=None, 
    db_name=None,
    collection_name=None,
    n_samples=5, 
    wrong_key_prob=0.2,
    output_dir="comparison_plots",
    specific_words=None,
    show_plots=True
):
    """
    High-level function to easily visualize and compare real vs generated gaze paths.
    
    Parameters:
        mongodb_uri: MongoDB connection URI (uses .env if None)
        db_name: Name of the database (uses .env if None)
        collection_name: Name of the collection (uses .env if None)
        n_samples: Number of samples to analyze
        wrong_key_prob: Probability of wrong key fixations in synthetic data
        output_dir: Directory to save plots
        specific_words: List of specific words to analyze (if None, uses recent samples)
        show_plots: Whether to display the plots (in addition to saving them)
        
    Returns:
        Dictionary with summary statistics and metrics
    """
    
    if None in (mongodb_uri, db_name, collection_name):
        load_dotenv()
        mongodb_uri = mongodb_uri or os.getenv('MONGO_URI')
        db_name = DB_NAME or os.getenv('DB_NAME')
        collection_name = DB_COLLECTION or os.getenv('COLLECTION_NAME')
    
    client = MongoClient(mongodb_uri)
    db = client[db_name]
    key_positions = generate_key_positions()
    
    if specific_words:
        collection = db[collection_name]
        real_gestures = []
        for word in specific_words:
            word_samples = list(collection.find({"word": word}).sort('_id', -1).limit(1))
            if word_samples:
                real_gestures.extend(word_samples)
        if not real_gestures:
            raise ValueError(f"No samples found for specified words: {specific_words}")
    else:
        real_gestures = get_last_n_gestures(db, collection_name, n=n_samples)
    
    synthetic_gestures = generate_n_gestures(real_gestures, key_positions, wrong_key_prob)
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_x = [pos[0] for pos in key_positions.values()]
    all_y = [pos[1] for pos in key_positions.values()]
    
    for i, gesture in enumerate(synthetic_gestures):
        real_gaze_data = np.array([(point[0], point[1]) for point in real_gestures[i]['data']])
        all_x.extend(list(real_gaze_data[:, 0]) + [point[0] for point in gesture])
        all_y.extend(list(real_gaze_data[:, 1]) + [point[1] for point in gesture])
    
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    normalized_key_positions = {
        key: (normalize_data_points(pos[0], min_x, max_x), 
              normalize_data_points(pos[1], min_y, max_y)) 
        for key, pos in key_positions.items()
    }
    
    distances_w = []
    word_metrics = {}
    
    for i, gesture in enumerate(synthetic_gestures):
        real_gaze_data = np.array([(point[0], point[1]) for point in real_gestures[i]['data']])
        word = real_gestures[i].get('word', 'UNKNOWN')
        
        normalized_real_gaze = np.array([
            [normalize_data_points(x, min_x, max_x), 
             normalize_data_points(y, min_y, max_y)] 
            for x, y in real_gaze_data
        ])
        
        normalized_generated_gesture = np.array([
            [normalize_data_points(x, min_x, max_x), 
             normalize_data_points(y, min_y, max_y)] 
            for x, y in gesture
        ])
        
        wasserstein_distance = calculate_wasserstein_distance(
            [normalized_generated_gesture], 
            [normalized_real_gaze], 
            word_labels_generated=[word], 
            word_labels_real=[word],  
            distance_metric='L2'
        )
        distances_w.append(wasserstein_distance)
        
        if word not in word_metrics:
            word_metrics[word] = []
        word_metrics[word].append(wasserstein_distance)
        
        fig, ax = plt.subplots(figsize=(15, 10))
        
        plot_keyboard_layout(ax, normalized_key_positions)
        
        real_colors = np.linspace(0.2, 1, len(normalized_real_gaze))
        ax.scatter(normalized_real_gaze[:, 0], normalized_real_gaze[:, 1], 
                  s=50, c=real_colors, cmap='Reds', label='Real Gaze Points', alpha=0.7)
        ax.plot(normalized_real_gaze[:, 0], normalized_real_gaze[:, 1], 
               linestyle='-', color='red', alpha=0.5, label='Real Path')
        
        # Plot generated gaze path with time-based coloring
        gen_colors = np.linspace(0.2, 1, len(normalized_generated_gesture))
        ax.scatter(normalized_generated_gesture[:, 0], normalized_generated_gesture[:, 1], 
                  s=50, c=gen_colors, cmap='Blues', label='Generated Gaze Points', alpha=0.7)
        ax.plot(normalized_generated_gesture[:, 0], normalized_generated_gesture[:, 1], 
               linestyle='-', color='blue', alpha=0.5, label='Generated Path')
        
        # Add markers for start and end points
        ax.scatter([normalized_real_gaze[0, 0]], [normalized_real_gaze[0, 1]], 
                 color='darkred', s=100, marker='o', label='Real Start')
        ax.scatter([normalized_real_gaze[-1, 0]], [normalized_real_gaze[-1, 1]], 
                 color='darkred', s=100, marker='X', label='Real End')
        
        ax.scatter([normalized_generated_gesture[0, 0]], [normalized_generated_gesture[0, 1]], 
                 color='darkblue', s=100, marker='o', label='Gen Start')
        ax.scatter([normalized_generated_gesture[-1, 0]], [normalized_generated_gesture[-1, 1]], 
                 color='darkblue', s=100, marker='X', label='Gen End')
        
        ax.set_title(f'Generated vs Real Gaze Path for Word: "{word}" (W-Distance: {wasserstein_distance:.4f})')
        ax.set_xlabel('X Coordinate (normalized)')
        ax.set_ylabel('Y Coordinate (normalized)')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'comparison_{word}_{i}.png'))
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    words = list(word_metrics.keys())
    avg_distances = [np.mean(word_metrics[w]) for w in words]
    std_distances = [np.std(word_metrics[w]) for w in words]
    
    sorted_indices = np.argsort(avg_distances)
    words = [words[i] for i in sorted_indices]
    avg_distances = [avg_distances[i] for i in sorted_indices]
    std_distances = [std_distances[i] for i in sorted_indices]
    
    bars = ax.bar(words, avg_distances, yerr=std_distances, capsize=10,
                  color='skyblue', edgecolor='navy', alpha=0.7)
    
    for bar, val in zip(bars, avg_distances):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_title('Wasserstein Distance by Word (Lower is Better)')
    ax.set_xlabel('Word')
    ax.set_ylabel('Wasserstein Distance')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'wasserstein_distances_by_word.png'))
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    avg_dist = np.mean(distances_w)
    std_dist = np.std(distances_w)
    min_dist = np.min(distances_w)
    max_dist = np.max(distances_w)
    
    print("\n===== Wasserstein Distance Summary =====")
    print(f"Average: {avg_dist:.4f}")
    print(f"Std Dev: {std_dist:.4f}")
    print(f"Min: {min_dist:.4f}")
    print(f"Max: {max_dist:.4f}")
    
    print("\n===== Word-level Statistics =====")
    for word in words:
        avg = np.mean(word_metrics[word])
        std = np.std(word_metrics[word])
        print(f"Word '{word}': Avg = {avg:.4f}, Std = {std:.4f}")
    
    results = {
        'wasserstein_distances': distances_w,
        'average_distance': avg_dist,
        'std_distance': std_dist,
        'min_distance': min_dist,
        'max_distance': max_dist,
        'word_metrics': word_metrics,
        'words': words,
        'output_dir': output_dir
    }
    
    return results


if __name__ == "__main__":
    results = visualize_gaze_comparison(
        n_samples=10,
        wrong_key_prob=0.2,
        output_dir="gaze_comparisons",
        show_plots=False  # set to True to display plots during execution
    )
    print(f"Visualizations saved to: {results['output_dir']}")

