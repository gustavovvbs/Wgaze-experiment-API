import numpy as np 
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from gaze_utils import generate_gaze_points, generate_key_positions

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

def load_real_data(mongodb_uri, db_name, collection_name, words_filename, real_samples_to_fetch):
    """
    Load real data from MongoDB and filter by words in words_filename.
    """
    with open(words_filename, 'r') as f:
        valid_words_list = [w.strip().upper() for w in f.readlines()]
    
    client = MongoClient(mongodb_uri)
    db = client[db_name]
    coll = db[collection_name]
    
    last_gestures = list(coll.find().sort('_id', -1).limit(real_samples_to_fetch))
    
    real_gesture_words = []
    real_gesture_data = []
    for g in last_gestures:
        w = g['word'].upper()
        if w in valid_words_list:
            real_gesture_words.append(w)
            real_gesture_data.append(g['data'])
    
    return real_gesture_words, real_gesture_data, valid_words_list


def process_real_data(real_gesture_data, num_points):
    """
    Process real data into fixed-size 2D arrays.
    """
    real_processed = []
    for arr in real_gesture_data:
        arr_np = np.array(arr)
        arr_np = pad_or_truncate(arr_np, num_points)
        real_processed.append(arr_np)
    return np.array(real_processed)  # shape: (N_real, num_points, 2)

def generate_synthetic_data(unique_words, key_positions, synthetic_per_word, num_points):
    """
    Generate synthetic data for a list of words.
    """
    
    synthetic_data = []
    synthetic_labels = []
    for w in unique_words:
        for _ in range(synthetic_per_word):
            gp = generate_gaze_points(w, key_positions, num_points=num_points)
            gp = np.array(gp)
            synthetic_data.append(gp)
            synthetic_labels.append(w)
    
    return np.array(synthetic_data), np.array(synthetic_labels)


def to_gaussian_features(array_3d):
    # array_3d: shape (N, num_points, 2)
    key_positions = generate_key_positions()
    feats_list = []
    for sample in array_3d:
        feats = compute_gaussian_probabilities(sample, key_positions, sigma=85.0)
        feats_list.append(feats)
    return np.array(feats_list)

def prepare_train_test_data(real_features, synthetic_features, real_y, synth_y, 
                           train_mode, real_synth_ratio=1/150.0, real_test_ratio=0.2):
    """
    Prepare train and test data based on the specified train_mode.
    """
    if train_mode == "all_combined":
        # Original approach: combine real+synthetic, do single train/test split
        X_all = np.concatenate([synthetic_features, real_features], axis=0)
        y_all = np.concatenate([synth_y, real_y], axis=0)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, shuffle=True
        )
        train_X, train_y = X_train, y_train
        test_X, test_y = X_val, y_val
        
        return train_X, train_y, test_X, test_y, "random train_test_split of real+synthetic"
        
    elif train_mode == "only_synthetic":
        # Train only on synthetic, test only on real
        train_X, train_y = synthetic_features, synth_y
        test_X, test_y = real_features, real_y
        
        return train_X, train_y, test_X, test_y, "Train=synthetic, Test=real"
        
    elif train_mode == "ratio":
        # Train on all synthetic plus a fraction of real
        # Test on the remaining real
        N_synth = synthetic_features.shape[0]
        desired_real_train = int(N_synth * real_synth_ratio)
        
        desired_real_test = int(len(real_features) * real_test_ratio)
        
        if desired_real_train > len(real_features) - desired_real_test:
            desired_real_train = len(real_features) - desired_real_test
            print(f"[WARNING] Adjusted desired_real_train to {desired_real_train} to reserve real data for testing.")
        
        if desired_real_train < 0:
            raise ValueError("Not enough real data to satisfy the real_test_ratio. Reduce real_test_ratio or increase real_synth_ratio.")
        
        real_indices = np.arange(len(real_features))
        np.random.shuffle(real_indices)
        
        train_real_idx = real_indices[:desired_real_train]
        test_real_idx = real_indices[desired_real_train:desired_real_train + desired_real_test]
        
        if len(test_real_idx) == 0 and desired_real_test > 0:
            raise ValueError("No real data left for testing. Adjust real_test_ratio or real_synth_ratio.")
        
        train_X = np.concatenate([synthetic_features, real_features[train_real_idx]], axis=0)
        train_y = np.concatenate([synth_y, real_y[train_real_idx]], axis=0)
        
        test_X = real_features[test_real_idx]
        test_y = real_y[test_real_idx]
        
        summary = f"ratio={real_synth_ratio:.5f}. Real in training: {desired_real_train}, Real in test: {len(test_real_idx)}"
        return train_X, train_y, test_X, test_y, summary
        
    else:
        raise ValueError("train_mode must be one of: 'all_combined', 'only_synthetic', 'ratio'.")