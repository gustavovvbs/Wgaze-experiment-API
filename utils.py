import numpy as np
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment
from fastdtw import fastdtw


def calculate_l2_distance_with_temporal(gesture1, gesture2):
    return np.linalg.norm(gesture1 - gesture2)

def calculate_dtw_distance_with_temporal(gesture1, gesture2):
    
    distance, _ = fastdtw(gesture1, gesture2, dist=euclidean)
    return distance

def set_distances_between_words_to_infinity(distance_matrix, word_labels_generated, word_labels_real):

    word_labels_generated = np.array(word_labels_generated).reshape(-1, 1)  # (num_generated, 1)
    word_labels_real = np.array(word_labels_real).reshape(1, -1)            # (1, num_real)

    #mask p tacar a distancia entre palavras diferentes p infinito
    mismatch_mask = word_labels_generated != word_labels_real  # fzendo o broadcasting p (num_generated, num_real)

    distance_matrix[mismatch_mask] = np.inf

    return distance_matrix

def calculate_pairwise_distances_with_temporal(generated_gestures, real_gestures, distance_metric='L2'):
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

    cost_matrix = calculate_pairwise_distances_with_temporal(generated_gestures, real_gestures, distance_metric)
    cost_matrix = set_distances_between_words_to_infinity(cost_matrix, word_labels_generated, word_labels_real)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    selected_distances = cost_matrix[row_ind, col_ind]

    finite_distances = selected_distances[np.isfinite(selected_distances)]

    if len(finite_distances) == 0:
        return np.inf

    average_min_distance = finite_distances.mean()

    return average_min_distance
