import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from config import *
from gaze_utils import generate_key_positions
from data_utils import (
    load_real_data, 
    process_real_data, 
    generate_synthetic_data, 
    to_gaussian_features,
    prepare_train_test_data
)
from dataset import create_data_loaders
from models import ConvClassifier
from train import train_model, evaluate_model
from visualize import plot_history, plot_confusion_matrix, visualize_gaze_path

def run_experiment(
    mongodb_uri=MONGODB_URI,
    db_name=DB_NAME,
    collection_name=COLLECTION_NAME,
    words_filename=WORDS_FILENAME,
    real_samples_to_fetch=REAL_SAMPLES_TO_FETCH,
    synthetic_per_word=SYNTHETIC_PER_WORD,
    num_points=NUM_POINTS,
    num_epochs=NUM_EPOCHS,
    lr=LEARNING_RATE,
    device=DEVICE,
    train_mode=TRAIN_MODE,
    real_synth_ratio=REAL_SYNTH_RATIO,
    real_test_ratio=REAL_TEST_RATIO
):
    """
    High-level function for experiment pipeline
    """
    print("[INFO] Loading key positions ...")
    key_positions = generate_key_positions()
    num_keys = len(key_positions)
    
    print(f"[INFO] Fetching last {real_samples_to_fetch} gestures from DB ...")
    real_gesture_words, real_gesture_data, valid_words_list = load_real_data(
        mongodb_uri, db_name, collection_name, words_filename, real_samples_to_fetch
    )
    print(f"[INFO] Found {len(real_gesture_words)} real gestures matching your words.txt")
    
    real_processed = process_real_data(real_gesture_data, num_points)
    
    unique_real_words = set(real_gesture_words)
    print("[INFO] Generating synthetic data for these words:", unique_real_words)
    synthetic_data, synthetic_labels = generate_synthetic_data(
        unique_real_words, key_positions, synthetic_per_word, num_points
    )
    
    # convert to gaussian features
    print("[INFO] Converting real data to Gaussian features ...")
    real_features = to_gaussian_features(real_processed)
    
    print("[INFO] Converting synthetic data to Gaussian features ...")
    synthetic_features = to_gaussian_features(synthetic_data)
    
    # prepare labels
    all_words = list(unique_real_words)
    label_encoder = LabelEncoder()
    label_encoder.fit(all_words)
    
    # encode real and synthetic
    real_y = label_encoder.transform(real_gesture_words)
    synth_y = label_encoder.transform(synthetic_labels)
    
    # split data based on train_mode
    train_X, train_y, test_X, test_y, mode_info = prepare_train_test_data(
        real_features, synthetic_features, real_y, synth_y, 
        train_mode, real_synth_ratio, real_test_ratio
    )
    print(f"[INFO] Using '{train_mode}' train_mode: {mode_info}")
    
    train_loader, test_loader = create_data_loaders(train_X, train_y, test_X, test_y, batch_size=32)
    
    model = ConvClassifier(num_keys=num_keys, num_classes=len(label_encoder.classes_))
    model.to(device)
    
    print("[INFO] Starting training ...")
    model, history = train_model(
        model, train_loader, test_loader, num_epochs, lr, device
    )
    
    plot_history(history)
    
    print("[INFO] Final evaluation on test set:")
    cm, top4_acc = evaluate_model(model, test_loader, label_encoder, device)
    
    plot_confusion_matrix(cm, label_encoder.classes_)
    
    print("[DONE]")
    return model, label_encoder, key_positions  

if __name__ == "__main__":
    model, label_encoder, key_positions = run_experiment(
        mongodb_uri=MONGODB_URI,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME,
        words_filename=WORDS_FILENAME,
        real_samples_to_fetch=REAL_SAMPLES_TO_FETCH,
        synthetic_per_word=150,
        num_points=NUM_POINTS,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        device=DEVICE,
        train_mode=TRAIN_MODE,
        real_synth_ratio=REAL_SYNTH_RATIO,
        real_test_ratio=REAL_TEST_RATIO
    )