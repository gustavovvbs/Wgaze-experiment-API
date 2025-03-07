import os 
import torch

MONGODB_URI = "mongodb+srv://gustavovvbs:czIfQXzCyM2jSSfz@wgaze.bgcts.mongodb.net/"
DB_NAME = "wgaze"
COLLECTION_NAME = "batches"
WORDS_FILENAME = "data/words.txt"

REAL_SAMPLES_TO_FETCH = 200
SYNTHETIC_PER_WORD = 150
NUM_POINTS = 170
NUM_EPOCHS = 22
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_MODE = 'ratio'  # 'all_combined', 'only_synthetic', ou 'ratio'
REAL_SYNTH_RATIO = 1/150.0
REAL_TEST_RATIO = 0.2

WINDOW_WIDTH = 2048
WINDOW_HEIGHT = 1152
KEY_WIDTH = 120
KEY_HEIGHT = 120
HORIZONTAL_SPACING = 30
VERTICAL_SPACING = 200

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'