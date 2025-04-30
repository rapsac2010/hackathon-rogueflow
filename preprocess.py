import os
import shutil  # Import the shutil module

import numpy as np
import pandas as pd
import kaggle

from utils import trim_videos_ffmpeg

# Constants
FPS = 30
SECONDS_PER_VIDEO = 6
DATA_DIR = 'data'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
DIRS_TO_CREATE = [
    'data/test_trimmed',
    'data/train_trimmed_500',
    'data/train_trimmed_1000',
    'data/train_trimmed_1500',
]

def download_data():
    # kaggle competitions download -c nexar-collision-prediction


def load_data(data_dir):
    """Loads train and test CSVs and prepares initial DataFrames."""
    df_full = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    df_full['file'] = df_full['id'].apply(lambda x: f'{x:05d}.mp4')
    df_test['file'] = df_test['id'].apply(lambda x: f'{x:05d}.mp4')

    df_test['cutoff_placeholder'] = np.nan  # Used as placeholder for trim_videos

    df_full['cutoff_time_500'] = df_full['time_of_event'] - 0.5
    df_full['cutoff_time_1000'] = df_full['time_of_event'] - 1
    df_full['cutoff_time_1500'] = df_full['time_of_event'] - 1.5

    return df_full, df_test


def prepare_directories(dirs_to_create):
    """Removes existing directories and creates empty ones."""
    print("Preparing output directories...")
    for dir_path in dirs_to_create:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    print("Directories ready.")


def trim_train_videos(train_dir, output_dir_base, df_full, total_frames, offsets):
    """Trims training videos for specified offsets."""
    print('Trimming training set...')
    for i, offset_ms in enumerate(offsets):
        print(f'{offset_ms} ms offset... ({i+1}/{len(offsets)})')
        output_dir = f"{output_dir_base}_{offset_ms}"
        cutoff_col = f'cutoff_time_{offset_ms}'
        trim_videos_ffmpeg(train_dir, output_dir, df_full, total_frames, cutoff_col)

    print('Training set trimming complete.')


def trim_test_videos(test_dir, output_dir, df_test, total_frames):
    """Trims test videos."""
    print('Trimming test set...')
    trim_videos_ffmpeg(test_dir, output_dir, df_test, total_frames, 'cutoff_placeholder')
    print('Test set trimming complete.')

def main():
    """Main preprocessing pipeline."""
    np.random.seed(42) # Setting seed here affects any random operations if needed later

    df_full, df_test = load_data(DATA_DIR)
    prepare_directories(DIRS_TO_CREATE)

    total_frames = FPS * SECONDS_PER_VIDEO
    train_offsets = [500, 1000, 1500]

    trim_train_videos(TRAIN_DIR, 'data/train_trimmed', df_full, total_frames, train_offsets)
    trim_test_videos(TEST_DIR, 'data/test_trimmed', df_test, total_frames)

    print("Preprocessing finished.")




if __name__ == "__main__":
    main()