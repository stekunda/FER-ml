import os
import random

TRAIN_DATA_PATH = os.path.join(
    os.getcwd(), '../FER with DL/data', 'train_data_explore')


# Reducing the amount of data to 1000 samples per class for faster training
def delete_files(directory, num_files_to_keep):
    for subdir in os.listdir(directory):
        if subdir == 'disgust':
            continue
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            files = os.listdir(subdir_path)
            if len(files) > num_files_to_keep:
                files_to_delete = random.sample(
                    files, len(files) - num_files_to_keep)
                for file in files_to_delete:
                    os.remove(os.path.join(subdir_path, file))


# Call the function for the training data directory
delete_files(TRAIN_DATA_PATH, 500)
