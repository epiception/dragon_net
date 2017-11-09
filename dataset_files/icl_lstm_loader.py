import numpy as np
import scipy.misc as smc
from skimage import io, color
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
plt.ion()


cur_folder_path = (os.path.dirname(os.path.abspath(__file__)))
print(cur_folder_path)
parsed_set = np.loadtxt(cur_folder_path + "/dataset_file.txt", dtype=str, delimiter=" ")

IMG_HT = 480
IMG_WDT = 640

total_train = 3780
total_validation = 720
partition_limit = 180
sequence_size = 3

training_set = parsed_set[:total_train]
validation_set = parsed_set[total_train:total_train + total_validation]

def shuffle_sets():
    np.random.shuffle(training_set)
    # np.random.shuffle(validation_set)

def load_partition(no, mode):
    if(mode == "Training"):
        subset = training_set[no*partition_limit:(no + 1)*partition_limit]
    if(mode == "Validation"):
        subset = validation_set[no*partition_limit:(no + 1)*partition_limit]

    frame_sequences = np.zeros((partition_limit*sequence_size, IMG_HT, IMG_WDT, 2), dtype = np.float32)
    depth_sequences = np.zeros((partition_limit*sequence_size, IMG_HT, IMG_WDT), dtype = np.float32)

    entry_idx = 0

    for entry in tqdm(subset):
        main_path = entry[0]
        frame_nos = entry[1:5]
        depth_nos = entry[5:8]

        for idx in range(len(frame_nos) - 1):
            frame_sequences[entry_idx + idx, :, :, 0] = smc.imread(main_path + frame_nos[idx], True)
            frame_sequences[entry_idx + idx, :, :, 1] = smc.imread(main_path + frame_nos[idx + 1], True)

        for idx in range(len(depth_nos)):
            depth_sequences[entry_idx + idx, :, :] = smc.imread(main_path + depth_nos[idx])

        entry_idx +=3

    return frame_sequences, depth_sequences
