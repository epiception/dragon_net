import numpy as np
import glob
from natsort import natsorted, ns
import os

sequence_length = 3

name_list = []
frame_limits = []

for i in range(4):
    folder_frame_names = natsorted(glob.glob("/media/dataset/data/dataset/living_room_traj%d_frei_png/rgb/*.png"%i), alg = ns.PATH)
    name_list = name_list + folder_frame_names
    frame_limits.append(len(folder_frame_names))

init_idx = 0
total_sequences = 0
file = open("Sequential_set_frames.txt", "w")

for limit in frame_limits:
    for idx in range(0,limit - sequence_length):
        main_path = name_list[init_idx + idx][:55]
        file.write(main_path + " ")
        frame_list_small = []
        for no in range(idx,idx + sequence_length + 1):
            frame_list_small.append("rgb/%d.png"%no)
        file.write(" ".join(frame_list_small) + " ")

        depth_list_small = []
        for no in range(idx,idx + sequence_length):
            depth_list_small.append("depth/%d.png"%no)
        file.write(" ".join(depth_list_small) + " ")
        file.write("\n")
        total_sequences +=1
    print("***************")
    init_idx = init_idx + limit

file.close()
dataset = np.loadtxt("Sequential_set_frames.txt", delimiter=" ", dtype=str)
np.random.shuffle(dataset)
np.savetxt("Sequential_set_frames.txt", dataset, fmt='%s')
