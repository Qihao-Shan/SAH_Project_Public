import csv

import numpy as np
import pandas as pd

valid_ss_df = pd.read_csv('valid_ss_test.csv')
valid_ss_np = valid_ss_df.to_numpy()
print(np.shape(valid_ss_np))

patient_id_slice = valid_ss_np[:, 0]
included_patients = np.unique(patient_id_slice)
valid_ss_np_ = valid_ss_np[:, 1:]

ss_length = 12  # 120 min x 12

valid_ss_np_3d = valid_ss_np_.reshape((np.shape(valid_ss_np_)[0], int(ss_length), int(np.shape(valid_ss_np_)[1] / ss_length)))

example = valid_ss_np_3d[0, :, :]
print(np.shape(valid_ss_np_3d))
sum_array = np.sum(example, axis=0)
print(example[:, sum_array == sum_array])

# make profile
num_ss = int(np.shape(valid_ss_np_)[0])
num_dim = int(np.shape(valid_ss_np_)[1] / ss_length)
P = np.zeros((num_dim, num_ss))

# normalize all
std_mat_ = np.std(valid_ss_np_3d, axis=1)
std_mat = np.tile(std_mat_, (ss_length, 1, 1)).transpose((1, 0, 2))
std_mat[std_mat == 0] = float('nan')
mean_mat_ = np.mean(valid_ss_np_3d, axis=1)
mean_mat = np.tile(mean_mat_, (ss_length, 1, 1)).transpose((1, 0, 2))
norm_big_mat = (valid_ss_np_3d - mean_mat) / std_mat
print(np.shape(valid_ss_np_3d), np.sum(valid_ss_np_3d == valid_ss_np_3d))
print(np.shape(std_mat_), np.sum(std_mat_ == std_mat_) * 12)
print(np.shape(norm_big_mat), np.sum(norm_big_mat == norm_big_mat))
print('final data rate: ', np.sum(norm_big_mat == norm_big_mat) / np.size(norm_big_mat))

motif_mat = np.ones((np.size(included_patients), ss_length + 2, num_dim)) * float('nan')

f1 = open('motif_data.csv', 'w')
writer = csv.writer(f1)
writer.writerow(['Patient id', 'Dim id'] + list(range(ss_length)))

for patient_id in range(np.size(included_patients)):
    print('Doing patient ', included_patients[patient_id])
    for dim_id in range(num_dim):
        ss_mat = norm_big_mat[patient_id_slice == included_patients[patient_id], :, dim_id]
        if not np.any(ss_mat == ss_mat):
            chosen_motif = np.ones(ss_length) * float('nan')
        elif np.size(ss_mat) > ss_length:
            ss_mat_extended = np.tile(ss_mat, (np.shape(ss_mat)[0], 1, 1))
            ss_mat_extended_ = ss_mat_extended.transpose((1, 0, 2))
            dist_mat = np.sqrt(np.sum((ss_mat_extended - ss_mat_extended_) ** 2, axis=2))
            dist_array = np.nanmean(dist_mat, axis=1)
            chosen_motif = ss_mat[np.nanargmin(dist_array), :]
        else:
            chosen_motif = ss_mat.flatten()
        line = np.hstack((included_patients[patient_id], dim_id, chosen_motif))
        writer.writerow(line)
        motif_mat[patient_id, :, dim_id] = line
f1.close()

