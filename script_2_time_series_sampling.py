import numpy as np
import pandas as pd
import csv

traj_data_df = pd.read_csv('patient_traj.csv')
label_name_list = traj_data_df.keys().tolist()
label_name_array = traj_data_df.keys().to_numpy()
print(len(traj_data_df.keys().tolist()))
print(len(traj_data_df))

traj_data_np = traj_data_df.to_numpy()
print(np.shape(traj_data_np))

traj_data_not_nan = traj_data_np == traj_data_np
print(np.sum(traj_data_np[traj_data_np == traj_data_np]))
data_prop = np.sum(traj_data_not_nan, axis=0) / np.shape(traj_data_np)[0]
print('old data proportion ', data_prop)

# redo time array by 1440 min
time_step = 60 * 2  # 1440
patient_name_array = np.unique(traj_data_np[:, 0])
new_traj_array = np.array([]).reshape((0, len(label_name_list)))
patient_name_slice = traj_data_np[:, 0]

for patient_name in patient_name_array:
    traj_block = traj_data_np[patient_name_slice == patient_name]
    current_time = np.min(traj_block[:, 1])
    num_data = 0
    while current_time <= np.max(traj_block[:, 1]):
        time_slice = traj_block[:, 1]
        current_time_block = traj_block[np.logical_and(current_time <= time_slice, current_time + time_step > time_slice)]
        if patient_name == 133 and current_time == 26645:
            print(current_time_block[:, 21 + 2])
        new_line = np.ones(len(label_name_list)) * float('nan')
        new_line[0] = patient_name
        new_line[1] = current_time
        new_line[2:] = np.nanmean(current_time_block[:, 2:], axis=0)
        new_traj_array = np.vstack((new_traj_array, new_line))
        current_time += time_step
        num_data += 1
    print('Patient ', patient_name, ' done ', np.shape(traj_block), num_data)

# check nan
new_traj_not_nan = new_traj_array == new_traj_array
data_prop = np.sum(new_traj_not_nan, axis=0) / np.shape(new_traj_array)[0]
print('data proportion ', data_prop)

# select parameters
selected_data_boolean = data_prop > 0

trimmed_big_mat = new_traj_array[:, selected_data_boolean]
sum_array = np.sum(trimmed_big_mat, axis=1)
print('reduced dimension: ', np.shape(trimmed_big_mat))
print('selected features: ', label_name_array[selected_data_boolean])
print('all valid time steps: ', np.sum(sum_array == sum_array))

f3 = open('Valid_Traj_Param.csv', 'w')
writer_3 = csv.writer(f3)
trimmed_data_prop = data_prop[selected_data_boolean]
label_name_array_ = label_name_array[selected_data_boolean]
for i in range(len(label_name_array[selected_data_boolean])):
    writer_3.writerow([label_name_array_[i], trimmed_data_prop[i]])
f3.close()

patient_name_array = np.unique(trimmed_big_mat[:, 0])  # 0-174
fst_clm_traj_data = trimmed_big_mat[:, 0]

# select valid subsequences
subsequence_length = 12
valid_subsequence_array = np.array([]).reshape((0, 1 + subsequence_length * (np.shape(trimmed_big_mat)[1] - 2)))

f2 = open('valid_ss_test.csv', 'w')
writer = csv.writer(f2)
writer.writerow(['Patient index'] + list(range(subsequence_length * (np.shape(trimmed_big_mat)[1] - 2))))
included_patient = []

f4 = open('valid_ss_with_start_time.csv', 'w')
writer_f4 = csv.writer(f4)
writer_f4.writerow(['Patient index', 'Dimension', 'Measurement time'] + list(range(subsequence_length)) + list(range(subsequence_length)))

for patient_name in patient_name_array:
    traj_block = trimmed_big_mat[patient_name == fst_clm_traj_data, :]
    traj_block_sorted = traj_block[traj_block[:, 1].argsort()]
    traj_block_sorted_ = traj_block_sorted[:, 2:]  # remove patient id & time
    new_valid_ss = 0
    if np.shape(traj_block_sorted_)[0] >= subsequence_length:
        for starting_ind in range(np.shape(traj_block_sorted_)[0] - subsequence_length + 1):
            subsequence_block = traj_block_sorted_[starting_ind:starting_ind+subsequence_length, :]
            subsequence_flat = np.hstack((patient_name, subsequence_block.flatten()))
            if np.any(np.sum(subsequence_block, axis=0) == np.sum(subsequence_block, axis=0)):
                valid_subsequence_array = np.vstack((valid_subsequence_array, subsequence_flat))
                writer.writerow(subsequence_flat)
                new_valid_ss += 1

                subsequence_block_sum = np.sum(subsequence_block, axis=0)
                ra = np.arange(np.size(subsequence_block_sum))
                ra = ra[subsequence_block_sum == subsequence_block_sum]
                ss_starting_time = traj_block_sorted[starting_ind, 1]
                for dim_id in ra:
                    ss = subsequence_block[:, dim_id]
                    norm_ss = (ss - np.mean(ss)) / np.std(ss)
                    row = np.hstack((patient_name, dim_id, ss_starting_time, ss, norm_ss))
                    writer_f4.writerow(row)
    print('Patient ', patient_name, ' done ', np.shape(traj_block_sorted_), new_valid_ss)
    if new_valid_ss > 0:
        included_patient.append(patient_name)
    # else:
    #     print(traj_block[:, :8])

f2.close()
f4.close()
print(np.shape(valid_subsequence_array))
print('Included patients ', len(included_patient), included_patient)







