import numpy as np
import pandas as pd
import csv
from datetime import datetime


class PatientTimeSeriesDf:
    def __init__(self, patient_code, patient_df):
        self.patient_code = patient_code
        self.patient_df = patient_df


raw_data_df = pd.read_excel(io='SAB_04_19_bis_09_24_Multiobjective.xlsx', skiprows=2)
patient_code_array = raw_data_df['Code'].to_numpy()
# patient_code_array = patient_code_array[patient_code_array == patient_code_array]
print(patient_code_array)
print(np.size(patient_code_array))

f_out = open('Valid_Patients.csv', 'w')
writer_1 = csv.writer(f_out)
writer_1.writerow(['Index', 'Name', 'Begin time', 'CVS'])

# go through all patients first looking for measurement labels
patient_df_list = []
measurement_label_array = np.array([])
valid_patient_code_list = []
fst_round_patient_selection = np.array([])
counter = 0
for ind in range(np.size(patient_code_array)):
    if patient_code_array[ind] == patient_code_array[ind]:
        try:
            ts_df = pd.read_csv('Longitudinal_Data/'+patient_code_array[ind]+'.csv', dtype='str', delimiter='\t', encoding_errors='ignore', on_bad_lines='skip')
            patient_df_entry = PatientTimeSeriesDf(patient_code_array[ind], ts_df)
            patient_df_list.append(patient_df_entry)
            measurement_label_array = np.hstack((measurement_label_array, np.unique(ts_df['ANGABE'].to_numpy())))
            print(patient_code_array[ind], len(ts_df))
            valid_patient_code_list.append(patient_code_array[ind])
            fst_round_patient_selection = np.hstack((fst_round_patient_selection, 1))
            writer_1.writerow([counter, patient_code_array[ind], ts_df['FALL_BEGINN'].iloc[0], raw_data_df['CVS'].iloc[ind]])
            counter += 1
        except FileNotFoundError:
            fst_round_patient_selection = np.hstack((fst_round_patient_selection, 0))
            print('File Not Found ', patient_code_array[ind])
    else:
        fst_round_patient_selection = np.hstack((fst_round_patient_selection, 0))
print(fst_round_patient_selection)
print(len(patient_df_list))
f_out.close()
measurement_label_array = np.unique(measurement_label_array)
for label_ind in range(np.size(measurement_label_array)):
    measurement_label_array[label_ind] = measurement_label_array[label_ind].replace(' ', '')
measurement_label_array = np.unique(measurement_label_array)
print(len(measurement_label_array), measurement_label_array)
print('Valid patients', len(valid_patient_code_list), len(patient_df_list))

# get traj mat
valid_patient_traj_list = []  # list of np arrays
not_accounted_values = []
for ind in range(len(valid_patient_code_list)):
    traj = np.array([]).reshape((0, len(measurement_label_array) + 1))
    begin_time = patient_df_list[ind].patient_df['FALL_BEGINN'].iloc[0]
    dt_0 = datetime.strptime(begin_time, '%Y-%m-%d %H:%M')
    for row_ind in range(len(patient_df_list[ind].patient_df)):
        measurement_name = patient_df_list[ind].patient_df['ANGABE'].iloc[row_ind].replace(' ', '')
        storage_ind = 1 + np.where(measurement_label_array == measurement_name)[0][0]  #
        measurement_reading = patient_df_list[ind].patient_df['WERT'].iloc[row_ind]
        if type(measurement_reading) == str:
            measurement_reading = measurement_reading.replace(',', '.')
            measurement_reading = measurement_reading.replace(' ', '')
            measurement_reading = measurement_reading.replace('>', '')
            measurement_reading = measurement_reading.replace('<', '')
        try:
            traj_entry = float(measurement_reading)
        except ValueError:
            # print('Value not accounted: ', measurement_name, measurement_reading)
            invalid_entry = measurement_name + ' ' + str(measurement_reading)
            if invalid_entry not in not_accounted_values:
                not_accounted_values.append(invalid_entry)
            traj_entry = float('nan')
        measurement_time = patient_df_list[ind].patient_df['EINTRAGUNG'].iloc[row_ind]
        dt_i = datetime.strptime(measurement_time, '%Y-%m-%d %H:%M')
        diff_min = (dt_i - dt_0).seconds / 60
        diff_d = (dt_i - dt_0).days
        total_diff = diff_d * 1440 + diff_min

        # insert into traj
        if not np.any(traj[:, 0] == total_diff):
            # new line
            new_line = np.ones(len(measurement_label_array) + 1) * float('nan')
            new_line[0] = total_diff
            new_line[storage_ind] = traj_entry
            traj = np.vstack((traj, new_line))
        else:
            # search for line
            time_ind = np.where(traj[:, 0] == total_diff)[0][0]
            traj[time_ind, storage_ind] = traj_entry
        # print(total_diff, storage_ind, traj_entry, np.shape(traj))
    print('patient ', ind, ' done ', valid_patient_code_list[ind])
    print(np.shape(traj), np.sum(traj == traj) / np.size(traj))
    # print(traj)
    valid_patient_traj_list.append(traj)

print(sorted(not_accounted_values))

with open('patient_traj.csv', 'w') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(np.hstack((np.array(['Code', 'Time step']), measurement_label_array)))
    for patient_ind in range(len(valid_patient_code_list)):
        for row_ind in range(np.shape(valid_patient_traj_list[patient_ind])[0]):
            line = np.hstack((patient_ind, valid_patient_traj_list[patient_ind][row_ind]))
            writer.writerow(line)
    f_out.close()


