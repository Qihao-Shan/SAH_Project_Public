import numpy as np
import pandas as pd
import data_processing


raw_data_df = pd.read_excel(io='SAB_04_19_bis_09_24_Multiobjective.xlsx', skiprows=2)
print(raw_data_df.keys())
print(raw_data_df['Code'].to_numpy())

considered_feature_n_yes_value = [['Age at diagnosis', None], ['Hypertension', 1], ['Diabetes', 1], ['Diabetes', 2],
                                  ['Hyperchol.', 1],
                                  ['pAVK', 1], ['Heart diseases', 1], ['Heart diseases', 2], ['Heart diseases', 3],
                                  ['Heart diseases', 4], ['Heart diseases', 5], ['Previous Stroke', 1],
                                  ['Smoking', 1], ['Alcohol', 1], ['Thrombosis', 1], ['Hormonal contraception', 1],
                                  ['Tumor disease', 1], ['Obesity', 1], ['Autoimmune diseases', 1],
                                  ['prior to admission', 1], ['prior to admission', 2], ['prior to admission', 3],
                                  ['prior to admission', 4], ['prior to admission', 5], ['prior to admission', 6],
                                  ['prior to admission', 7], ['Type of bleeding', 1], ['Type of bleeding', 2],
                                  ['Type of bleeding', 3], ['Type of bleeding', 4], ['Localisation of ICH', 1],
                                  ['Localisation of ICH', 2], ['Localisation of ICH', 3], ['Localisation of ICH', 4],
                                  ['Localisation of ICH', 5], ['Localisation of ICH', 6],
                                  ['MLV', 1], ['IVH ', 1], ['HCP', 1], ['Ischemia', 1], ['Multiplicity', 1],
                                  ['Localisation', 1], ['Localisation', 2], ['Localisation', 3], ['Localisation', 4],
                                  ['Localisation', 6], ['Localisation', 7], ['Localisation', 8], ['Localisation', 9],
                                  ['Localisation', 10], ['Localisation', 11], ['H&H', None], ['GCS', None], ['WFNS', None],
                                  ['Fisher', None], ['Type', 1], ['Type', 2], ['Intraoperative rupture', 1],
                                  ['temp. Clipping', 1], ['ICH', 1], ['Hemicraniectomy', 1], ['Hemicraniectomy', 2],
                                  ['Postoperative Ischemia', 1], ['Postoperative Rebleeding ', 1], ['Complications', 1],
                                  ['> symptomatic', 1], ['Ischemia CT', 1], ['Ischemia MRI', 1], ['HCP.1', 1], ['EVD', 1]]
feature_names = []
processed_mat = np.array([]).reshape((0, len(raw_data_df.index)))

for i in range(len(considered_feature_n_yes_value)):
    name = considered_feature_n_yes_value[i][0]
    yes_value = considered_feature_n_yes_value[i][1]
    feature_names.append(name + '_' + str(yes_value))
    raw_clm = raw_data_df[name].to_numpy()
    processed_mat = np.vstack((processed_mat, data_processing.convert_column(raw_clm, yes_value)))
processed_mat = processed_mat.T

print('Feature names: ', feature_names)
print(np.shape(processed_mat), 'num patients x num features')
print(np.sum(processed_mat), 'this should be a real number so no nan value')
num_patients = np.shape(processed_mat)[0]


# Read motif features
motif_length = 12
motif_feature_df = pd.read_csv('new_criteria_from_motifs.csv')
motif_feature_np = motif_feature_df.to_numpy()
motif_feature_numbers = motif_feature_np[:, 2 + motif_length:]
motif_keys = motif_feature_df.keys().tolist()
patient_ind_string_list = motif_keys[2 + motif_length:]
patient_ind_array = np.array([])

for i in range(len(patient_ind_string_list)):
    patient_ind_array = np.hstack((patient_ind_array, float(patient_ind_string_list[i][12:])))
print('Patient array, ', patient_ind_array)

# 182 -> 175 -> 156
fst_round_patient_selection = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
                                        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

snd_round_patient_selection = np.zeros(int(np.sum(fst_round_patient_selection)))
for i in patient_ind_array:
    snd_round_patient_selection[int(i)] = 1
total_patient_selection = np.copy(fst_round_patient_selection)
total_patient_selection[total_patient_selection == 1] = snd_round_patient_selection
print('Final patients: ', np.sum(total_patient_selection))

# concat behind processed_mat
for i in range(np.shape(motif_feature_numbers)[0]):
    clm = np.zeros(num_patients)
    motif_feature_slice = motif_feature_numbers[i, :]
    clm[total_patient_selection == 1] = motif_feature_slice
    processed_mat = np.hstack((processed_mat, clm.reshape((-1, 1))))
    feature_names.append('Motif_' + str(int(motif_feature_np[i, 0])) + '_' + str(int(motif_feature_np[i, 1])))

print('Feature names with motifs: ', feature_names)
print(np.shape(processed_mat), 'num patients x num features')
print(np.sum(processed_mat), 'this should be a real number so no nan value')

response_name = 'CVS'
response_array = data_processing.convert_column(raw_data_df[response_name].to_numpy(), yes_value=3)
print('Response ', response_array)

np.random.seed(10)
feature_set_obj = data_processing.NonDomFeatureSet2(processed_mat, response_array, feature_names)
std_array = np.std(processed_mat, axis=0)
feature_set_obj.feature_under_consideration[std_array == 0] = 0
feature_set_1, sign_set_1 = feature_set_obj.feature_selection_round()
feature_set_1_names = []
for ind in feature_set_1:
    feature_set_1_names.append(feature_names[ind])
print(feature_set_1_names, sign_set_1)

feature_set_2, sign_set_2 = feature_set_obj.feature_selection_round()
feature_set_2_names = []
for ind in feature_set_2:
    feature_set_2_names.append(feature_names[ind])
print(feature_set_2_names, sign_set_2)

