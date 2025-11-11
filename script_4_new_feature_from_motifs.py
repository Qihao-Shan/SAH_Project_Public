import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_processing
import csv

motif_mat_df = pd.read_csv('motif_data.csv')
motif_mat_np = motif_mat_df.to_numpy()
motif_length = 12
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
print(np.shape(motif_mat_np))

included_patients = np.sort(np.unique(motif_mat_np[:, 0]))
motif_mat_sum = np.sum(motif_mat_np, axis=1)
motif_mat_3d = motif_mat_np.reshape((-1, 86, motif_length + 2))
not_nan_mat = np.reshape(motif_mat_sum == motif_mat_sum, (-1, 86))
num_patients = np.shape(not_nan_mat)[0]
motif_mat_trimmed = motif_mat_np[motif_mat_sum == motif_mat_sum, :]
dim_array = motif_mat_trimmed[:, 1]
num_motifs = np.shape(motif_mat_trimmed)[0]
print(np.shape(motif_mat_trimmed))
print(motif_mat_trimmed)

# every motif is a new feature, look for similarity in other patients
sim_mat = np.ones((num_motifs, num_patients)) * float('nan')

# shift the original motif to account for cycles
for motif_ind in range(num_motifs):
    # print('Doing motif ', motif_ind, end='\r')
    base_patient = int(motif_mat_trimmed[motif_ind, 0])
    dim_investigated = int(motif_mat_trimmed[motif_ind, 1])
    motif = motif_mat_trimmed[motif_ind, 2:]
    motif_concat = np.hstack((motif, motif))
    fst_motif_mat = np.array([]).reshape((-1, motif_length))
    for i in range(motif_length):
        fst_motif_mat = np.vstack((fst_motif_mat, motif_concat[i:i+motif_length]))
    # print(fst_motif_mat)
    for snd_patient_ind in range(num_patients):
        if not_nan_mat[snd_patient_ind, dim_investigated] == not_nan_mat[snd_patient_ind, dim_investigated]:
            snd_motif = motif_mat_3d[snd_patient_ind, dim_investigated, 2:]
            # print(fst_motif_mat, np.shape(snd_motif))
            snd_motif_mat = np.tile(snd_motif, (motif_length, 1))
            dist_array = np.sqrt(np.sum((fst_motif_mat - snd_motif_mat)**2, axis=1))
            # print(np.shape(dist_array))
            dist = np.min(dist_array)
            sim_mat[motif_ind, snd_patient_ind] = dist

std_sim = np.nanstd(sim_mat, axis=1)
not_nan_sim = np.sum(sim_mat == sim_mat, axis=1)
print(std_sim)
print(not_nan_sim)

combined_mat = np.vstack((std_sim, not_nan_sim)).T
ranking = data_processing.non_dom_sorting(-combined_mat)
print(ranking)
cutoff_rank = 100

fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(std_sim, not_nan_sim, marker='x')
plt.scatter(std_sim[ranking < cutoff_rank], not_nan_sim[ranking < cutoff_rank], marker='x', c='r')
plt.xlabel('StD of Euclidean distances to other motifs')
plt.ylabel('Number of valid data entries')
for i in range(np.size(std_sim)):
    if ranking[i] < cutoff_rank:
        # ax.annotate(str(i), xy=(std_sim[i], not_nan_sim[i]))
        print(i, motif_mat_trimmed[i, :], std_sim[i], not_nan_sim[i])
plt.savefig('MotifSelection.png', dpi=300)
plt.show()


header = ['Base Patient Ind', 'Base Dim']
for i in range(motif_length):
    header.append('Motif ' + str(i))
for i in range(num_patients):
    header.append('Sim Patient ' + str(included_patients[i]))

f1 = open('new_criteria_from_motifs.csv', 'w')
writer = csv.writer(f1)
writer.writerow(header)
motif_mat_printing = np.array([]).reshape((-1, num_patients))

for i in range(np.size(std_sim)):
    if ranking[i] < cutoff_rank:
        sim_mat_slice = sim_mat[i, :]
        print(sim_mat_slice)
        row_1 = motif_mat_trimmed[i, :]
        row_2 = np.zeros(num_patients)
        row_2[sim_mat_slice < np.nanmedian(sim_mat_slice)] = 1
        row = np.hstack((row_1, row_2))
        writer.writerow(row)
f1.close()



