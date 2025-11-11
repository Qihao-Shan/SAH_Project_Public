import numpy as np
from scipy.stats import kruskal


def convert_column(raw_clm, yes_value):
    if yes_value is not None:
        op_clm = np.array([])
        for i in range(np.size(raw_clm)):
            if type(raw_clm[i]) is str:
                if raw_clm[i].find(str(yes_value)) >= 0:  # search for yes value
                    added_value = 1
                else:
                    added_value = 0
            elif type(raw_clm[i]) is int or type(raw_clm[i]) is float:
                if raw_clm[i] == yes_value:
                    added_value = 1
                else:
                    added_value = 0
            else:
                added_value = 0  # assign no value
            op_clm = np.hstack((op_clm, added_value))
        return op_clm
    else:
        value_clm = np.array([])
        for i in range(np.size(raw_clm)):
            if type(raw_clm[i]) is int or type(raw_clm[i]) is float:
                value = raw_clm[i]
            else:
                value = float('nan')
            value_clm = np.hstack((value_clm, value))
        op_clm = np.zeros(np.size(value_clm))
        op_clm[value_clm > np.nanmedian(value_clm)] = 1
        return op_clm


def non_dom_sorting(mat):
    """
    :param mat: num samples * num features
    :return:
    """
    if mat.ndim == 1:
        return np.argsort(np.argsort(mat))
    elif mat.ndim == 2:
        num_samples = np.shape(mat)[0]
        front_no_array = -np.ones(num_samples)
        current_front = 0
        domination_count = np.zeros(num_samples)
        domination_mat = np.zeros((num_samples, num_samples))
        for i in range(np.shape(mat)[0]):
            for j in range(np.shape(mat)[0]):
                if np.all(mat[i] <= mat[j]) and not np.all(mat[i] == mat[j]):
                    domination_mat[i, j] = 1
                elif np.all(mat[i] >= mat[j]) and not np.all(mat[i] == mat[j]):
                    domination_count[i] += 1
            if domination_count[i] == 0:
                front_no_array[i] = 0
        ra = np.arange(num_samples)
        while np.size(front_no_array[front_no_array == current_front]) != 0:
            for i in ra[front_no_array == current_front]:
                domination_slice = domination_mat[i]
                for j in ra[domination_slice == 1]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        front_no_array[j] = current_front + 1
            current_front += 1
        return front_no_array
    else:
        print('to many dimensions for non dom sorting')
        return 0


class NonDomFeatureSet2:
    def __init__(self, feature_mat, response_array, feature_names):
        self.feature_mat = feature_mat  # num elements x num features
        self.response_array = response_array  # num elements
        self.num_elements = np.size(response_array)
        self.num_features = np.shape(feature_mat)[1]
        self.feature_under_consideration = np.ones(self.num_features)
        self.feature_ind_array = np.arange(self.num_features)
        self.feature_sign_array = np.ones(self.num_features)  # when not under consideration -> solid value; when under try pos neg
        self.feature_names = feature_names

    def feature_selection_objective_fn(self, features_chosen_current_round):
        criterion_array_pos = np.array([])
        criterion_array_neg = np.array([])
        feature_ind_under_consideration = self.feature_ind_array[self.feature_under_consideration == 1]
        sign_array_pos = np.copy(self.feature_sign_array)
        sign_array_neg = np.copy(self.feature_sign_array)
        sign_array_neg[self.feature_under_consideration == 1] = -1
        for next_feature_ind in feature_ind_under_consideration:
            features_chosen_current_round_ = np.copy(features_chosen_current_round)
            features_chosen_current_round_[next_feature_ind] = 1
            # pos
            sign_mat_pos = np.tile(sign_array_pos, (self.num_elements, 1))
            data_mat_pos = self.feature_mat * sign_mat_pos
            data_mat_pos = data_mat_pos[:, features_chosen_current_round_ == 1]
            ranking_pos = non_dom_sorting(data_mat_pos)
            res = kruskal(ranking_pos[self.response_array == 1], ranking_pos[self.response_array == 0])
            criterion_array_pos = np.hstack((criterion_array_pos, res.pvalue))
            # neg
            sign_mat_neg = np.tile(sign_array_neg, (self.num_elements, 1))
            data_mat_neg = self.feature_mat * sign_mat_neg
            data_mat_neg = data_mat_neg[:, features_chosen_current_round_ == 1]
            ranking_neg = non_dom_sorting(data_mat_neg)
            res = kruskal(ranking_neg[self.response_array == 1], ranking_neg[self.response_array == 0])
            criterion_array_neg = np.hstack((criterion_array_neg, res.pvalue))
        return criterion_array_pos, criterion_array_neg

    def feature_selection_round(self):
        features_chosen_this_round = np.zeros(self.num_features)
        last_criterion = float('nan')
        final_feature_set = np.array([])
        chosen_feature_sign = np.array([])
        ordered_feature_set = np.array([])
        ordered_sign_set = np.array([])
        stop_cond = False
        while not stop_cond:
            feature_ind_under_consideration = self.feature_ind_array[self.feature_under_consideration == 1]
            criterion_array_pos, criterion_array_neg = self.feature_selection_objective_fn(features_chosen_this_round)
            optimal_pos = np.min(criterion_array_pos)
            optimal_neg = np.min(criterion_array_neg)
            if last_criterion == np.nanmin(np.array([optimal_pos, optimal_neg])):
                final_feature_set = self.feature_ind_array[features_chosen_this_round == 1]
                chosen_feature_sign = self.feature_sign_array[features_chosen_this_round == 1]
                # print('Feature set: ', final_feature_set, chosen_feature_sign)
                print('Ordered FS: ', np.array2string(ordered_feature_set, separator=', '),
                      np.array2string(ordered_sign_set, separator=', '))
                stop_cond = True
            elif optimal_pos <= optimal_neg:
                new_feature = feature_ind_under_consideration[np.argmin(criterion_array_pos)]
                print('Choose feature: ', new_feature, ' pos ', self.feature_names[new_feature], ' p=', np.min(criterion_array_pos))
                features_chosen_this_round[new_feature] = 1
                self.feature_under_consideration[new_feature] = 0
                last_criterion = optimal_pos
                ordered_feature_set = np.hstack((ordered_feature_set, new_feature))
                ordered_sign_set = np.hstack((ordered_sign_set, 1))
            else:
                new_feature = feature_ind_under_consideration[np.argmin(criterion_array_neg)]
                print('Choose feature: ', new_feature, ' neg ', self.feature_names[new_feature], ' p=', np.min(criterion_array_neg))
                features_chosen_this_round[new_feature] = 1
                self.feature_under_consideration[new_feature] = 0
                self.feature_sign_array[new_feature] = -1
                last_criterion = optimal_neg
                ordered_feature_set = np.hstack((ordered_feature_set, new_feature))
                ordered_sign_set = np.hstack((ordered_sign_set, -1))
            # print(last_criterion)
        return final_feature_set, chosen_feature_sign


def produce_ranking_array_from_feature_set(full_data, feature_set, sign_array):
    cropped_data_mat = np.array([]).reshape((-1, np.shape(full_data)[0]))
    for ind in range(np.size(feature_set)):
        cropped_data_mat = np.vstack(
            (cropped_data_mat, sign_array[ind] * full_data[:, feature_set[ind]]))
    cropped_data_mat = cropped_data_mat.T
    ranking_array = non_dom_sorting(cropped_data_mat)
    return ranking_array





