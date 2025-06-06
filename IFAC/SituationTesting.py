# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
from copy import deepcopy
from .Rule import get_instances_covered_by_rule_base
from scipy.spatial.distance import cdist
from load_datasets import distance_function_income_pred

class SituationTesting:
    def __init__(self, reference_group_list, decision_label, desirable_label, k, t):
        self.reference_group_list = reference_group_list
        self.decision_label = decision_label
        self.desirable_label = desirable_label
        self.k = k
        self.t = t

    #the data argument that is passed here will be used for the kNN comparison
    def fit(self, data):
        #we need to divide the data into the instances that are part of the reference group, and the ones that are not
        relevant_data = deepcopy(data)
        self.all_reference_group_data = pd.DataFrame([])
        for reference_group in self.reference_group_list:
            reference_group_data = get_instances_covered_by_rule_base(reference_group, relevant_data)
            self.all_reference_group_data = pd.concat([self.all_reference_group_data, reference_group_data], axis=0)
            relevant_data = relevant_data.drop(reference_group_data.index)
        self.non_reference_group_data = relevant_data
        return


    def compute_k_nearest_neighbours_of_reference_and_non_reference(self, dataset):
        distance_matrix_to_non_reference = cdist(dataset, self.non_reference_group_data, metric=distance_function_income_pred)
        distance_df_to_non_reference = pd.DataFrame(distance_matrix_to_non_reference, index=dataset.index, columns=self.non_reference_group_data.index)

        # Find the k nearest neighbors of the non_reference_group for each index in the dataset
        nearest_non_reference_neighbors = distance_df_to_non_reference.apply(lambda row: row.nsmallest(self.k).index.tolist(), axis=1)

        nearest_non_reference_neighbors_df = pd.DataFrame(nearest_non_reference_neighbors.tolist(), index=dataset.index,
                                            columns=[f'Neighbor_{i + 1}' for i in range(self.k)])


        distance_matrix_to_reference = cdist(dataset, self.all_reference_group_data,
                                                 metric=distance_function_income_pred)
        distance_df_to_reference = pd.DataFrame(distance_matrix_to_reference, index=dataset.index,
                                                    columns=self.all_reference_group_data.index)

        # Find the k nearest neighbors of the reference group for each index in the dataset
        nearest_reference_neighbors = distance_df_to_reference.apply(lambda row: row.nsmallest(self.k).index.tolist(), axis=1)
        nearest_reference_neighbors_df = pd.DataFrame(nearest_reference_neighbors.tolist(), index=dataset.index,
                                                          columns=[f'Neighbor_{i + 1}' for i in range(self.k)])


        return nearest_non_reference_neighbors_df, nearest_reference_neighbors_df

    def positive_decision_ratio(self, data, neighbours_indices):
        decision_info_of_neighbours = data.loc[neighbours_indices, self.decision_label]
        positive_decision_count = (decision_info_of_neighbours == self.desirable_label).sum()  # Count the number of 'high' incomes
        return positive_decision_count/ len(neighbours_indices)  # Compute the ratio


    #return true if instance is being discriminated
    def predict(self, data):
        nearest_non_reference_neighbors_df, nearest_reference_neighbors_df = self.compute_k_nearest_neighbours_of_reference_and_non_reference(data)

        pos_ratio_non_reference_neighbours = nearest_non_reference_neighbors_df.apply(lambda row: self.positive_decision_ratio(self.non_reference_group_data, row), axis=1)
        pos_ratio_reference_neighbours = nearest_reference_neighbors_df.apply(lambda row: self.positive_decision_ratio(self.all_reference_group_data, row), axis=1)

        disc_scores = pos_ratio_reference_neighbours - pos_ratio_non_reference_neighbours
        disc_labels = disc_scores>self.t

        combined_situation_test_info_df = pd.DataFrame({
            'disc_score': disc_scores,
            'disc_label': disc_labels,
            'closest_non_reference': nearest_non_reference_neighbors_df.values.tolist(),
            'closest_reference': nearest_reference_neighbors_df.values.tolist()
        }, index=data.index)

        sit_test_info = combined_situation_test_info_df.apply(create_sit_test_info, axis=1)

        return disc_labels, sit_test_info



class SituationTestingInfo:

    def __init__(self, disc_score, discriminated_label, closest_non_reference, closest_reference):
        self.disc_score = disc_score
        self.discriminated_label = discriminated_label
        self.closest_non_reference = closest_non_reference
        self.closest_reference = closest_reference


    def __str__(self):
        str_repr = f"Disc Score: {self.disc_score:.2f}"
        str_repr += "\nClosest neighbours from reference group:\n"
        str_repr += str(self.closest_reference)
        str_repr += "\nClosest neighbours from non reference groups:\n"
        str_repr += str(self.closest_non_reference)
        return str_repr


def create_sit_test_info(row):
    return SituationTestingInfo(
        disc_score=row['disc_score'],
        discriminated_label=row['disc_label'],
        closest_non_reference=row['closest_non_reference'],
        closest_reference=row['closest_reference']
    )