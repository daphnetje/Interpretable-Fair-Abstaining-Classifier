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

from Dataset import Dataset
import pandas as pd

def load_income_data():
    raw_data = pd.read_csv('data/income_sample.csv')
    descriptive_dataframe = raw_data[
        ['age', 'marital status', 'education', 'workinghours', 'workclass', 'occupation', 'race', 'sex', 'income']]

    age_dict = {"Younger than 25": 1, "25-29": 2, "30-39": 3, "40-49": 4, "50-59": 5, "60-69": 6, "Older than 70": 7}
    education_dict = {"No Elementary School": 1, "Elementary School": 2, "Middle School": 3,
                      "Started High School, No Diploma": 4, "High School or GED Diploma": 5,
                      "Started College, No Diploma": 6, "Associate Degree": 7, "Bachelor Degree": 8,
                      "Master or other Degree Beyond Bachelor": 9, "Doctorate Degree": 10}
    workinghours_dict = {"Less than 20": 1, "20-39": 2, "40-49": 3, "More than 50": 4}
    dicts_ordinal_to_numeric = {'age': age_dict, 'education': education_dict, 'workinghours': workinghours_dict}

    categorical_features = ['marital status', 'occupation', 'workclass', 'race', 'sex']

    sensitive_attributes = ['sex', 'race']
    reference_group_list = [{'sex': 'Male', 'race': 'White alone'}]

    dataset = Dataset(descriptive_dataframe, dicts_ordinal_to_numeric, decision_attribute="income", undesirable_label="low",
                      desirable_label="high", sensitive_attributes=sensitive_attributes, reference_group_list=reference_group_list, categorical_features=categorical_features,
                      distance_function=distance_function_income_pred)

    return dataset

#order of features: ['age_num', 'marital status', 'education_num', 'workinghours_num', 'workclass', 'occupation', 'race', 'sex', 'income']]
def distance_function_income_pred(x1, x2):
    age_dict = {"Younger than 25": 1, "25-29": 2, "30-39": 3, "40-49": 4, "50-59": 5, "60-69": 6, "Older than 70": 7}
    age_diff = abs(age_dict[x1[0]] - age_dict[x2[0]]) / 6

    if x1[1] == x2[1]:
        marital_status_diff = 0
    else:
        marital_status_diff = 0.5

    education_dict = {"No Elementary School":1, "Elementary School":2, "Middle School":3,
                            "Started High School, No Diploma":4, "High School or GED Diploma":5,
                            "Started College, No Diploma":6, "Associate Degree":7, "Bachelor Degree":8,
                            "Master or other Degree Beyond Bachelor":9, "Doctorate Degree":10}
    education_diff = abs(education_dict[x1[2]] - education_dict[x2[2]]) / 9

    workinghours_dict = {"Less than 20": 1, "20-39": 2, "40-49": 3, "More than 50": 4}
    workinghours_diff = abs(workinghours_dict[x1[3]] - workinghours_dict[x2[3]])/3

    if x1[4] == x2[4]:
        workclass_diff = 0
    else:
        workclass_diff = 0.5

    if x1[5] == x2[5]:
        occupation_diff = 0
    else:
        occupation_diff = 0.5

    return age_diff + marital_status_diff + education_diff + workinghours_diff + workclass_diff + occupation_diff
