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

from load_datasets import load_income_data
from IFAC import IFAC
from IFAC.Reject import Reject

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    income_prediction_data = load_income_data()
    train, test = income_prediction_data.split_into_train_test(test_fraction=2000)

    ifac = IFAC(coverage=0.8, fairness_weight=1.0, val1_ratio=0.2, val2_ratio=0.2, base_classifier='Random Forest')

    ifac.fit(train)

    # predictions is an array containing either the prediction label (income = 'high'/'low')
    # or an instance of a "Reject" object. A "Reject" object stores the reason for rejecting
    # (unfairness/uncertainty), the original prediciton and prediction probability, and in
    # case of an unfairness-based reject the explanation behind it (which discriminatory
    # rule the rejection was based upon + Situation Testing information)
    predictions, information_flipped_instances = ifac.predict(test)

    for flip in information_flipped_instances:
        print(flip)

    for prediction in predictions:
        if isinstance(prediction, Reject):
            print(prediction)
