# IFAC

## An Interpretable Fair Abstaining Classifier

This is the implementation of IFAC, an Interpretable Fair Abstaining Classifier as firstly introduced in:

- Lenders, D., Pugnana, A., Pellungrini, R., Calders, T., Pedreschi, D., & Giannotti, F. (2024, August). Interpretable and Fair Mechanisms for Abstaining Classifiers. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 416-433).

Like other abstaining classifiers IFAC rejects the predictions of a base classifier. However, unlike other abstaining classifiers, it doesn’t only base its rejections on the uncertainty of their predictions, but also their unfairness. The unfairness of a base-classifiers predictions are assessed through explainable-by-design methods of discriminatory association rules and situation testing.

With the first, it is assessed whether an instance belongs to any at-risk groups in the data, while with the latter the individual fairness of a prediction is assessed.

Depending on whether the predictions of a base classifier are unfair/fair and uncertain/certain, there are four possible scenarios IFAC considers:

- **Fair & Certain**: There are no reasons to doubt the original decision, hence it is *kept*
- **Fair & Uncertain**: These decisions are *rejected* for uncertainty
- **Unfair & Certain**: These decisions are *rejected* for unfairness
- **Unfair & Uncertain**: Since there is double reason to doubt the original decision, these decisions are *flipped*

Whenever a prediction is rejected or flipped because of unfairness, the outcome of both fairness analyses are outputted by IFAC.

## Minimal Working Example
The code below shows how to apply IFAC on an income prediction task. Note that *load_income_data()* returns an object of the *Dataset* class, as defined in the project. 
In this class, information on the decision attribute of a dataset as well as its desirable and non-desirable label are encoded. Further, object of this class specify which attributes in the data are considered as sensitive (e.g. *race* and *sex*), and which sensitive attribute values correspond to possibly *favoured* groups (e.g. *race: White, sex: Male*) and possibly *discriminated* groups (e.g. *race: Black, sex: Male*). 


```sh
#Imports
from load_datasets import load_income_data  
from IFAC import IFAC

#Load the data
income_prediction_data = load_income_data()

#Split into train test set
train, test = income_prediction_data.split_into_train_test(test_fraction=2000)

#Initialize IFAC
ifac = IFAC(coverage=0.8, fairness_weight=1.0, val1_ratio=0.2, 
val2_ratio=0.2, base_classifier='Random Forest')

#Fit on the train data
ifac.fit(train)

#Predict test data
predictions, information_flipped_instances = ifac.predict(test)

```

Whenever IFAC rejects a prediction of the base classifier, it outputs an instance of the *Reject* class. Depending on whether rejections were made out of uncertainty or unfairness concerns, different informations is encoded in these instances. 

```sh  
for prediction in predictions:  
    if isinstance(prediction, Reject):  
        print(prediction)
```

### Example of an Uncertainty-Based Reject
**Uncertain Reject-for this instance**
{'age': '40-49', 'marital status': 'Married', 'education': 'Associate Degree', 'workinghours': '40-49', 'workclass': 'private', 'occupation': 'Repair/Maintenance', 'race': 'Black or African American alone', 'sex': 'Male'}

**Prediction that would have been made:** low
**Prediction Probability:** 0.503


### Example of an Unfairness-Based Reject
**Unfairness Reject-for this instance**
{'age': '50-59', 'marital status': 'Married', 'education': 'Started College, No Diploma', 'workinghours': '40-49', 'workclass': 'governmental', 'occupation': 'Office/Administrative Support', 'race': 'White alone', 'sex': 'Female'}

**Prediction that would have been made:** low
**Prediction Probability:** 0.717
**Rejection Based on this Discriminatory Pattern**
(education = Started College, No Diploma AND sex = Female AND age = 50-59) -> (income = low), Support: 0.021, Confidence: 0.899, Lift: 0.000, SLift: 0.525
**Situation Testing Score:** 0.90
**Closest neighbours from favoured group:**
[216, 1039, 2585, 515, 2027, 784, 1089, 1307, 1311, 1320]
**Closest neighbours from non favoured groups:**
[1249, 2281, 2507, 1156, 1737, 2216, 455, 2029, 2014, 2541]



## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
