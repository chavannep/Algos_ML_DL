# %pylab inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, multilabel_confusion_matrix, accuracy_score, precision_score, f1_score, recall_score
from sklearn.pipeline import Pipeline

from plotDecisionBoundary import plot_decision_boundary

##################
# Loading data
##################
# Feature : nb_variables
# Centers : nb_classes
nb_class = 3
# Label=class=target in last column
data, target = make_blobs(n_features=2, centers=nb_class, random_state=1)
# Automatic feature naming if no csv loaded
columns_name = []
for i in range(data.shape[1]):
    columns_name.append('Feature_%d'%i)
data = pd.DataFrame(data, columns=columns_name)
target = pd.Series(target, name='Target')
df = pd.concat([data,target], axis=1, join='inner')

df.describe(percentiles=[0.25, 0.50, 0.75])


##################
# Data viz
##################
var_0 = data.iloc[:, 0]
var_1 = data.iloc[:, 1]
fig, ax = plt.subplots()
scatter = ax.scatter(var_0, var_1, marker="o", c=target.iloc[:], 
            cmap=plt.cm.coolwarm, edgecolor="k")
# plt.xlim(min(var_0), max(var_0))
# plt.ylim(min(var_1), max(var_1))
plt.xlabel(var_0.name)
plt.ylabel(var_1.name)
plt.title("Data viz over two chosen features")
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="best", title="Labels")
ax.add_artist(legend1)
plt.show()

###############
# Loading and splitting data into training/test sets
###############
# Normalization occurs later in make_pipeline(StandardScaler(),...)
X_train, X_test, y_train, y_test = train_test_split( 
    data, target, test_size=0.3, shuffle=True, random_state=1)


###############
# Splitting previous training data into k-1 sets (2nd training) and one CV set
############### 
kf = KFold(n_splits=5, shuffle=True, random_state=1)
kf_indices = list(kf.split(X_train))
# kf.split returns an iterator ((that would be consumed after one iteration). 

# for i, (train_indices, cv_indices) in enumerate(kf_indices):
#     print("Training data for fold %d:" % i, train_indices)
#     print("CV data for fold %d:" % i, cv_indices)


###############
# Model cross-validation
###############
neighbors = [3, 5, 7]

model = Pipeline([('scaler', StandardScaler()), 
                  ('KNN', KNeighborsClassifier())])

grid_search = GridSearchCV(model,
                    param_grid={'KNN__n_neighbors': neighbors},
                    cv=kf_indices,
                    scoring='accuracy', # 'f1' if binary, 'accuracy' if multi
                    n_jobs=2, return_train_score=True)

##################
# Training model
##################
grid_search.fit(X_train, y_train)

print("Best hyperparameter(s) on the CV set:", grid_search.best_params_)
print("Best estimator on the CV set:", grid_search.best_estimator_ )
print("Best score on the CV set:", grid_search.best_score_ )

###############
# Cross-validation : grid search evaluation
###############  
# Same changing _test_ -> _train_
param_1_name = pd.DataFrame(grid_search.cv_results_['params']).iloc[:,0].name
param_1_values = pd.DataFrame(grid_search.cv_results_['params']).iloc[:,0]
cv_param = pd.DataFrame({param_1_name : param_1_values,
        'mean_test_score' : grid_search.cv_results_['mean_test_score'],
        'std_test_score' : grid_search.cv_results_['std_test_score']}
                         )

plt.errorbar(cv_param[param_1_name], cv_param['mean_test_score'],
             yerr=cv_param['std_test_score'])
# plt.xlim(min(cv_param[param_1_name]), max(cv_param[param_1_name]))
# plt.ylim((min(cv_param['mean_test_score']), max(cv_param['mean_test_score'])))
plt.ylabel("Accuracy score to maximize (-)")
plt.xlabel("Nb estimators")
plt.title("Testing error obtained by cross-validation")
plt.show()

#############
# Learning curve : influence of the training set size
#############

# Compute the learning curve for a decision tree and vary the proportion of 
# the training set from 10% to 100%.
train_sizes = np.linspace(0.1, 1.0, num=5, endpoint=True)

# Use a ShuffleSplit cross-validation to assess our predictive model.
# cv = ShuffleSplit(n_splits=30, test_size=0.2)

results = learning_curve(
    grid_search.best_estimator_, X_train, y_train, train_sizes=train_sizes, cv=kf_indices,
    scoring='accuracy', n_jobs=2)
train_size, train_scores, test_scores = results[:3]
train_errors, test_errors = train_scores, test_scores
     
fig, ax = plt.subplots()
plt.errorbar(train_size, train_errors.mean(axis=1),
             yerr=train_errors.std(axis=1), label="Train set error")
plt.errorbar(train_size, test_errors.mean(axis=1),
             yerr=test_errors.std(axis=1), label="CV set error")
plt.legend()
plt.xscale("log")
plt.xlabel("Number of samples in the training set")
plt.ylabel("Accuracy score (-)")
plt.title("Learning curve : assessing influence of training set size")
plt.show()

# Training error : if error very small, then the trained model is overfitting the training data.

# Testing error alone : the more samples in training set, the lower the testing error. 
# We are searching for the plateau of the testing error for which there is no benefit to adding samples anymore 

# If already on a plateau and adding new samples in the training set does not reduce testing error, 
# Bayes error rate may be reached using the available model. 
# Using a more complex model might be the only possibility to reduce the testing error further.


##################
# Model prediction
##################
y_pred = grid_search.predict(X_test)

##################
# Decision boundary
##################
plot = plot_decision_boundary(model = grid_search, 
                                data_0_test = X_test[var_0.name], 
                                data_1_test = X_test[var_1.name], 
                                target_test = y_test)

##################
# Performance generalization
##################
# Number of correct answers
# print('Nb of test set elements correctly predicted : {:.2f}'.format(accuracy_score(y_test, y_pred, normalize=False)))
print('Accuracy score on test set : {:.2f}'.format(accuracy_score(y_test, y_pred, normalize=True)))


# average='micro' : global computation over classes. For ex. F1 : \
# Calculate metrics globally by counting the total true positives, false negatives and false positives.
# average='macro' : It calculates metrics for each class individually and then takes unweighted mean of the measures.
# average='weighted' : The weights for each class are the total number of samples of that class. 

print('F1 score (macro) : {:.2f}'.format(f1_score(y_test, y_pred, average='macro')))
print('Precision score (macro) : {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
print('Recall score (macro) : {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))




# Confusion matrix (binary classification)
# True negative  : CM_00
# False negative : CM_10
# True positive  : CM_11
# False positive : CM_01

# Proportion of cases (normalized by nb of param in Y_test)
print('Confusion matrix (nb of elem for each label)')
print(multilabel_confusion_matrix(y_test, y_pred))
print('Classification report to summarize') 
print(classification_report(y_test, y_pred))




	



