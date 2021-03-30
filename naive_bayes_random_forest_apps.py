from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

# naive bayes is naive because it has no sense of the ordering of a dataset. In some scenarios,
# the ordering of things are important.

# Feature subset selection via Random Forest relevant features
X_train_RF1 = df.drop('Classification', axis=1)         # Select features for X
y_train_RF1 = df['Classification']                      # Select classification for y
X_train_RF1 = df.drop(['glucose_range', 'insulin_range'], axis=1)

rf = RandomForestClassifier(random_state=1)  # build initial random forest classifier using GINI
rf.fit(X_train_RF1, y_train_RF1)                                        # fit to data

rf_cv_score = cross_val_score(rf, X_train_RF1, y_train_RF1, cv=5)        #5-fold cross validation
print("Initial Random Forest 5-Fold cross-vadiation score average: %0.3f" % (rf_cv_score.mean()))
print("Random Forest determined relevant features:")
print(X_train_RF1.columns)
print(rf.feature_importances_)

print('\n')
print(rf.base_estimator_)
print('\n')
print(rf.classes_)

# Feature subset selection via filter method
X_train = df.drop(['Classification', 'Insulin', 'Leptin', 'Adiponectin',
                  'MCP.1', 'BMI', 'Age'], axis=1)  # Select features for X
y_train = df['Classification']                      # Select classification for y

print(X_train.columns)