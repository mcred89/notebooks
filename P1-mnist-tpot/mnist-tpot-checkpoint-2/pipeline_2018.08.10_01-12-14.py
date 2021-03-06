import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:0.9895980512222403
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.9000000000000001, min_samples_leaf=5, min_samples_split=15, n_estimators=100)),
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.1, max_depth=7, max_features=0.6500000000000001, min_samples_leaf=12, min_samples_split=7, n_estimators=100, subsample=0.55)),
    StackingEstimator(estimator=GaussianNB()),
    StackingEstimator(estimator=BernoulliNB(alpha=0.001, fit_prior=True)),
    KNeighborsClassifier(n_neighbors=6, p=2, weights="distance")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
