from pandas import read_csv, DataFrame
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import StackingClassifier
from joblib import dump
import settings
import logging

## Import data
logging.info("Importing data...")
data_train = read_csv("data/data_train.csv")
data_test  = read_csv("data/data_test.csv")

## Encode output
logging.info("Encoding output...")
le = LabelEncoder()
le.fit(data_train.category.unique())
y_train = le.transform(data_train.category)
y_test  = le.transform(data_test.category)

## Compute document tf-idf
logging.info("Computing tf-idf...")
vectorizer = TfidfVectorizer(
    lowercase    = False, 
    max_df       = 0.5, 
    min_df       = 5, 
    ngram_range  = (1, 3), 
    norm         = None, 
    max_features = None, 
    sublinear_tf = False
)
X_train = vectorizer.fit_transform(data_train.text)
X_test  = vectorizer.transform(data_test.text)
logging.info("Xtrain: {}".format(X_train.shape))
logging.info("Xtest: {}".format(X_test.shape))

## Train SVM
logging.info("Fitting support vector classifier...")
svm = LinearSVC(
    C            = 0.5, 
    class_weight = "balanced",
    random_state = 0
)
glm = LogisticRegression(
    solver       = 'liblinear', 
    class_weight = "balanced", 
    max_iter     = 1500,
    random_state = 0
)
clf = StackingClassifier(
    estimators      = [('svm', svm)], 
    final_estimator = glm, 
    cv              = 5, 
    n_jobs          = 5
)
clf.fit(X_train, y_train)
dump(clf, 'output/svm_model.joblib')

## Test fitted model
logging.info("Predicting test set...")
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)
logging.info("Overall Accuracy: {:.2f}%".format(
    100 * metrics.accuracy_score(y_test, y_pred)
))
logging.info("Balanced Accuracy: {:.2f}%".format(
    100 * metrics.balanced_accuracy_score(y_test, y_pred)
))
logging.info("Micro F1-score: {:.2f}%".format(
    100 * metrics.f1_score(y_test, y_pred, average = "micro")
))
logging.info("Macro F1-score: {:.2f}%".format(
    100 * metrics.f1_score(y_test, y_pred, average = "macro")
))
logging.info("Log-loss: {:.5f}".format(
    metrics.log_loss(y_test, y_prob)
))

## Save predictions
logging.info("Persisting predictions on disk...")
col_names = []
labels = le.classes_
for i in range(len(labels)): 
    col_names.append("prob_{}".format(labels[i]))
data_pred = DataFrame(
    data    = y_prob,
    index   = range(y_prob.shape[0]),
    columns = col_names
)
data_pred["target"] = le.inverse_transform(y_test)
data_pred["pred"] = le.inverse_transform(y_pred)
data_pred.to_csv("output/svm_prediction.csv")
