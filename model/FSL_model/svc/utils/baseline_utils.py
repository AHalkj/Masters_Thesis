from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

def transform_x(x, vocabulary: list):
    '''
        input: x: word vector
               vocabulary: vocabulary created on training set
        output: feature array of size (len(x), len(vocabulary))
    '''
    tfIdfVectorizer = Pipeline([
        ('vect', CountVectorizer(vocabulary=vocabulary, lowercase=False)),
        ('tfidf', TfidfTransformer(norm='l1', use_idf=True))
    ])
    x_vectorized = tfIdfVectorizer.fit_transform(x)
    return x_vectorized.toarray()

def cTfIdfClassifier(vocabulary):
    pipeline = Pipeline([
        ('vect', CountVectorizer(vocabulary=vocabulary, lowercase=False)),
        ('tfidf', TfidfTransformer(norm='l1', use_idf='True')),
        ('SVC', SVC(**svm_best_parameters)),
    ])
    return pipeline

svm_best_parameters = {'C': 2, 'break_ties': False, 'cache_size': 200, 'class_weight': 'balanced', 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3,\
    'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'probability': True, 'random_state': None, 'shrinking': False, 'tol': 0.01, 'verbose': False}

rand_forest_best_parameters = {'bootstrap': False, 'ccp_alpha': 0.0, 'class_weight': {0: 1, 1: 8}, 'criterion': 'gini', 'max_depth': 15, 'max_features': 'sqrt',\
    'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 3, 'min_weight_fraction_leaf': 0.0,\
    'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}

extraTrees_best_parameters = {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': {0: 1, 1: 8}, 'criterion': 'entropy', 'max_depth': None,\
    'max_features': None, 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 2, 'min_samples_split': 4, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 100, 'n_jobs': None, 'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}

logistic_best_parameters = {'C': 0.5, 'class_weight': 'balanced_subsample', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': 0,\
    'max_iter': 100, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'none', 'random_state': None, 'solver': 'sag', 'tol': 0.0001, 'verbose': 0,\
    'warm_start': False}
