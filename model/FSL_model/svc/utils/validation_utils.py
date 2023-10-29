from  sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.utils import resample
import numpy as np
import time
from functools import wraps

def bootstrap_clf(X_test_vec, y_test_np, clf):
    '''
        Input: X_test_vec: Vectorized test data (n x m)
               y_test_np: labels (list or array)
               clf: classifier object with predict function
        Output: average_precision for 100 resampled test_sets
    '''
    avg_precs = []
    for i in range(100):
        boot, y_test = resample(X_test_vec,y_test_np, replace=True,n_samples = int(X_test_vec.shape[0]/2))
        y_pred = clf.predict(boot)
        avg_prec = average_precision_score(y_test, y_pred)
        avg_precs.append(avg_prec)
    return np.array(avg_precs)

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

def plot_prec_rec_curve(y_proba, y_test):
    y_proba = np.array(y_proba)
    y_test = np.array(y_test)
    precs = []
    recs = []
    # thresholds = np.arange(0,1,0.01)
    thresholds = np.concatenate([np.arange(0,0.05,0.0001),np.arange(0.05,0.1,0.005), np.arange(0.1,1,0.01)])
    for thr in thresholds: 
        y_pred =  1*(y_proba > thr)
        print(y_test)
        print(y_pred)
        prec, rec = precision_score(y_test, y_pred), recall_score(y_test, y_pred)
        precs.append(prec), recs.append(rec)
    return recs, precs

