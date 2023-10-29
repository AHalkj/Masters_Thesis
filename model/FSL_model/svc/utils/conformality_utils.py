import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import time

import torch

import sys
sys.path.append('../../')
from svc.utils.baseline_utils import transform_x
from svc.utils.text_processing import create_vocabulary, stemming_tokenizer
from svc.utils.baseline_utils import svm_best_parameters
from svc.utils.nn_utils import cnn_predict        , initialize_model, get_device, best_params


# ------------ conformal class average ----------------------------------


def find_pos_neg_distances(x_pos_avg, x_neg_avg, x):
    '''
        input: x_pos_avg, x_neg_avg: class averages, feature vectors of size len(vocabulary)
               x: matrix of feature vectors 
    '''
    # compute distance to class averages
    distances_neg = 1 - cosine_similarity(x, x_neg_avg.reshape(1, -1))
    distances_pos = 1 - cosine_similarity(x, x_pos_avg.reshape(1, -1))
    return distances_neg, distances_pos


def find_average_classes(x_train, y_train, vocabulary: list):
    # find vector embeddings:
    x_vectorized = transform_x(x_train, vocabulary)

    x_count_pos = x_vectorized[[idx for idx,
                                e in enumerate(y_train) if e == 1], :]
    x_count_neg = x_vectorized[[idx for idx,
                                e in enumerate(y_train) if e == 0], :]

    # average vectors for negative and positive class
    x_neg_avg = np.sum(x_count_neg, axis=0)/x_count_neg.shape[0]
    x_pos_avg = np.sum(x_count_pos, axis=0)/x_count_pos.shape[0]
    return x_neg_avg / np.linalg.norm(x_neg_avg), x_pos_avg / np.linalg.norm(x_pos_avg)


def distance_to_average(x_train, vocabulary: list, x_neg_avg, x_pos_avg, option: str):
    '''
        input: sentences and vocabulary -> get transformed into feature vectors
               class averages calculated on the training set
        output: distances of size (len(x_train),2) to both classes
    '''
    x_vectorized = transform_x(x_train, vocabulary)

    distance_to_negative_class, distance_to_positive_class = find_pos_neg_distances(
        x_pos_avg, x_neg_avg, x_vectorized)

    if option == 'preclassify_zeros':
        print('Preclassifiy Zero-Vectors')
        zero_count = 0
        for i in range(x_vectorized.shape[0]):
            if np.sum(np.abs(x_vectorized[i, :])) == 0:
                zero_count = zero_count + 1
                distance_to_negative_class[i] = 0
                distance_to_positive_class[i] = 1

    distances = np.concatenate(
        [distance_to_negative_class, distance_to_positive_class], axis=1)
    return distances


# ----------- conformal scores clostest 3 ---------------


def get_x_counts(x, y, vocabulary: list):
    x_vectorized_test = transform_x(x, vocabulary)
    x_count_pos = x_vectorized_test[[
        idx for idx, e in enumerate(y) if e == 1], :]
    x_count_neg = x_vectorized_test[[
        idx for idx, e in enumerate(y) if e == 0], :]
    return x_count_pos, x_count_neg, x_vectorized_test


def avg_closest_distance(x_count_pos, x_count_neg, x_test, y_test,  option: str):
    pos_dist_test = pairwise_distances(x_count_pos, x_test, metric='cosine')
    neg_dist_test = pairwise_distances(x_count_neg, x_test, metric='cosine')

    pos_dist_test.sort(axis=0)
    neg_dist_test.sort(axis=0)
    if option == 'train': # we need to discard the first one because it's going to be itself in this case
        avg_dist_pos = np.average(pos_dist_test[1:4, :], axis=0)
        avg_dist_neg = np.average(neg_dist_test[1:4, :], axis=0)

        return np.concatenate([avg_dist_neg[:, np.newaxis], avg_dist_pos[:, np.newaxis]], axis=1)
    if option == 'test':
        avg_dist_pos = np.average(pos_dist_test[:3, :], axis=0)
        avg_dist_neg = np.average(neg_dist_test[:3, :], axis=0)
        return np.concatenate([avg_dist_neg[:, np.newaxis], avg_dist_pos[:, np.newaxis]], axis=1)


def confidence_scores(df_test, df_train, **kwargs):
    t0 = time.time()
    x_test = pd.Series(df_test.text)
    x_train = pd.Series(df_train.text)
    y_test = df_test.label.astype(int).to_list()
    y_train = df_train.label.astype(int).to_list()
    vocabulary = create_vocabulary(x_train)
    if kwargs.get('scoring_function').lower() == 'nearestneighbors':
        print('Scoring Function: Closest 3 neighbors')

        x_test = transform_x(x_test, vocabulary)
        x_count_pos, x_count_neg, x_train = get_x_counts(
            x_train, y_train, vocabulary)

        distances_conf = avg_closest_distance(
            x_count_pos, x_count_neg, x_train, y_train, 'train')

        distances_test = avg_closest_distance(
            x_count_pos, x_count_neg, x_test, y_test, 'test')

    if kwargs.get('scoring_function').lower() == 'classaverage':
        print('Scoring function: Distance to class average')

        x_neg_avg_train, x_pos_avg_train = find_average_classes(
            x_train, y_train, vocabulary)

        # percentiles = calculate_percentile(X_test, y_test,X_train, y_train, vocabulary, x_neg_avg_train, x_pos_avg_train)
        distances_conf = distance_to_average(
            x_train,  vocabulary, x_neg_avg_train, x_pos_avg_train, kwargs.get('preclassification', 'preclassify_zeros'))

        distances_test = distance_to_average(
            x_test,  vocabulary, x_neg_avg_train, x_pos_avg_train, kwargs.get('preclassification', 'preclassify_zeros'))

    if kwargs.get('scoring_function').lower() == 'predict_probas':
        print('Scoring function: Sklearn Predict Probabilities') 
        # create validation set. Needs to be from training set (len of test cannot change)
        # Uspampling not a problem, just don't want the bias from the probas of the train
        # Make it not too long, so ficed to 300.
        x_test = transform_x(x_test, vocabulary)
        y_test = np.array(y_test)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 300 / int(len(x_train)))
        x_val, x_train = transform_x(x_val, vocabulary), transform_x(x_train, vocabulary)
        # create vocab from training set
        # create TfIdfClassifier
        clf = SVC()
        clf.set_params(**svm_best_parameters)
        clf.fit(x_train, y_train)
        
        clf.predict(x_test)
        
        
        ### now calibrate the probabilities
        
        calibrator = CalibratedClassifierCV(clf, cv='prefit')
        calibrator.fit(x_val, np.array(y_val))

        # predict with or without threshold
        distances_conf = calibrator.predict_proba(x_train)
        distances_test = calibrator.predict_proba(x_test)
    
    
    if kwargs.get('scoring_function').lower() == 'cnn_predict':
        x_test = pd.Series(df_test.text)
        x_train = pd.Series(df_train.text)
        model = kwargs.get('model')
        word2idx = kwargs.get('word2idx')
        for text in x_train:
            pred = cnn_predict(text, model.to('cpu'), word2idx).unsqueeze(0)
            try: 
                distances_conf = torch.cat((distances_conf, pred), dim = 0)
            except:        
                distances_conf = pred # pd.cat((distances_test, pred), dim = 1) 

        distances_conf = distances_conf.detach().numpy()
        
        for text in x_test:
            pred = cnn_predict(text, model.to('cpu'), word2idx).unsqueeze(0)
        
            try: 
                distances_test = torch.cat((distances_test, pred), dim = 0)
            except:        
                distances_test = pred # pd.cat((distances_test, pred), dim = 1)
        
        distances_test = distances_test.detach().numpy()

    
    distances_conf =  [[distances_conf[i,0] for i in range(len(x_train)) if y_train[i] == 0], [distances_conf[i,1] for i in range(len(x_train)) if y_train[i] == 1] ]   
    holdout_len = len(y_test)
    t1 = time.time() - t0
    # p_vals_neg = [percentileofscore(distances_conf[:,0], distances_test[i,0]) for i in range(holdout_len)]
    # p_vals_pos = [percentileofscore(distances_conf[:,1], distances_test[i,1]) for i in range(holdout_len)]
    p_vals_neg = [percentileofscore(distances_conf[0], distances_test[i,0]) for i in range(holdout_len)]
    p_vals_pos = [percentileofscore(distances_conf[1], distances_test[i,1]) for i in range(holdout_len)]
    p_vals = np.array([p_vals_neg, p_vals_pos])
    
    # p_vals_neg = [percentileofscore(
    #     distances_conf[:, 0], distances_test[i, 0]) for i, _ in enumerate(y_test)]
    # p_vals_pos = [percentileofscore(
    #     distances_conf[:, 1], distances_test[i, 1]) for i, _ in enumerate(y_test)]
    # p_vals = np.array([p_vals_neg, p_vals_pos])

    # Output one minus the second-largest p value calculated as the confidence of the prediction
    confidence = 100 - np.min(p_vals, axis=0)
    # Output the largest p value calculated as the credibility of the prediction
    credibility = np.max(p_vals, axis=0)
    if kwargs.get('return_pred') == True:
        prediction = np.argmax(p_vals, axis = 0)
        return [credibility, confidence, prediction]
    elif kwargs.get('return_pvals') == True:
        return p_vals
    elif kwargs.get('timeit') == True:
        return [credibility, confidence], t1
    else:
        return [credibility, confidence]

def conformal_scores(df_test, df_train, **kwargs):
    t0 = time.time()
    df_train = df_train.assign(
            stemmed_text = df_train.text.apply(lambda x: stemming_tokenizer(x))
    ).fillna(' ')
    df_test = df_test.assign(
            stemmed_text = df_train.text.apply(lambda x: stemming_tokenizer(x))
        ).fillna(' ')
    x_test = pd.Series(df_test.stemmed_text)
    x_train = pd.Series(df_train.stemmed_text)
    y_test = df_test.label.astype(int).to_list()
    y_train = df_train.label.astype(int).to_list()
    vocabulary = create_vocabulary(x_train)
    if kwargs.get('scoring_function').lower() == 'nearestneighbors':
        print('Scoring Function: Closest 3 neighbors')

        x_test = transform_x(x_test, vocabulary)
        
        x_count_pos, x_count_neg, x_train = get_x_counts(
            x_train, y_train, vocabulary)

        distances_conf = avg_closest_distance(
            x_count_pos, x_count_neg, x_train, y_train, 'train')

        distances_test = avg_closest_distance(
            x_count_pos, x_count_neg, x_test, y_test, 'test')

    if kwargs.get('scoring_function').lower() == 'classaverage':
        print('Scoring function: Distance to class average')

        x_neg_avg_train, x_pos_avg_train = find_average_classes(
            x_train, y_train, vocabulary)

        # percentiles = calculate_percentile(X_test, y_test,X_train, y_train, vocabulary, x_neg_avg_train, x_pos_avg_train)
        distances_conf = distance_to_average(
            x_train,  vocabulary, x_neg_avg_train, x_pos_avg_train, kwargs.get('preclassification', None))

        distances_test = distance_to_average(
            x_test,  vocabulary, x_neg_avg_train, x_pos_avg_train, kwargs.get('preclassification', None))

    if kwargs.get('scoring_function').lower() == 'predict_probas':
        print('Scoring function: Sklearn Predict Probabilities') 
        # create validation set. Needs to be from training set (len of test cannot change)
        # Uspampling not a problem, just don't want the bias from the probas of the train
        # Make it not too long, so ficed to 300.
        x_test = transform_x(x_test, vocabulary)
        y_test = np.array(y_test)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 300 / int(len(x_train)))
        x_val, x_train = transform_x(x_val, vocabulary), transform_x(x_train, vocabulary)
        # create vocab from training set
        # create TfIdfClassifier
        clf = SVC()
        clf.set_params(**svm_best_parameters)
        clf.fit(x_train, y_train)
        
        clf.predict(x_test)
        
        
        ### now calibrate the probabilities
        
        calibrator = CalibratedClassifierCV(clf, cv='prefit')
        calibrator.fit(x_val, np.array(y_val))

        # predict with or without threshold
        distances_conf = calibrator.predict_proba(x_train)
        distances_test = calibrator.predict_proba(x_test)
    
    
    if kwargs.get('scoring_function').lower() == 'cnn_predict':
        x_test = pd.Series(df_test.text)
        x_train = pd.Series(df_train.text)
        model = kwargs.get('model')
        word2idx = kwargs.get('word2idx')
        for text in x_train:
            pred = cnn_predict(text, model.to('cpu'), word2idx).unsqueeze(0)
            try: 
                distances_conf = torch.cat((distances_conf, pred), dim = 0)
            except:        
                distances_conf = pred # pd.cat((distances_test, pred), dim = 1) 

        distances_conf = distances_conf.detach().numpy()
        
        for text in x_test:
            pred = cnn_predict(text, model.to('cpu'), word2idx).unsqueeze(0)
        
            try: 
                distances_test = torch.cat((distances_test, pred), dim = 0)
            except:        
                distances_test = pred # pd.cat((distances_test, pred), dim = 1)
        
        distances_test = distances_test.detach().numpy()

    
    distances_conf =  [*[distances_conf[i,0] for i in range(len(x_train)) if y_train[i] == 0], *[distances_conf[i,1] for i in range(len(x_train)) if y_train[i] == 1] ]   
    t1 = time.time()

    holdout_len = len(y_test)

    # p_vals_neg = [percentileofscore(distances_conf[:,0], distances_test[i,0]) for i in range(holdout_len)]
    # p_vals_pos = [percentileofscore(distances_conf[:,1], distances_test[i,1]) for i in range(holdout_len)]
    p_vals_neg = [percentileofscore(distances_conf, distances_test[i,0]) for i in range(holdout_len)]
    p_vals_pos = [percentileofscore(distances_conf, distances_test[i,1]) for i in range(holdout_len)]
    p_vals = np.array([p_vals_neg, p_vals_pos])
    
    # p_vals_neg = [percentileofscore(
    #     distances_conf[:, 0], distances_test[i, 0]) for i, _ in enumerate(y_test)]
    # p_vals_pos = [percentileofscore(
    #     distances_conf[:, 1], distances_test[i, 1]) for i, _ in enumerate(y_test)]
    # p_vals = np.array([p_vals_neg, p_vals_pos])

    # Output one minus the second-largest p value calculated as the confidence of the prediction
    confidence = 100 - np.min(p_vals, axis=0)
    # Output the largest p value calculated as the credibility of the prediction
    credibility = np.max(p_vals, axis=0)
    
    if kwargs.get('return_pred') == True:
        prediction = np.argmax(p_vals, axis = 0)
        return [credibility, confidence, prediction]
    elif kwargs.get('return_pvals') == True:
        return p_vals
    elif kwargs.get('timeit') == True: 
        return [credibility, confidence], t1-t0
    else:
        return [credibility, confidence]

def prediction_set(p_vals, thr):
    assert p_vals.shape[0] == 2
    predset = []
    for i in range(p_vals.shape[1]):
        p_val = p_vals[:,i]
        predset.append([1*(val < thr) for val in p_val])
    return predset    


    
scoring_title_dict = {'nearestneighbors': 'Nearest Neighbours Metric', 'classaverage': 'Class Average Metric',\
    'predict_probas': 'Sklearn - Predict Probabilities', 'cnn_predict': 'CNN - Predict Probabilities' }
confcreddict = {'confidence': 'Confidence', 'credibility': 'Credibility'}

if __name__ == '__main__':
    df_train = pd.read_csv('../cnn/221222_train.csv').fillna(' ')
    df_test = pd.concat([pd.read_csv('../cnn/221222_test.csv'), pd.read_csv('../cnn/221222_val.csv')]).fillna(' ')
    
    embeddings = torch.load('../cnn/221222_fintext_embedding.pt')
    embeddings = torch.tensor(embeddings)
    
    device = get_device()
    cnn_non_static, _ = initialize_model(  pretrained_embedding=embeddings,
            freeze_embedding= False,
            learning_rate= 0.05, 
            dropout=best_params.get('dropout'),
            device=device,
            filter_sizes=[best_params.get('num_filters_0'),best_params.get('num_filters_1'),best_params.get('num_filters_2')],
            num_filters=[best_params.get('region_size'), best_params.get('region_size'), best_params.get('region_size')]
            )

    cnn_non_static.load_state_dict(torch.load('../cnn/models/best_model_second.pt'))
    scores = confidence_scores(df_test, df_train, scoring_function = 'cnn_predict',  return_pred = True, model = cnn_non_static)
    

    

