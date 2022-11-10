########################################################################################################
# ml.py
#
########################################################################################################

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import warnings
import random
import os
import time
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score

OUTPUT_FP = '../output/classification/'
OUTPUT_TOPICS_FP = '../output/classification/topics/'
OUTPUT_TEXT_FP = '../output/classification/text/'
OUTPUT_PHASED_FP = '../output/classification/phased/'

CLASS_LABELS = ['random sample', 'dem dist', 'informed consent', 'data public', 'irb', 'limitations', 'anonymity',
                'data quality', 'missing values', 'ethics section', 'generalizability', 'target pop', 'hypotheses',
                'data availability', 'preprocessing', 'var selection', 'var construction', 'var reconstruction',
                'list models', 'model steps', 'model choice', 'define measures', 'common uses', 'cv', 'hyperparameters',
                'attempted models', 'infrastructure/packages', 'biases/fairness', 'model comparison',
                'experiment cases', 'results use', 'results misuse']

TOPIC_DIST_COLS = ['topic1', 'topic2', 'topic3', 'topic4', 'topic5', 'topic6', 'topic7', 'topic8', 'topic9', 'topic10',
                   'topic11', 'topic12', 'topic13', 'topic14', 'topic15', 'topic16', 'topic17', 'topic18', 'topic19',
                   'topic20', 'topic21', 'topic22', 'topic23', 'topic24', 'topic25', 'topic26', 'topic27', 'topic28',
                   'topic29', 'topic30', 'topic31', 'topic32', 'topic33', 'topic34', 'topic35']

TRAINING_COLS = ['topic1', 'topic2', 'topic3', 'topic4', 'topic5', 'topic6', 'topic7', 'topic8', 'topic9', 'topic10',
                 'topic11', 'topic12', 'topic13', 'topic14', 'topic15', 'topic16', 'topic17', 'topic18', 'topic19',
                 'topic20', 'topic21', 'topic22', 'topic23', 'topic24', 'topic25', 'topic26', 'topic27', 'topic28',
                 'topic29', 'topic30', 'topic31', 'topic32', 'topic33', 'topic34', 'topic35', 'text']


########################################################################################################
# get_train_and_test
# Inputs: training - training data, testing - testing data, criteria - class label
# Return: X_train - training X set, Y_train - training Y set, X_validate - testing X set, Y_validate -
# testing X set, X_train_2 - training X set for phase 2, X_validate_2 - testing X set for phase 2
# Description: undersamples and formats training and testing data
########################################################################################################
def get_train_and_test(training, testing, criteria):
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()

    # undersample training data
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_equal, Y_equal = undersample.fit_resample(training[TRAINING_COLS], training[criteria])

    # get training data
    X_train = X_equal['text']
    x_train_counts = count_vect.fit_transform(X_train)
    training_vocab = count_vect.vocabulary_
    X_train = tfidf_transformer.fit_transform(x_train_counts)
    X_train_2 = X_equal[TOPIC_DIST_COLS]
    Y_train = Y_equal

    # get testing data
    print(type(testing))
    X_validate = testing['text']
    count_vect_2 = CountVectorizer(vocabulary=training_vocab)
    x_train_counts = count_vect_2.fit_transform(X_validate)
    X_validate = tfidf_transformer.fit_transform(x_train_counts)
    X_validate_2 = testing[TOPIC_DIST_COLS]
    Y_validate = testing[CLASS_LABELS]

    return X_train, X_validate, Y_train, Y_validate, X_train_2, X_validate_2


########################################################################################################
# get_ROC
# Inputs: classifier - classifier to generate ROC oof, model_name - name of model, X_validate -
# testing X set, Y_validate - testing X set, output_fp - output filepath to save ROC figure
# Return: N/A
# Description: Generates ROC figure for inputted classfier
########################################################################################################
def get_ROC(classifier, model_name, criteria, X_validate, Y_validate, output_fp):

    # plot and save ROC curves
    if model_name != 'NeuralNet':
        predictedprob = classifier.predict_proba(X_validate)
        predictedprob = predictedprob[::, 1]
        auc_score = metrics.roc_auc_score(Y_validate, predictedprob)
        fpr, tpr, thresholds = metrics.roc_curve(Y_validate, predictedprob)
        plt.plot(fpr, tpr, label="AUC=" + str(auc_score))
        plt.plot([0, 1], [0, 1], color='red', linestyle='dashed')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(output_fp, str(model_name) + "_" + criteria + ".png"))
        plt.clf()


########################################################################################################
# cross_validation
# Inputs: model - name of model, X_train - training X set, Y_train - training Y set
# Return: cv_results - cross_val_score
# Description: Performs k fold cross validation with the testing set (k=5) for the inputted model.
# Then, returns out average accuracy and standard deviation.
########################################################################################################
def cross_validation(model, X_train, Y_train):

    # Sets up parameters for cross validation
    num_folds = 5
    seed = 5
    scoring = 'accuracy'

    # Run K fold Cross Validation and get Mean Accuracy & Standard Deviation
    kfold = StratifiedKFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    return cv_results


########################################################################################################
# get_best_model
# Inputs:key - model name, model - model, X_train - train X set, Y_train - train Y set, X_validate -
# test X set, Y_validate - test Y set, file - file, type - experiment
# type
# Return: best_model - the best model from grid search, train_predictions - training predictions,
# test_predictions - testing predictios, f1 - f1 score
# Description: If SVM or CART, performs grid search. Identifies best model, performs cross validation
# and holdout, and calculates F1 scores.
########################################################################################################
def get_best_model(key, model, X_train, Y_train, X_validate, Y_validate, file, type):

    # set up grid search
    if key == 'KNN':
        n_neighbors = list(range(2, 5))
        grid = dict(n_neighbors=n_neighbors)
    elif key == 'SVM':
        kernel = ["linear", "rbf", "sigmoid", "poly"]
        grid = dict(kernel=kernel)
    elif key == 'CART':
        gain = ['gini', 'entropy']
        max_depth = range(1, 10)
        grid = dict(criterion=gain, max_depth=max_depth)

    # run grid search
    if (key != 'NaiveBayes') & (key != 'NeuralNet'):
        cv_fold = RepeatedStratifiedKFold(n_splits=3, n_repeats=50, random_state=1)
        grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1,
                                   cv=cv_fold, scoring="accuracy")
        search_results = grid_search.fit(X_train, Y_train)
        best_model = search_results.best_estimator_

        cv = search_results.best_score_
        test_predictions = best_model.predict(X_validate)
        train_predictions = best_model.predict(X_train)
        accuracy = best_model.score(X_validate, Y_validate)
        f1 = f1_score(Y_validate, test_predictions, average='macro')

        print("key: " + key + ", best params: " + str(search_results.best_params_))

        if key == 'CART':
            print("root: " + str(best_model.tree_.feature[0]))

    # train models which don't have parameters
    else:
        model.fit(X_train, Y_train)
        best_model = model

        cv = cross_validation(best_model, X_train, Y_train)
        test_predictions = model.predict(X_validate)
        train_predictions = model.predict(X_train)
        accuracy = accuracy_score(Y_validate, test_predictions)
        f1 = f1_score(Y_validate, test_predictions, average='macro')

    # write results to file
    if type != 'phase1':
        file.write("\nValidation Dataset Testing Results:" + '\n')
        file.write('\n' + str(key) + ' Accuracy: ' + str(accuracy) + '\n')
        file.write(str(confusion_matrix(Y_validate, test_predictions)) + '\n')
        file.write(str(classification_report(Y_validate, test_predictions)) + '\n')

        file.write("\nCross Validation Results:")
        msg = "\n%s: %f (%f)" % (key, cv.mean(), cv.std())
        file.write(msg)

    return best_model, train_predictions, test_predictions, f1


########################################################################################################
# run_final_predictive_analytics
# Inputs: X_t_text - x train text, X_v_text - x validate text, Y_t - y train, Y_v - y validate,
# X_t_topic - x train topic, X_v_topic - x train topic, models - list of models, type - experiment type,
# criteria - class label, file - output file
# Return: Scores - F1 scores from phase 1
# Description: Performs predictive analytics on the inputted dictionary of models. Specifically, performs
# cross validation and validaiton on validation dataset, sets up the training data for phase 2, and
# conducts phase 2.
########################################################################################################
def run_final_predictive_analytics(X_t_text, X_v_text, Y_t, Y_v, X_t_topic, X_v_topic, models, type, criteria, file):

    train_predictions = {}
    test_predictions = {}
    scores = {}

    if type != 'phase1':
        file.write('\n\n\nWORKING WITH CRITERIA ' + criteria + '\n\n')
        print()

    # build a set of models with different classifiers
    for key in models:

        # modeling for topic features or stage-2 of multi-stage models
        if type == 'topic' or type == 'phase2':
            best_model, train_prediction, test_prediction, score = get_best_model(key, models[key], X_t_topic, Y_t,
                                                                                  X_v_topic, Y_v[criteria], file, type)
            train_predictions[key] = train_prediction
            test_predictions[key] = test_prediction
            scores[key] = score

            file.write('\n\n-------------------------------------------------------------\n')

        # modeling for text-based features
        elif type == 'text':
            best_model, train_prediction, test_prediction, score = get_best_model(key, models[key], X_t_text, Y_t,
                                                                                  X_v_text, Y_v[criteria], file, type)
            test_predictions[key] = test_prediction
            scores[key] = score

            file.write('\n\n-------------------------------------------------------------\n')

        # modeling for stage-1 of multi-stage models
        else:
            best_model, train_prediction, test_prediction, score = get_best_model(key, models[key], X_t_text, Y_t,
                                                                                  X_v_text, Y_v[criteria], file, type)
            train_predictions[key] = train_prediction
            test_predictions[key] = test_prediction
            scores[key] = score

            col_label = key + "_pred"
            X_t_topic[col_label] = train_predictions.get(key)
            X_v_topic[col_label] = test_predictions.get(key)

    return scores, test_predictions


########################################################################################################
# get_models
# Inputs: N/A
# Return: models - dictionary of all models to test
# Description: Sets up storage dictionaries for models
########################################################################################################
def get_models():

    # create dictionary of classifiers for modeling
    models = {'NaiveBayes': MultinomialNB(), 'CART': DecisionTreeClassifier(), 'KNN': KNeighborsClassifier(),
              'SVM': SVC(probability=True), 'NeuralNet': Perceptron()}

    return models


########################################################################################################
# calc_scores
# Inputs: criteria - criteria label, top_f1 - dict with best F1 score from each experiment, top_preds -
# dict with best set of predictions by F1 score from each experiment, preds - df of best predictions
# Return: N/A
# Description: isolates the best set of predictions by F1 score for each criteria and adds it to a df
########################################################################################################
def calc_scores(criteria, top_f1, top_preds, preds):

    # add a criteria's best predictions to the df of best predictions
    max_key = max(top_f1, key=top_f1.get)
    print(type(top_preds.get(max_key)))
    preds[criteria] = list(top_preds.get(max_key))


########################################################################################################
# analyze_preds
# Inputs: preds - df of best predictions for each criteria by section, test - test dataset
# Return: N/A
# Description: groups predicted scores by document
########################################################################################################
def analyze_preds(preds, test):

    # group predictions by document number
    preds['doc_num'] = test['doc_num']
    paper_preds = preds.groupby(by='doc_num').max().reset_index()
    paper_preds['predicted_score'] = paper_preds[CLASS_LABELS].sum(axis=1) / len(CLASS_LABELS) * 100
    paper_preds = paper_preds.drop(columns=['doc_num'])

    paper_preds.to_csv(f'{OUTPUT_FP}/predictions/document_predictions.csv', index=False)


########################################################################################################
# get_output_files
# Inputs: N/A
# Return: output - dictionary of output files
# Description: opens output files and creates dictionary for easy access
########################################################################################################
def get_output_files():

    # open output files and create dictionary
    topics_fp = open(f'{OUTPUT_TOPICS_FP}models_performance_section.txt', 'w')

    text_fp = open(f'{OUTPUT_TEXT_FP}models_performance_section.txt', 'w')
    text_semi_fp = open(f'{OUTPUT_TEXT_FP}models_performance_section_SEMI.txt', 'w')

    allF1_fp = open(f'{OUTPUT_PHASED_FP}models_performance_section_allF1.txt', 'w')
    bestF1_fp = open(f'{OUTPUT_PHASED_FP}models_performance_section_bestF1.txt', 'w')
    allF1_semi_fp = open(f'{OUTPUT_PHASED_FP}models_performance_section_SEMI_allF1.txt', 'w')
    bestF1_semi_fp = open(f'{OUTPUT_PHASED_FP}models_performance_section_SEMI_bestF1.txt', 'w')

    output = {'topic': topics_fp, 'text': text_fp, 'text_semi': text_semi_fp, 'allF1': allF1_fp, 'bestF1': bestF1_fp,
              'allF1_semi': allF1_semi_fp, 'bestF1_semi': bestF1_semi_fp}

    return output


########################################################################################################
# run_experiments
# Inputs: criteria - transparency criteria, test - experiment, models - dict of models, output_fp - dict
# of output files, X_t_text - x train text, X_v_text - x validate text, Y_t - y train, Y_v - y validate,
# X_t_topics - x train topics, X_v_topics - x validate topics, X_ts_text - x train semi text, X_vs_text -
# x validate semi text, Y_ts - y train semi, Y_vs - y validate semi, X_ts_topics - x train semi topics,
# X_vs_topics - x train semi topics
# Return: N/A
# Description: runs either the text-only, topics-only, all F1 phased, or best F1 phased classification
# experiment
########################################################################################################
def run_experiments(criteria, test, models, output_fp, X_t_text, X_v_text, Y_t, Y_v, X_t_topics, X_v_topics, X_ts_text,
                    X_vs_text, Y_ts, Y_vs, X_ts_topics, X_vs_topics, top_f1, top_preds):

    print('\n\n' + criteria + ', ' + test + '\n')

    # run experiment with multi-stage (b) model configuration
    if test == 'bestF1':

        # baseline
        key = test
        scores, b_pred1 = run_final_predictive_analytics(X_t_text, X_v_text, Y_t, Y_v, X_t_topics, X_v_topics, models,
                                                         'phase1', criteria, output_fp[key])
        max_key = max(scores, key=scores.get)
        for model in models:
            if model != max_key:
                col_to_drop = model + "_pred"
                X_t_topics = X_t_topics.drop([col_to_drop], axis=1)
                X_v_topics = X_v_topics.drop([col_to_drop], axis=1)
        scores, b_pred2 = run_final_predictive_analytics(X_t_text, X_v_text, Y_t, Y_v, X_t_topics, X_v_topics, models,
                                                         'phase2', criteria, output_fp[key])

        # save best predictions
        max_key = max(scores, key=scores.get)
        top_f1["bestF1_baseline"] = scores.get(max_key)
        top_preds["bestF1_baseline"] = b_pred2.get(max_key)

        # seed-biased
        key_semi = test + '_semi'
        scores, s_pred1 = run_final_predictive_analytics(X_ts_text, X_vs_text, Y_ts, Y_vs, X_ts_topics, X_vs_topics,
                                                         models, 'phase1', criteria, output_fp[key_semi])
        max_key = max(scores, key=scores.get)
        for model in models:
            if model != max_key:
                col_to_drop = model + "_pred"
                X_ts_topics = X_ts_topics.drop([col_to_drop], axis=1)
                X_vs_topics = X_vs_topics.drop([col_to_drop], axis=1)
        scores, s_pred2 = run_final_predictive_analytics(X_ts_text, X_vs_text, Y_ts, Y_vs, X_ts_topics, X_vs_topics,
                                                         models, 'phase2', criteria, output_fp[key_semi])

        # save best predictions
        max_key = max(scores, key=scores.get)
        top_f1["bestF1_seeded"] = scores.get(max_key)
        top_preds["bestF1_seeded"] = s_pred2.get(max_key)

    # run experiment with multi-stage (a) model configuration
    elif test == 'allF1':

        # baseline
        key = test
        scores, b_pred1 = run_final_predictive_analytics(X_t_text, X_v_text, Y_t, Y_v, X_t_topics, X_v_topics, models,
                                                         'phase1', criteria, output_fp[key])
        scores, b_pred2 = run_final_predictive_analytics(X_t_text, X_v_text, Y_t, Y_v, X_t_topics, X_v_topics, models,
                                                         'phase2', criteria, output_fp[key])

        # save best predictions
        max_key = max(scores, key=scores.get)
        top_f1["allF1_baseline"] = scores.get(max_key)
        top_preds["allF1_baseline"] = b_pred2.get(max_key)

        # seed-biased
        key_semi = test + '_semi'
        scores, s_all_1_pred = run_final_predictive_analytics(X_ts_text, X_vs_text, Y_ts, Y_vs, X_ts_topics,
                                                              X_vs_topics, models, 'phase1', criteria,
                                                              output_fp[key_semi])
        scores, s_all_2_pred = run_final_predictive_analytics(X_ts_text, X_vs_text, Y_ts, Y_vs, X_ts_topics,
                                                              X_vs_topics, models, 'phase2', criteria,
                                                              output_fp[key_semi])

        # save best predictions
        max_key = max(scores, key=scores.get)
        top_f1["allF1_seeded"] = scores.get(max_key)
        top_preds["allF1_seeded"] = s_all_2_pred.get(max_key)

    # run experiment with single-stage models (ngram features)
    elif test == 'text':

        # baseline
        key = test
        scores, b_pred = run_final_predictive_analytics(X_t_text, X_v_text, Y_t, Y_v, X_t_topics, X_v_topics, models,
                                                        test, criteria, output_fp[key])

        # save best predictions
        max_key = max(scores, key=scores.get)
        top_f1["text_baseline"] = scores.get(max_key)
        top_preds["text_baseline"] = b_pred.get(max_key)

        # seed-biased
        key_semi = test + '_semi'
        scores, s_pred = run_final_predictive_analytics(X_ts_text, X_vs_text, Y_ts, Y_vs, X_ts_topics, X_vs_topics,
                                                        models, test, criteria, output_fp[key_semi])

        # save best predictions
        max_key = max(scores, key=scores.get)
        top_f1["text_seeded"] = scores.get(max_key)
        top_preds["text_seeded"] = s_pred.get(max_key)

    # run experiment with topic dataset (single-stage)
    else:

        # only uses topic distribution features
        key = test
        scores, b_pred = run_final_predictive_analytics(X_t_text, X_v_text, Y_t, Y_v, X_t_topics, X_v_topics, models,
                                                        test, criteria, output_fp[key])

        # save best predictions
        max_key = max(scores, key=scores.get)
        top_f1["topic"] = scores.get(max_key)
        top_preds["topic"] = b_pred.get(max_key)


########################################################################################################
# classification
# Inputs: train_FP - training data, train_semi_FP - seeded training data, test_FP - testing data
# Return: N/A
# Description: reads in data and conducts classification experiments with various types of training data
# for classification based on topic distribution, based on text, and based on both topics and text
########################################################################################################
def classification(train_FP, train_semi_FP, test_FP):

    # start time and set seed
    start_time = time.time()
    random.seed(1)
    np.random.seed(1)
    warnings.filterwarnings('ignore')

    # read data
    train = pd.read_csv(train_FP)
    train_semi = pd.read_csv(train_semi_FP)
    test = pd.read_csv(test_FP)

    # get dict of models
    models = get_models()

    # get output files
    output_fp = get_output_files()

    # classify
    tests = ['topic', 'text', 'allF1', 'bestF1']
    preds = pd.DataFrame()
    for criteria in CLASS_LABELS:

        top_preds = {}
        top_f1 = {}

        # Generate training and testing sets
        X_t_text, X_v_text, Y_t, Y_v, X_t_topics, X_v_topics = get_train_and_test(train, test, criteria)
        X_ts_text, X_vs_text, Y_ts, Y_vs, X_ts_topics, X_vs_topics = get_train_and_test(train_semi, test, criteria)

        # run experiments for each criteria
        for item in tests:
            run_experiments(criteria, item, models, output_fp, X_t_text, X_v_text, Y_t, Y_v, X_t_topics, X_v_topics,
                            X_ts_text, X_vs_text, Y_ts, Y_vs, X_ts_topics, X_vs_topics, top_f1, top_preds)

        calc_scores(criteria, top_f1, top_preds, preds)

    print(preds)
    analyze_preds(preds, test)

    # print total time
    print("\nTotal Program Time:" + str((time.time() - start_time)) + " seconds ")


########################################################################################################
# main
# Inputs: N/A
# Return: N/A
# Description: N/A
########################################################################################################
def main():

    print('running ml.py main')


if __name__ == "__main__":
    main()
