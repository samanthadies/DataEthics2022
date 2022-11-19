########################################################################################################
# ml.py
# This program runs classification experiments at the section level for each transparency criteria class
# label. We build models following two model configuration: single-stage and multi-stage. We use four
# datasets, the baseline (text) data, the topics data (topic distributions), and seed-biased (text with
# seeds) data. We then identify the best model for each criteria and generate paper level scores using
# the best models.
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


def get_train_and_test(training, testing, criteria, class_labels, topic_dist_cols, training_cols):
    """
    undersamples and formats training and testing data
    :param training: training data
    :param testing: testing data
    :param criteria: class label
    :param class_labels: list of all class labels
    :param topic_dist_cols: list of coluns with topic distributions
    :param training_cols: list of all columns to use for training
    :return: X_train - training X set, Y_train - training Y set, X_validate - testing X set, Y_validate - testing X set,
    X_train_2 - training X set for phase 2, X_validate_2 - testing X set for phase 2
    """

    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()

    # undersample training data
    undersample = RandomUnderSampler(sampling_strategy='majority')
    X_equal, Y_equal = undersample.fit_resample(training[training_cols], training[criteria])

    # get training data
    X_train = X_equal['text']
    x_train_counts = count_vect.fit_transform(X_train)
    training_vocab = count_vect.vocabulary_
    X_train = tfidf_transformer.fit_transform(x_train_counts)
    X_train_2 = X_equal[topic_dist_cols]
    Y_train = Y_equal

    # get testing data
    print(type(testing))
    X_validate = testing['text']
    count_vect_2 = CountVectorizer(vocabulary=training_vocab)
    x_train_counts = count_vect_2.fit_transform(X_validate)
    X_validate = tfidf_transformer.fit_transform(x_train_counts)
    X_validate_2 = testing[topic_dist_cols]
    Y_validate = testing[class_labels]

    return X_train, X_validate, Y_train, Y_validate, X_train_2, X_validate_2


def get_ROC(classifier, model_name, criteria, X_validate, Y_validate, output_fp):
    """
    Generates ROC figure for inputted classfier
    :param classifier: classifier for which to generate ROC
    :param model_name: name of model
    :param criteria: class label
    :param X_validate: testing X set
    :param Y_validate: ttesting class label
    :param output_fp: filepath for saving output
    :return: void
    """

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


def cross_validation(model, X_train, Y_train):
    """
    Performs k fold cross validation with the testing set (k=5) for the inputted model. Then, returns out average
    accuracy and standard deviation.
    :param model: name of model
    :param X_train: training X set
    :param Y_train: training class labels
    :return: cv_results - cross validation score
    """

    # Sets up parameters for cross validation
    num_folds = 5
    seed = 5
    scoring = 'accuracy'

    # Run K fold Cross Validation and get Mean Accuracy & Standard Deviation
    kfold = StratifiedKFold(n_splits=num_folds, random_state=seed, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    return cv_results


def get_best_model(key, model, X_train, Y_train, X_validate, Y_validate, file, type):
    """
    If SVM or CART, performs grid search. Identifies best model, performs cross validation and holdout, and
    calculates F1 scores.
    :param key: model name
    :param model: model
    :param X_train: train X set
    :param Y_train: train class labels
    :param X_validate: test X set
    :param Y_validate: test class labels
    :param file: output file
    :param type: experiment tag
    :return: best_model - best model from grid search, train_predictions - training predicitons, test_predictions -
    testing predictions, f1 - f1 score
    """

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


def run_final_predictive_analytics(X_t_text, X_v_text, Y_t, Y_v, X_t_topic, X_v_topic, models, type, criteria, file):
    """
    Performs predictive analytics on the inputted dictionary of models. Specifically, performs cross validation and
    validaiton on validation dataset, sets up the training data for phase 2, and conducts phase 2.
    :param X_t_text: x train text
    :param X_v_text: x validate text
    :param Y_t: y train
    :param Y_v: y validate
    :param X_t_topic: x train topics
    :param X_v_topic: x validate topics
    :param models: list of models
    :param type: experiment tag
    :param criteria: class label
    :param file: output file
    :return: Scores - f1 scores from phase 1
    """

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


def get_models():
    """
    Sets up storage dictionaries for models
    :return: models - dictionary of all models to test
    """

    # create dictionary of classifiers for modeling
    models = {'NaiveBayes': MultinomialNB(), 'CART': DecisionTreeClassifier(), 'KNN': KNeighborsClassifier(),
              'SVM': SVC(probability=True), 'NeuralNet': Perceptron()}

    return models


def calc_scores(criteria, top_f1, top_preds, preds):
    """
    isolates the best set of predictions by F1 score for each criteria and adds it to a df
    :param criteria: criteria label
    :param top_f1: dict w/ best f1 scores from each experiment
    :param top_preds: dict with best set of predictions by f1 score for each experiment
    :param preds: df of best predictions
    :return: void
    """

    # add a criteria's best predictions to the df of best predictions
    max_key = max(top_f1, key=top_f1.get)
    print(type(top_preds.get(max_key)))
    preds[criteria] = list(top_preds.get(max_key))


def analyze_preds(preds, test, class_labels):
    """
    groups predicted scores by document
    :param preds: df of best predictions for each criteria by section
    :param test: test dataset
    :param class_labels: list of all class labels
    :return: void
    """

    # group predictions by document number
    preds['doc_num'] = test['doc_num']
    paper_preds = preds.groupby(by='doc_num').max().reset_index()
    paper_preds['predicted_score'] = paper_preds[class_labels].sum(axis=1) / len(class_labels) * 100
    paper_preds = paper_preds.drop(columns=['doc_num'])

    paper_preds.to_csv(f'{OUTPUT_FP}/predictions/document_predictions.csv', index=False)


def get_output_files():
    """
    opens output files and creates dictionary for easy access
    :return: output - dictionary of output files
    """

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


def run_experiments(criteria, test, models, output_fp, X_t_text, X_v_text, Y_t, Y_v, X_t_topics, X_v_topics, X_ts_text,
                    X_vs_text, Y_ts, Y_vs, X_ts_topics, X_vs_topics, top_f1, top_preds):
    """
    runs either the text-only, topics-only, all F1 phased, or best F1 phased classification experiment
    :param criteria: transparency criteria
    :param test: experiment tag
    :param models: dict of models
    :param output_fp: dict of output files
    :param X_t_text: X train text
    :param X_v_text: X validate text
    :param Y_t: y train text
    :param Y_v: y validate text
    :param X_t_topics: X train topics
    :param X_v_topics: X validate topics
    :param X_ts_text: X train seed-biased
    :param X_vs_text: X validate seed-biased
    :param Y_ts: y train seed-biased
    :param Y_vs: y validate seed-biased
    :param X_ts_topics: X train seed-biased topics
    :param X_vs_topics: X validate seed-biased topics
    :param top_f1: dict of best f1 scores
    :param top_preds: dict of best predictions
    :return:
    """

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


def classification(train_FP, train_semi_FP, test_FP):
    """
    reads in data and conducts classification experiments with various types of training data for classification
    based on topic distribution, based on text, and based on both topics and text
    :param train_FP: training data
    :param train_semi_FP: seeded training data
    :param test_FP: testing data
    :return: void
    """

    # start time and set seed
    start_time = time.time()
    random.seed(1)
    np.random.seed(1)
    warnings.filterwarnings('ignore')

    # read data
    train = pd.read_csv(train_FP)
    train_semi = pd.read_csv(train_semi_FP)
    test = pd.read_csv(test_FP)

    labels = pd.read_csv('../data/cleaned/classification/class_labels.csv')
    class_labels = labels.columns
    topics = pd.read_csv('../data/cleaned/classification/topic_dist_cols.csv')
    topic_dist_cols = topics.columns
    training = pd.read_csv('../data/cleaned/classification/training_cols.csv')
    training_cols = training.columns

    # get dict of models
    models = get_models()

    # get output files
    output_fp = get_output_files()

    # classify
    tests = ['topic', 'text', 'allF1', 'bestF1']
    preds = pd.DataFrame()
    for criteria in class_labels:

        top_preds = {}
        top_f1 = {}

        # Generate training and testing sets
        X_t_text, X_v_text, Y_t, Y_v, X_t_topics, X_v_topics = get_train_and_test(train, test, criteria, class_labels,
                                                                                  topic_dist_cols, training_cols)
        X_ts_text, X_vs_text, Y_ts, Y_vs, X_ts_topics, X_vs_topics = get_train_and_test(train_semi, test, criteria,
                                                                                        class_labels, topic_dist_cols,
                                                                                        training_cols)

        # run experiments for each criteria
        for item in tests:
            run_experiments(criteria, item, models, output_fp, X_t_text, X_v_text, Y_t, Y_v, X_t_topics, X_v_topics,
                            X_ts_text, X_vs_text, Y_ts, Y_vs, X_ts_topics, X_vs_topics, top_f1, top_preds)

        calc_scores(criteria, top_f1, top_preds, preds)

    analyze_preds(preds, test, class_labels)

    # print total time
    print("\nTotal Program Time:" + str((time.time() - start_time)) + " seconds ")


def main():
    """

    :return: void
    """

    print('running ml.py main')


if __name__ == "__main__":
    main()
