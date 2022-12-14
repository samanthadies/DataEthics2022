########################################################################################################
# main.py
# This program is the main driver program for the transparency modeling task. It starts by running
# Phase 1 which includes exploratory data analysis and preprocessing. Phase 2, the iterative topic
# modeling stage, is then executed. This stage should be repeated until the user is satisfied with
# the final set of topics and seed words. Phase 3 is the final topic modeling run and the
# classification step. Lastly, Phase 4 conducts analysis and produces visualizations based
# on the classification results.
########################################################################################################

import eda
import textCleaning
import topics
import ml


def phase1():
    """
    runs Phase 1: the data exporation and preprocessing phase
    :return: void
    """

    # Phase 1: data exploration and preprocessing
    print("Phase 1:\n")

    print("Performing EDA")
    eda.perform_eda('../data/raw/dslc_full_final.csv')

    print("Preprocessing dataframes")
    textCleaning.preprocess('../data/raw/classification/')


def phase2():
    """
    runs Phase 2: the optimizing topics and seed word list phase (repeat manually until satisfied with topics)
    :return: void
    """

    print("\nPhase 2:\n")

    print("Topic modeling. Repeat until satisfied with final topics and seeds.")
    topics.optimize_topics('../data/cleaned/training_text.csv', '../data/raw/seeds/seed_words.csv', 35, 1)


def phase3():
    """
    runs Phase 3: the final topic modeling run and classification phase
    :return: void
    """

    print("\nPhase 3:\n")

    print("Final topic modeling run")
    topics.topics_for_classification('../data/cleaned/training_text.csv',
                                     '../data/cleaned/testing_text.csv',
                                     '../data/cleaned/training.csv',
                                     '../data/cleaned/training_biased.csv',
                                     '../data/cleaned/testing.csv',
                                     '../data/raw/seeds/seed_words.csv', 35)

    print("Performing classification")
    ml.classification('../data/cleaned/classification/training.csv',
                      '../data/cleaned/classification/training_biased.csv',
                      '../data/cleaned/classification/testing.csv')


def phase4():
    """
    Runs Phase 4, the analysis phase
    :return: void
    """

    print("\nPhase 4:\n")

    print("Analyzing classification results")
    eda.analyze_predictions('../output/classification/predictions/document_predictions.csv')


def main():
    """
    Driver program which runs all four phases of transparency modeling from eda and preprocessing to final analysis
    :return: void
    """

    # Phase 1: data exploration and unsupervised learning
    phase1()

    # Phase 2: optimizing topics and seed word list (repeat until satisfied)
    phase2()

    # Phase 3: final topic modeling run and classification
    phase3()

    # Phase 4: analysis and visualization
    phase4()


if __name__ == '__main__':
    main()
