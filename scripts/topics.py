########################################################################################################
# topics.py
# This program serves two purposes. First, it can be used for iterative topic modeling. It should be
# run as many times as necessary to optimize topics and seed word list. Second, it should be run with
# the classification datasets and is used to build the topic features.
########################################################################################################

# from gdtm.models import GTM
import shutil
import random
from gensim.models.phrases import Phrases, Phraser
from nltk.corpus import stopwords
from gensim import corpora
from gdtm.helpers.common import load_flat_dataset
from gdtm.helpers.weighting import compute_idf_weights
from topics_helper.src.gdtm.wrappers import GTMMallet
import pandas as pd
import warnings
from tqdm import tqdm
tqdm.pandas()


BASE_WRITE_FP = '../output/topics/'


def filter_stopwords(dataset):
    """
    identifies a list of stopwords and removes them from the dataset
    :param dataset: dataset of documents
    :return: filtered_dataset - dataset without stopwords
    """

    en_stop = stopwords.words('english')
    en_stop.extend(['twitter', 'tweet', 'tweeter', 'reddit', 'subreddit', 'subreddits', 'facebook', 'instagram',
                    'social', 'medium', 'mental', 'health', 'suicide', 'suicidal', 'depression', 'depressed', 'wa',
                    'post', 'word', 'difference', 'individual', 'model', 'support', 'time', 'analysis', 'data',
                    'disorder', 'information', 'language', 'people', 'research', 'study', 'topic', 'community',
                    'condition', 'controll', 'dataset', 'expression', 'feature', 'number', 'online', 'sample', 'table',
                    'work', 'diagnosis', 'ha', 'illness', 'mh', 'psychological', 'sw'])

    filtered_dataset = []
    for sentence in dataset:
        filtered_sentence = [w for w in sentence if w not in en_stop]
        filtered_dataset.append(filtered_sentence)

    return filtered_dataset


def build_corpus(dataset):
    """
    drives iterative topic modeling and writes results to file
    :param dataset: dataset of documents
    :return: corpus - BoW model of documents, dictionary - list of words present in corpus
    """

    # identify unigrams and bigrams in documents
    phrases = Phrases(dataset)
    bigrams = Phraser(phrases)
    bigram_list = [bigrams[doc] for doc in dataset]

    # build dictionary
    dictionary = corpora.Dictionary(bigram_list)
    dictionary.filter_extremes()

    # creat BoW corpus
    corpus = [dictionary.doc2bow(doc) for doc in dataset]

    return corpus, dictionary


def write_topics(model, topic_num, iteration_num):
    """
    creates output files with doc topics and top words for each iteration
    :param model: topic model
    :param topic_num: number of topics
    :param iteration_num: the current version
    :return: void
    """

    top_words_file = BASE_WRITE_FP + 'top_words_v' + str(iteration_num) + '.txt'
    f = open(top_words_file, "w")

    # write top 30 words per topic to file
    for item in model.show_topics(num_topics=topic_num, num_words=30, formatted=False):
        f.write(str(item) + '\n')
    f.write('\n\n\n')

    # write top 200 words per topic to file
    for item in model.show_topics(num_topics=topic_num, num_words=200, formatted=False):
        f.write(str(item) + '\n')

    # create second file with distribution of topics for each document
    doc_topics_file = BASE_WRITE_FP + 'doc_topics_v' + str(iteration_num) + '.txt'
    shutil.copyfile(model.fdoctopics(), doc_topics_file)


def optimize_topics(data, seeds, topic_num, iteration_num):
    """
    drives iterative topic modeling and writes results to file
    :param data: data for topics
    :param seeds: list of seed words
    :param topic_num: number of topics
    :param iteration_num: current version number
    :return: void
    """

    random.seed(1)

    # load data
    data_to_load = data
    dataset = load_flat_dataset(data_to_load, delimiter=' ')
    dataset.pop(0)

    # prepare documents for topic modeling
    filtered_dataset = filter_stopwords(dataset)
    corpus, dictionary = build_corpus(filtered_dataset)

    # prep seeds
    gtm_path = 'topics_helper/mallet-gtm/bin/mallet'
    general_seed_topics_file = seeds
    general_seed_topics_words = load_flat_dataset(general_seed_topics_file, delimiter=',')
    general_seed_weights = compute_idf_weights(dataset, general_seed_topics_words)

    model = GTMMallet(gtm_path, corpus, num_topics=topic_num, id2word=dictionary, alpha=1, beta=0.01, workers=1,
                      iterations=1000, seed_topics_file=general_seed_topics_file, over_sampling_factor=1,
                      seed_gpu_weights=general_seed_weights, )

    write_topics(model, topic_num, iteration_num)


def build_features(df, doc_topics, output_FP, topics):
    """
    adds document topic distribution to classification df as additional features
    :param df: full classification dataset
    :param doc_topics: topic distributions
    :param output_FP: filepath for output dataset
    :param topics: topics distribution columns
    :return: void
    """

    topic_dist = pd.read_table(doc_topics, header=None)
    df[topics] = topic_dist[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                             27, 28, 29, 30, 31, 32, 33, 34, 35, 36]]

    df.to_csv(output_FP, index=False)


def topic_inference(test_text, test, model, topics):
    """
    infers topic distribution for the test data based on the training data's topic model
    :param test_text: documents for modeling
    :param test: full testing dataset
    :param model: topic model
    :param topics: topic distribution columns
    :return: void
    """

    # load data
    test_data_to_load = test_text
    test_dataset = load_flat_dataset(test_data_to_load, delimiter=' ')
    test_dataset.pop(0)
    test = pd.read_csv(test)

    # prepare documents for topic modeling
    filtered_dataset = filter_stopwords(test_dataset)
    corpus, dictionary = build_corpus(filtered_dataset)

    # get predictions for test set
    predictions = model.__getitem__(corpus)

    # convert predictions into dataframe of predicted document distributions
    pred_transpose = pd.DataFrame()
    count = 0
    for pred in predictions:
        dist = []
        for a, b in pred:
            dist.append(b)
        pred_transpose[str(count)] = dist
        count += 1

    test_doc_dist = pred_transpose.T
    test_doc_dist.columns = topics

    # build topic features for classificaiton
    id = list(range(1, 334))
    test_doc_dist['id'] = id
    test['id'] = id
    test = pd.merge(test, test_doc_dist, on='id', how='inner')
    test = test.drop(columns=['id'])
    test = test.copy()

    test.to_csv('../data/cleaned/classification/testing.csv', index=False)


def topics_for_classification(train_text, test_text, train, train_semi, test, seeds, topic_num):
    """
    drives the final topic modeling run and creates topic features for classification.
    :param train_text: training data text
    :param test_text: testing data text
    :param train: full training dataset
    :param train_semi: full seed-biased training dataset
    :param test: full testing dataset
    :param seeds: final set of seed words
    :param topic_num: number of topics
    :return: void
    """

    random.seed(1)
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    # load data
    train_data_to_load = train_text
    train_dataset = load_flat_dataset(train_data_to_load, delimiter=' ')
    train_dataset.pop(0)
    train = pd.read_csv(train)

    topics = pd.read_csv('../data/cleaned/classification/topic_dist_cols.csv')
    topics = topics.columns

    # prepare documents for topic modeling
    filtered_dataset = filter_stopwords(train_dataset)
    corpus, dictionary = build_corpus(filtered_dataset)

    # prep seeds
    gtm_path = 'topics_helper/mallet-gtm/bin/mallet'
    general_seed_topics_file = seeds
    general_seed_topics_words = load_flat_dataset(general_seed_topics_file, delimiter=',')
    general_seed_weights = compute_idf_weights(train_dataset, general_seed_topics_words)

    # build model
    model = GTMMallet(gtm_path, corpus, num_topics=topic_num, id2word=dictionary, alpha=1, beta=0.01, workers=1,
                      iterations=1000, seed_topics_file=general_seed_topics_file, over_sampling_factor=1,
                      seed_gpu_weights=general_seed_weights, )

    shutil.copyfile(model.fdoctopics(), '../output/topics/final_doc_topics_training.txt')

    # build topic features for classificaiton
    build_features(train, '../output/topics/final_doc_topics_training.txt',
                   '../data/cleaned/classification/training.csv', topics)

    # add topic dist to semi supervised training data
    semi = pd.read_csv(train_semi)
    non_semi = pd.read_csv('../data/cleaned/classification/training.csv')
    semi[topics] = non_semi[topics]
    semi.to_csv('../data/cleaned/classification/training_biased.csv', index=False)

    # infer topics for test set
    topic_inference(test_text, test, model, topics)


def main():
    """

    :return: void
    """

    print('running topics.py main')


if __name__ == '__main__':
    main()
