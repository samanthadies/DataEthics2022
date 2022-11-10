from settings.common import (load_flat_dataset, load_topics, get_vocabulary,
                             word_co_frequency, word_frequency)
from .evaluation_metrics import (topic_coherence, topic_diversity, topic_entropy,
                                 topic_improvement_score, topic_classification_score, topic_coverage)


def save_metric_results(path, label, scores):
    for i in range(0, len(scores)):
        scores[i] = str(round(scores[i], 3))
    with open(path, 'a') as f:
        f.write('{},{}\n'.format(label, ','.join(scores)))


def analyze_dataset(dataset_path, seed_path=None, gt_path=None):
    dataset = load_flat_dataset(dataset_path)
    entropy_topn = len(get_vocabulary(dataset))
    freqs = {}
    freqs = word_frequency(freqs, dataset)
    cofreqs = {}
    cofreqs = word_co_frequency(cofreqs, dataset)
    seed_words = []
    if seed_path:
        seed_words = load_topics(seed_path)
    gt_words = []
    if gt_path:
        gt_words = load_topics(gt_path)
    return dataset, freqs, cofreqs, seed_words, gt_words, entropy_topn


def compute_metrics(topics, dataset, freqs, cofreqs, seed_topics, gt_topics, dataset_name, label, topn=20, entropy_topn=20, phi=1):
    coherence_score = topic_coherence(topics, freqs, cofreqs, topn)
    diversity_score = topic_diversity(topics, topn)
    coverage_score, coverage_min, coverage_max = topic_coverage(topics, gt_topics, dataset_name, label, topn=topn)
    entropy_score, entropy_min, entropy_max = topic_entropy(topics, gt_topics, topn=entropy_topn, phi=phi)
    improvement_score = topic_improvement_score(topics, seed_topics)
    classification_score = topic_classification_score(topics, dataset)
    return [coherence_score, diversity_score, coverage_score, improvement_score, classification_score, coverage_min, coverage_max, entropy_score, entropy_min, entropy_max]

def compute_all_results(topicset_paths, results_path, dataset_path, seed_path, final_path, gt_path, topn=20):
    '''

    :param results_paths: Dict of key = topicset label, value = topicset path
    :param dataset_path:
    :param noise_path:
    :param topn:
    :return:
    '''
    dataset_name = dataset_path.replace('data/', '').replace('.csv', '')
    dataset, freqs, cofreqs, seed_topics, gt_topics, entropy_topn = analyze_dataset(dataset_path, seed_path, gt_path)
    final_topics = load_topics(final_path)
    for topicset_label in topicset_paths.keys():
        topicset_path = topicset_paths[topicset_label]
        topics = load_topics(topicset_path)
        topics = [x for x in topics if len(x) > 0]
        scores = compute_metrics(topics, dataset, freqs, cofreqs, seed_topics, final_topics, dataset_name, topicset_label, topn, entropy_topn)
        save_metric_results(results_path, topicset_label, scores)
