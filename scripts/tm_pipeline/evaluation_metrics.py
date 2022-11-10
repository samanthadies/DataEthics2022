import math
from statistics import mean


def split_ngrams(topics):
    split_topics = []
    for topic in topics:
        split_topic = []
        for x in topic:
            l = x.split('$')
            l.append(x)
            split_topic.extend(l)
        split_topic = list(set(split_topic))
        split_topics.append(split_topic)
    return split_topics


def words_in_topic(test, gt, topn=20):
    words = 0
    for word in test[:topn]:
        if word in gt:
            words += 1
    return words


def classify_document(d, topics, topn):
    words_per_topic = [words_in_topic(d, x, topn) for x in topics]
    if max(words_per_topic) == 0:
        return -1
    return words_per_topic.index(max(words_per_topic))


def topic_classification(topics, dataset, topn=20):
    '''
    classify each document in a data set with one or none of the provided topics
    :param dataset:
    :param topics:
    :param topn:
    :return: list of topic counts, incremented each time a doc is classified as that topic
    '''
    class_count = [0]*len(topics)
    for d in dataset:
        klass = classify_document(d, topics, topn)
        if klass != -1:
            class_count[klass] += 1
    return class_count


def topic_classification_score(topics, dataset, topn=20):
    classes = topic_classification(topics, dataset, topn)
    return sum(classes) / len(dataset)


def topic_improvement(topics, seeds):
    '''
    return % increase in topic size from seeds to final topics
    :param topics:
    :param gt_topics:
    :param topn:
    :return:
    '''
    topic_improvements = []
    for i in range(0, len(topics)):
        topic = topics[i]
        seed = ['']
        if i < len(seeds):
            seed = seeds[i]
        topic_improvements.append(round(len(topic) - len(seed)/len(seed), 2))
    return topic_improvements


def topic_improvement_score(topics, seeds):
    return sum(topic_improvement(topics, seeds)) / len(topics)


def entropy(topic, final_topic, topn=50, phi=1):
    fake = []
    for i in range(0, len(topic)):
        w = topic[i]
        if w in final_topic:
            fake.append(w)
        if len(fake) >= len(final_topic)*phi:
            return i
    # print(set(final_topic) - set(fake))
    return topn


def topic_entropy(topics, final_topics, topn=50, phi=1):
    entropies = []
    for i in range(0, len(final_topics)):
        final_topic = final_topics[i]
        ent = min([entropy(x, final_topic, topn=topn, phi=phi) for x in topics])
        entropies.append(ent)
    # print(entropies)
    return mean(entropies), min(entropies), max(entropies)


def rank_to_band_old(rank):
    if rank <= 50:
        return 1
    elif rank <=100:
        return 5
    elif rank <= 500:
        return 20
    elif rank <= 5000:
        return 40
    elif rank <= 50000:
        return 60
    elif rank <= 500000:
        return 80
    else:
        return 100


def rank_to_band(rank):
    return 2*len(str(rank))


def word_rank(w, topic, topn=50):
    if w in topic:
        return topic.index(w) + 1
    return topn


def topic_rank_final(topics, final_topic, topn=50, num_ranks=3):
    min_rank = 100*len(final_topic)
    for topic in topics:
        ranks = []
        for i in range(0, len(final_topic)):
            w = final_topic[i]
            ranks.append(rank_to_band(word_rank(w, topic, topn)))
        rank = sum(ranks)
        # ranks = sorted(ranks)
        # check_rank = num_ranks
        # if num_ranks >= len(ranks):
        #     check_rank = len(ranks) - 1
        # rank = ranks[check_rank]
        if rank < min_rank:
            min_rank = rank
    return min_rank


def topic_rank(topics, final_topics, topn=50, num_ranks=3):
    '''

    :param topics:
    :param final_topics:
    :param topn:
    :param num_ranks: the nth word based on rank to consider.  If num_ranks = 5, take the 5th ranked word in the topic
    :return:
    '''
    ranks = []
    for i in range(0, len(final_topics)):
        final_topic = final_topics[i]
        ranks.append(topic_rank_final(topics, final_topic, topn, num_ranks))
    # print(num_ranks, mean(ranks), ranks)
    return sum(ranks), mean(ranks)


def topic_coverage(topics, gt_topics, dataset_name, label, topn=20):
    '''
    avg percent of words from each gt_topic captured by best topic in topics
    :param topics:
    :param gt_topics:
    :param topn:
    :return:
    '''
    coverage = []
    for i in range(0, len(gt_topics)):
        topic = gt_topics[i]
        coverage.append(max([words_in_topic(topic, x, topn) for x in topics]) / len(topic))
    # print(coverage)
    with open('results/icdm_paper/{}_recall.csv'.format(dataset_name), 'a') as f:
        cov = [f'{x:.2f}' for x in coverage]
        f.write('{},{}\n'.format(label, ','.join(cov)))
    return mean(coverage), min(coverage), max(coverage)


def topic_coverage_old(topics, gt_topics, topn=20):
    '''
    avg percent of words in top20 that belong to gt_topic
    :param topics:
    :param gt_topics:
    :param topn:
    :return:
    '''
    coverage = []
    for i in range(0, len(gt_topics)):
        gt_topic = gt_topics[i]
        topic = topics[i]
        coverage.append(words_in_topic(topic, gt_topic, topn)/min(topn, len(topic)))
    # print(coverage)
    return mean(coverage)


def mean_cof(topic, token, cofrequencies):
    if len(topic) < 2:
        return 0
    cof_count = 0
    for w in topic:
        if token != w:
            word_tup = tuple(sorted([token, w]))
            if word_tup in cofrequencies:
                cof_count += cofrequencies[word_tup]
    if token in topic:
        return cof_count / (len(topic) - 1)
    return cof_count / len(topic)


def silhouette(T, topic, token, cofrequencies):
    '''
    Maximizing mean cofrequency instead of minimizing distance.  Silhouette value of a given token from the given topic
    :param T: topic set
    :param topic: home topic of queried token
    :param token: queried token we wish to get silhouette value for
    :param cofrequencies: dictionary of cofrequencies in data set
    :return:
    '''
    a = mean_cof(topic, token, cofrequencies)
    b = 0
    for i in range(0, len(T)):
        t_i = T[i]
        if t_i != topic:
            topic_score = mean_cof(t_i, token, cofrequencies)
            if topic_score > b:
                b = topic_score
    if a == b:
        return 0
    return (a - b) / max(a, b)


def topic_silhouette(T, topic, cofrequencies):
    silhouettes = []
    for w in topic:
        silhouettes.append(silhouette(T, topic, w, cofrequencies))
    return silhouettes


def topicset_silhouettes(T, cofrequencies):
    silhouettes = []
    for topic in T:
        s = topic_silhouette(T, topic, cofrequencies)
        silhouettes.append(s)
    return silhouettes


def npmi(topic, frequencies, cofrequencies):
    v = 0
    x = max(2, len(topic))
    for i in range(0, len(topic)):
        w_i = topic[i]
        p_i = 0
        if w_i in frequencies:
            p_i = frequencies[w_i]
        for j in range(i+1, len(topic)):
            w_j = topic[j]
            p_j = 0
            if w_j in frequencies:
                p_j = frequencies[w_j]
            word_tup = tuple(sorted([w_i, w_j]))
            p_ij = 0
            if word_tup in cofrequencies:
                p_ij = cofrequencies[word_tup]
            if p_ij < 2:
                v -= 1
            else:
                pmi = math.log(p_ij / (p_i * p_j), 2)
                denominator = -1 * math.log(p_ij, 2)
                v += (pmi / denominator)
    return (2*v) / (x*(x-1))


def topic_npmis(T, frequencies, cofrequencies, topn=20):
    npmis = []
    for topic in T:
        n = npmi(topic[:topn], frequencies, cofrequencies)
        npmis.append(n)
    return npmis


def topic_coherence(T, frequencies, cofrequencies, topn=20):
    '''
    Computes the coherence of a topic set (average NPMI of topics)
    :param T:
    :param frequencies:
    :param cofrequencies:
    :param k: top-k words per topic to consider
    :return:
    '''
    npmis = topic_npmis(T, frequencies, cofrequencies, topn)
    if len(npmis) > 0:
        return mean(npmis)
    return 0


def topic_diversity(T, topn=20):
    '''
    fraction of words in top-n words of each topic that are unique
    :param T:
    :param k: top k words per topic
    :return:
    '''
    top_words = []
    for topic in T:
        top_words.extend(topic[:topn])
    unique_words = set(top_words)
    if len(top_words) > 0:
        return len(unique_words)/len(top_words)
    return 0


def noise_penetration(T, noise_words, topn=20):
    '''
    fraction of words in top-k words of each topic that are noise words
    :param T:
    :param noise_words: set of noise words
    :param k: top-k words of each topic
    :return:
    '''
    noise_count = 0
    for topic in T:
        for w in topic[:topn]:
            if w in noise_words:
                noise_count += 1
    if len(T) > 0:
        return noise_count / (len(T) * topn)
    return 1
