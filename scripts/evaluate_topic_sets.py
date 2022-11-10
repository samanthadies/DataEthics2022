from .tm_pipeline.evaluate_topic_set import compute_all_results


def get_param_string(param):
    return '_'.join([str(x).replace('.', '-') for x in param])


def main():
    dataset_names = ['sample_tweets']
    models = ['gtm']
    topn = 20

    gtm_params = [
        # (k, alpha, beta, seeds_file, sampling_scheme, over_sampling_factor), (k, alpha, beta, skew, noise_words_max, seeds_file, sampling_scheme, over_sampling_factor), topic weight
        [(50, 50, 0.01, 'data/{}_seed_topics.csv', 2, 1), (50, 50, 0.01, 25, 200, 'data/{}_seed_topics.csv', 1, 10), 10],
    ]

    for idx in range(0, len(dataset_names)):
            dataset_name = dataset_names[idx]
            results_path = 'results/{}_metrics.csv'.format(dataset_name)
            dataset_path = 'data/{}.csv'.format(dataset_name)

            seed_path = 'data/{}_seed_topics.csv'.format(dataset_name)
            final_path = 'results/{}/final/topics.csv'.format(dataset_name)
            gt_path = final_path
            topicset_paths = {}

            for model in models:
                ds_model_path = 'results/{}/{}/'.format(dataset_name, model)

                for params in gtm_params:
                    gtm_param_set = params[0]
                    topic_weight = params[2]
                    k = gtm_param_set[0]
                    ss = gtm_param_set[4]
                    osf = gtm_param_set[5]
                    label = 'gtm {} {}'.format(model, k, osf)
                    topicset_paths[label] = ds_model_path + 'topics_{}_{}_{}_{}.csv'.format(k, topic_weight, ss, osf)

            compute_all_results(topicset_paths, results_path, dataset_path, seed_path, final_path, gt_path, topn)


if __name__ == '__main__':
    main()
