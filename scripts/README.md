# Guided Topic Model
Implementations of Guided Topic Model (GTM) and Topic Noise Discriminator (TND), along with our evaluation metrics, can be found here.

### Requirements and Setup
to install relevant Python requirements:
> pip install -r requirements.txt

You must have the Java JDK installed on your computer to run TND. It can be downloaded [here](https://www.oracle.com/java/technologies/javase-downloads.html).  We originally built this with JDK 11, but have tested with JDK 8 and 16.

### Using TND and NLDA
All of the code in this section, with the exception of pseudocode, is included in the script `readme_script.py`.  There is another script called `run_models.py` that can be used to test models at scale using sets of parameters.

**Loading and Preparing Data for Modeling.**
Data sets should be loaded as a list of documents, where each document is a list of words.  We have a built-in function to load data.  We also convert the data set to a gensim corpus for use in our models.
```python
# dataset = [['this', 'is', 'doc', '1'], ['this', 'is', 'doc', '2']]
from gensim import corpora
from settings.common import load_flat_dataset

dataset = load_flat_dataset('data/sample_tweets.csv', delimiter=' ')
dictionary = corpora.Dictionary(dataset)
dictionary.filter_extremes()
corpus = [dictionary.doc2bow(doc) for doc in dataset]
```

**Parameters.**
Lines 230-249 of `run_models.py` contain example parameter settings for each of our models.
Here, we explain the parameters of TND and then of NLDA.

**TND Parameters.**
* k: the number of topics to approximate
* alpha: hyper-parameter tuning number of topics per document
* beta (Beta_0 in the paper): hyper-parameter tuning number of topics per word
* skew (Beta_1 in the paper): hyper-parameter tuning topic probability of a given word compared to its noise probability. Higher means more weight is given to the topic (words are less likely to be noise).
* noise_words_max: the number of noise words to save to a noise words file for use as a context noise list (words with highest probability of being noise)
* iterations: the number of iterations to run inference of topics

**TND Parameters for Embedding Space Incorporation.**
The version of TND that employs embedding spaces is split out in the Java implementation (based on the Mallet LDA implementation) to simplify the code base and keep the non-embedding version as fast as possible.
* embedding_path: the path to the trained embedding space
* closest_x_words (mu in the paper): the number of words to be sampled from the embedding space for each observed word (based on distance in the embedding space)
* tau: the number of iterations before using the embedding space (aka burnin period)

**GTM Parameters.**


**Running GTM using a wrapper**

We used the old Gensim wrapper for Mallet LDA to create a wrapper for GTM.
The `workers` parameter is the number of threads to dedicate to running the model.  We have found that four is sufficient for many mid-sized data sets on our servers.
TND and eTND can be used as follows (assuming we've already loaded and prepped our data):
```python
from tm_pipeline.gtmmallet import GTMMallet

gtm_path = 'mallet-gtm/bin/mallet'
tnd_path = 'mallet-gtm/bin/mallet'

model1 = GTMMallet(tnd_path, corpus, num_topics=30, id2word=dictionary, workers=4,
                   alpha=50, beta=0.01, skew=25, noise_words_max=200, iterations=1000, seed_topics_file='data/seed_topics.csv', over_sampling_factor=1)

topics = model1.show_topics(num_topics=30, num_words=20, formatted=False)
```

**Running GTM using an object**

We can run GTM using its class in `tm_pipeline`.
```python
from tm_pipeline.gtm import GTM

model = GTM(dataset=dataset, tnd_k=30, tnd_alpha=50, tnd_beta0=0.01, tnd_beta1=25, tnd_noise_words_max=200,
                 tnd_iterations=1000, gtm_iterations=1000, gtm_k=30, phi=10, topic_depth=100, top_words=20,
                 save_path='results/', mallet_tnd_path=tnd_path, mallet_gtm_path=gtm_path, random_seed=1824, run=True,
                seed_topics_file='data/seed_topics.csv', over_sampling_factor=1)
```

Setting `run=True` here (the default) will result in GTM being run through on initialization.  Setting it to false allows one to go through the model one step at a time, like so:
```python 
    model.prepare_data()
    model.compute_tnd()
    model.compute_gtm()
    model.filter_noise()
```

We can also pass through a pre-computed TND or LDA model, or both.
```python
from tm_pipeline.gtmmallet import GTMMallet
from tm_pipeline.tndmallet import TndMallet

gtm_model = GTMMallet(dataset, other_parameters)
tnd_model = TndMallet(dataset, other_parameters)

nlda = GTM(dataset, tnd_noise_distribution=tnd_model.load_noise_dist(), 
            gtm_tw_dist=gtm_model.load_word_topics(), phi=10, 
            topic_depth=100, save_path='results/', run=True)
```


**Testing many configurations at once.** 
In performing research on topic models, we often want to run a bunch of model parameter settings at once.  In the `run_models.py` file, we have wrappers for Mallet TND (and the embedded version, eTND) and GTM.  These are essentially deconstructed versions of the TND and GTM classes that allow for easier customization and for a lot of repeated experiments.

### Referencing GTM and TND
####Guided Topic Model Citations:
```
Churchill, Rob and Singh, Lisa and Ryan, Rebecca and Davis-Kean, Pamela. 2022. A Guided Topic Model for Social Media Data. The Web Conference (WWW).
```

```bibtex 
@inproceedings{churchill2022gtm,
author = {Churchill, Rob and Singh, Lisa and Ryan, Rebecca and Davis-Kean, Pamela},
title = {A Guided Topic Model for Social Media Data},
booktitle = {The Web Conference (WWW)},
year = {2022},
}
```

####Topic-Noise Discriminator Citations:
```
Churchill, Rob and Singh, Lisa. 2021. Topic-Noise Models: Modeling Topic and Noise Distributions in Social Media Post Collections. International Conference on Data Mining (ICDM).
```

```bibtex 
@inproceedings{churchill2021tnd,
author = {Churchill, Rob and Singh, Lisa},
title = {Topic-Noise Models: Modeling Topic and Noise Distributions in Social Media Post Collections},
booktitle = {ICDM 2021},
year = {2021},
}

### Citations
```
A. K. McCallum, “Mallet: A machine learning for language toolkit.”
2002.
```

```
P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, "Enriching Word Vectors with Subword Information." 2016.
```

```
R. Rehurek, P. Sojka, "Gensim–python framework for vector space modelling." 2011.
```
