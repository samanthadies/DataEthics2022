# DataEthics2022

short sentence describing repo

### Data

The current labeled data for the transparency topic modeling task can be found [here](https://github.com/samanthadies/DataEthics2022/tree/main/data). [Raw](https://github.com/samanthadies/DataEthics2022/tree/main/data/raw) contains the unproccessed data which is fed into [eda.py](https://github.com/samanthadies/DataEthics2022/blob/main/scripts/eda.py) and [textCleaning.py](https://github.com/samanthadies/DataEthics2022/blob/main/scripts/textCleaning.py) via the driver script, in addition to the final set of seed words for the topic modeling component of the task. [cleaned/classification/](https://github.com/samanthadies/DataEthics2022/tree/main/data/cleaned/classification) contains the processed, labeled data used in the first attempts at ethically-conscious transparency modeling.

### Scripts

#### main.py
This script is the main driver program for the transparency modeling task. It starts by running Phase 1 which includes exploratory data analysis and preprocessing. Phase 2, the iterative topic modeling stage, is then executed. This stage should be repeated until the user is satisfied with the final set of topics and seed words. Phase 3 is the final topic modeling run and the classification step. Lastly, Phase 4 conducts analysis and produces visualizations based on the classification results.

#### eda.py
This file computes basic statistics of a number of attributes from [dslc_full_final.csv](https://github.com/samanthadies/DataEthics2022/blob/main/data/raw/dslc_full_final.csv) and outputs the summary statistics. The program also creats appropriate visualizations of the data.

#### textCleaning.py
This file cleans the text training data for the scoring criteria by removing punctuation, numbers, and extra white spaces, lemmatizing, lowercasing, and converting the score into binary.

#### topics.py
This file serves two purposes. First, it can be used for iterative topic modeling using GTM. It should be run as many times as necessary to optimize topics and seed word list. Second, it should be run with the classification datasets and is used to build the topic features.

#### ml.py
This file runs classification experiments at the section level for each transparency criteria class label. We build models following two model configuration: single-stage and multi-stage. We use four datasets, the baseline (text) data, the topics data (topic distributions), and seed-biased (text with seeds) data. We then identify the best model for each criteria and generate paper level scores using the best models.

### Output

The output from the above scripts is categorized as eda, topics, classification, and visualizations. The [eda output](https://github.com/samanthadies/DataEthics2022/tree/main/output/eda) contains summary statistics and figures with feature and class label distributions. The [topics output](https://github.com/samanthadies/DataEthics2022/tree/main/output/topics) contains top words and topic distributions. The [classification output](https://github.com/samanthadies/DataEthics2022/tree/main/output/classification) contains results from each model run. Finally, the [visualizations output](https://github.com/samanthadies/DataEthics2022/tree/main/output/visualizations) contains a heatmap of the transparency scores.

### Acknowledgements

### References
1. Churchill, Rob and Singh, Lisa. 2022. A Guided Topic-Noise Model for Short Texts. The Web Conference (WWW).
