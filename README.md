# TabMap

### Interpretable Discovery of Patterns in Tabular Data via Spatially Semantic Topographic Maps

**TL;DR:** Python implementation of TabMap proposed in [our paper](). 

- TabMap unravels intertwined relationships in tabular data by transforming each data sample into a spatially semantic 2D topographic map, which we refer to as TabMap.
- A TabMap preserves the original feature values as pixel intensities, with the relationships among the features spatially encoded in the map (the strength of two inter-related features correlates with their distance on the map).
- Our approach makes it possible to apply 2D convolutional neural networks to extract association patterns in the data to facilitate data analysis, and offers interpretability by ranking features according to importance.
- We demonstrate TabMap's superior predictive performance across a diverse set of biomedical datasets.

## Table of Contents
- [Set up the Conda Environment](#set-up-the-conda-environment)
- [Train and evaluate the TabMap classifier](#train-and-evaluate-the-tabmap-classifier)
- [Notebook demo for TabMap visualization, classification and interpretation](#notebook-demo-for-tabmap-usage)

## Set Up the Conda Environment
```bash
git clone https://github.com/rui-yan/TabMap.git
cd TabMap
conda env create -f tabmap_conda.yml
conda activate tabmap
```

* NVIDIA GPU (Tested on Nvidia Quadro RTX 8000 48G x 1) on local workstations
* Python (3.10.13), torch (1.13.1), numpy (1.23.1), pandas (1.5.3), scikit-learn (1.4.2), scipy (1.10.1), seaborn (0.12.2); For further details on the software and package versions used, please refer to the `tabmap_conda.yml` file.

## Train and evaluate the TabMap classifier
### TabMap construction: transforming tabular data into 2D topographic maps
```python
from tabmap_construction import TabMapGenerator
generator = TabMapGenerator(metric='correlation', loss_fun='kl_loss')
generator.fit(features)
tabmaps = generator.transform(features)
```

#### Parameters:
* **metric: Metric used to compute the feature inter-relationships. *{'correlation', 'euclidean', 'gower'}***
* **loss_fun: Loss function used for computing the optimal transport. *{'kl_loss', 'sqeuclidean', 'square_loss'}***
* **epsilon: Entropic regularization parameter (>=0). default=0 (no regularization applied)***.
* **version: Version of the distance matrix calculation algorithm. default='v2.0'***.
Versions 'v1.0' and 'v2.0' use different methods for computing grid distances.
* **num_iter: Number of iterations for the optimal transport problem. default=10***.

#### TabMapGenerator class functions:
* **fit**(data, truncate=False): Computes the coupling matrix to map the feature space to the 2D map space. The `truncate` parameter determines whether to truncate or zero-pad the data to fit the 2D map.
* **transform**(data): Performs the mapping from feature space to image space.
* **fit_transform**(data, truncate=False): Fits the generator to the data and then performs the transformation.

### Train a 2D convolutional neural network (CNN) model for classification
```bash
cd code
python main.py
```
Refer to the [TabMap/code/main.py](https://github.com/rui-yan/TabMap/blob/main/code/main.py) file for details on model training and evaluation. This file also includes k-fold cross-validation, hyperparameter tuning, and comparisons with other classifiers used to generate the results presented in our paper.

## Notebook demo for TabMap visualization, classification and interpretation
TODO





