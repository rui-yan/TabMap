### Data preparation

Links to each of the datasets:
- parkinson: https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification
- qsar: https://archive.ics.uci.edu/dataset/254/qsar+biodegradation
- har: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- microMass: https://archive.ics.uci.edu/dataset/253/micromass
- arcene: https://archive.ics.uci.edu/dataset/167/arcene
- isolet: https://archive.ics.uci.edu/dataset/54/isolet
- p53: https://archive.ics.uci.edu/dataset/188/p53+mutants
- wdbc: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- bctil: https://singlecell.broadinstitute.org/single_cell/study/SCP2331/single-cell-profiling-of-breast-cancer-t-cells-reveals-a-tissue-resident-memory-subset-associated-with-improved-prognosis#study-summary
- lung: https://github.com/jundongl/scikit-feature/blob/master/skfeature/data/lung.mat
- tox-171: https://github.com/jundongl/scikit-feature/blob/master/skfeature/data/TOX-171.mat


After downloading the datasets, convert their format into a CSV file.
Then, create a folder named `DATASET_NAME` and save the corresponding CSV file as `data_raw.csv` in that folder. This ensures compatibility with the data loading functionality implemented in our code.
Four example datasets formatted this way are available in the [data folder](https://github.com/rui-yan/TabMap/tree/main/data) for reference. 

Alternatively, you can clone the repository and customize the [data loading code](https://github.com/rui-yan/TabMap/blob/main/tabmap/dataloader/dataset.py) to meet your needs.