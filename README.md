# bvnGPS

bvnGPS: The official implementation of "A robust antibiotics decision model based on integrative host transcriptomics and multi-class neural network".

Input: 

Host gene expression cohorts from Gen Expression Omnibus (The data in paper could be download and labeled with `load_data/download_and_save_data.R`) 

Output: 

1. bacterial GPS, viral GPS, and noninfected GPS;
2. decision tree, random forest and support vector machines based on GPS selected;
3. AUC performance of Decision tree, random forest and support vector on internal test data and external validation data. 