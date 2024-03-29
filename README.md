# bvnGPS

bvnGPS: The official implementation of "**bvnGPS: a generalizable diagnostic model for acute bacterial and viral infection using integrative host transcriptomics and pretrained neural networks**".

Input: 

Host gene expression cohorts from Gene Expression Omnibus (The data in the paper could be downloaded and labeled with `load_data/download_and_save_data.R`) 

Output: 

1. bacterial GPS, viral GPS, and noninfected GPS;
2. decision tree, random forest, and support vector machines based on GPS selected;
3. AUC performance of decision tree, random forest, and support vector machines on internal test data and external validation data. (Other performance could be calculated with `visualization/`, such as accuracy, sensitivity, specificity, and ROC curve)

