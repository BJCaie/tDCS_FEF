Code to run behavioural and EEG analyses in the paper 'Intra-individual variability in the effects of transcranial direct current stimulation on free choice saccade behaviour' at https://www.biorxiv.org/content/10.1101/2024.08.23.609379v1

Data to run the analyses can be found at

Permutation tests were calculated offline and saved at (link). The code used to generate these results is found in the scripts permutationTestingEEG.py and permutationTestingPsychometrics.py

The remainder of the repository is dedicated to reproducing the results in the linked paper.

01_plot_behaviour.ipynb reproduces the behavioural data analyses. To run this notebook on a local machine, the group data file 'grouped_sessions.pkl', the 'Behavioural' folder, and the 'Psychometric Permutations' folder need to be downloaded. The path variable should be changed to reflect the storage location on your local machine

02_plot_EEG.ipynb reproduces the EEG analyses. To run this locally, the 'EEG' and 'EEG Permutations' folders need to be downloaded.

03_plot_correlations.ipynb reproduces the correlational analysis. 
