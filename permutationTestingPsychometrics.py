from Scripts import DID, behaviour
import numpy as np
import pickle 
from itertools import product
import pandas as pd

## Setup 
path = r'E:\PhD\FEF'
subj_keys =  behaviour.get_immediate_subdirectories(fr'{path}\Final Data')
data_splits = ['Choice', 'RT', 'Prev_Choice', 'Prev_RT', 'Rep']
polarities = ['AN', 'CA']
freqs = np.linspace(4, 30, 25)
t = np.linspace(0, .75, 384)
permutation_num = 1000

## Load in EEG Data and set 
with open(fr'{path}\Combined_Data\pre_sessions.pkl', 'rb') as f:
    pre_sessions = pickle.load(f)
with open(fr'{path}\Combined_Data\post_sessions.pkl', 'rb') as f:
    post_sessions = pickle.load(f)


## Group-average psychometric permutations
pre = dict()
post = dict()
for polarity in polarities:
    pre[polarity] = list()
    post[polarity] = list()
    for subjnum, subj in enumerate(subj_keys):
        for date in pre_sessions[subj][polarity].keys():
            pre[polarity].append(pre_sessions[subj][polarity][date]['behave'])
            post[polarity].append(post_sessions[subj][polarity][date]['behave'])

    pre[polarity] = pd.concat([name for name in pre[polarity]])
    post[polarity] = pd.concat([name for name in post[polarity]])

true, null = DID.permutePsychometrics(pre['AN'], pre['CA'], post['AN'], post['CA'])
np.save(fr'{path}\group_average_null', null)
np.save(fr'{path}\group_average_true', true)

## Single Subject
for subj in subj_keys:
    DID.permuteSubject(subj, pre_sessions, post_sessions)

## Single Subject Choice History
conditions = ['Prev_Choice', 'Rep', 'Prev_RT']
for condition in conditions:
    for subj in subj_keys:
        DID.permuteHistory(condition = condition, pre_sessions= pre_sessions, post_sessions = post_sessions, 
                        subj_keys = [subj], split_type = subj)


## Session combinations
DID.permuteCombinations(subj_keys, pre_sessions, post_sessions)
