from Scripts import behaviour, eegAnalysis
import numpy as np
import pickle 
from itertools import product

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


## Compute Group Average
for axnum, data_type in enumerate(data_splits): 
    for subjnum, subj in enumerate(subj_keys):
        anodal_dates = pre_sessions[subj]['AN'].keys()
        cathodal_dates = pre_sessions[subj]['CA'].keys()
        combinations = list(product(anodal_dates, cathodal_dates))
        for combnum, combination in enumerate(combinations):
            if combination[0] in anodal_dates:
                an_pre_1, an_pre_2, _ = eegAnalysis.subtractTFRBehaviour(TFR = pre_sessions[subj]['AN'][combination[0]]['eeg'], 
                                    behave = pre_sessions[subj]['AN'][combination[0]]['behave'] , data_type = data_type, 
                                    subj = subj, date = combination[0], stat = 'all', plotting = False)

                an_post_1, an_post_2, _ = eegAnalysis.subtractTFRBehaviour(TFR = post_sessions[subj]['AN'][combination[0]]['eeg'], 
                                    behave = post_sessions[subj]['AN'][combination[0]]['behave'] , data_type = data_type, 
                                    subj = subj, date = combination[0], stat = 'all', plotting = False)
                
                ca_pre_1, ca_pre_2, _ = eegAnalysis.subtractTFRBehaviour(TFR = pre_sessions[subj]['CA'][combination[1]]['eeg'], 
                                    behave = pre_sessions[subj]['CA'][combination[1]]['behave'] , data_type = data_type, 
                                    subj = subj, date = combination[1], stat = 'all', plotting = False)

                ca_post_1, ca_post_2, _ = eegAnalysis.subtractTFRBehaviour(TFR = post_sessions[subj]['CA'][combination[1]]['eeg'], 
                                    behave = post_sessions[subj]['CA'][combination[1]]['behave'] , data_type = data_type, 
                                    subj = subj, date = combination[1], stat = 'all', plotting = False)
            else:
                ca_pre_1, ca_pre_2, _ = eegAnalysis.subtractTFRBehaviour(TFR = pre_sessions[subj]['CA'][combination[0]]['eeg'], 
                                    behave = pre_sessions[subj]['CA'][combination[0]]['behave'] , data_type = data_type, 
                                    subj = subj, date = combination[0], stat = 'all', plotting = False)

                ca_post_1, ca_post_2, _ = eegAnalysis.subtractTFRBehaviour(TFR = post_sessions[subj]['CA'][combination[0]]['eeg'], 
                                    behave = post_sessions[subj]['CA'][combination[0]]['behave'] , data_type = data_type, 
                                    subj = subj, date = combination[0], stat = 'all', plotting = False)
                
                an_pre_1, an_pre_2, _ = eegAnalysis.subtractTFRBehaviour(TFR = pre_sessions[subj]['AN'][combination[1]]['eeg'], 
                                    behave = pre_sessions[subj]['AN'][combination[1]]['behave'] , data_type = data_type, 
                                    subj = subj, date = combination[1], stat = 'all', plotting = False)

                an_post_1, an_post_2, _ = eegAnalysis.subtractTFRBehaviour(TFR = post_sessions[subj]['AN'][combination[1]]['eeg'], 
                                    behave = post_sessions[subj]['AN'][combination[1]]['behave'] , data_type = data_type, 
                                    subj = subj, date = combination[1], stat = 'all', plotting = False)
            
            ## Compute null distributions
            grandTFR = np.concatenate([an_pre_1, an_pre_2, an_post_1, an_post_2, ca_pre_1, ca_pre_2, ca_post_1, ca_post_2])
            null = eegAnalysis.computePolarityNull(grandTFR, permutation_num, band = 'All')

            # Compute true mean distribution
            trueAN = (np.mean(an_pre_1, axis = 0) - np.mean(an_pre_2, axis = 0)) -  (np.mean(an_post_1, axis = 0) - np.mean(an_post_2, axis = 0))
            trueCA = (np.mean(ca_pre_1, axis = 0) - np.mean(ca_pre_2, axis = 0)) -  (np.mean(ca_post_1, axis = 0) - np.mean(ca_post_2, axis = 0))
            true = trueAN - trueCA

            ## Compute p values
            sig = eegAnalysis.permPValue(true, null)

            np.save(fr'{path}\TFR_subtractions\{subj}_session_combination_{combnum}_{data_type}_null', null)
            np.save(fr'{path}\TFR_subtractions\{subj}_session_combination_{combnum}_{data_type}{data_type}_true', true)
            np.save(fr'{path}\TFR_subtractions\{subj}_session_combination_{combnum}_{data_type}{data_type}_sig', sig)

##############################################################################################################

## Compute Session average
for axnum, data_type in enumerate(data_splits): 
    for subjnum, subj in enumerate(subj_keys):
        TFR = dict()
        for polarity in polarities:
            TFR1a, TFR2a, TFR1b, TFR2b, TFRnull =[np.empty([0, freqs.shape[0], t.shape[0]]) for i in range(5)]
            for date in pre_sessions[subj][polarity].keys():
                    
                    a1, b1, _ = eegAnalysis.subtractTFRBehaviour(TFR = pre_sessions[subj][polarity][date]['eeg'], 
                                        behave = pre_sessions[subj][polarity][date]['behave'] , data_type = data_type, 
                                        subj = subj, date = date, stat = 'all', plotting = False)

                    a2, b2, _ = eegAnalysis.subtractTFRBehaviour(TFR = post_sessions[subj][polarity][date]['eeg'], 
                                        behave = post_sessions[subj][polarity][date]['behave'] , data_type = data_type, 
                                        subj = subj, date = date, stat = 'all', plotting = False)
                    
                    TFR1a = np.concatenate([TFR1a, a1])
                    TFR2a = np.concatenate([TFR2a, a2])
                    TFR1b = np.concatenate([TFR1a, b1])
                    TFR2b = np.concatenate([TFR2b, b2])
                    TFRnull = np.concatenate([TFRnull, a1, a2, b1, b2])
                     
            TFR[polarity] = {"1a": TFR1a,
                             "1b": TFR1b,
                             "2a": TFR2a,
                             "2b": TFR2b,
                             "null": TFRnull}
        
        ## Combine nulls from both 
        grandTFR = np.concatenate([TFR['AN']['null'], TFR['CA']['null']])

        ## Compute Null Distributions
        null = eegAnalysis.computePolarityNull(grandTFR, permutation_num, band = 'All')

        # Compute true mean distribution
        trueAN = (np.mean(TFR['AN']['1a'], axis = 0) - np.mean(TFR['AN']['1b'], axis = 0)) -  (np.mean(TFR['AN']['2a'], axis = 0) - np.mean(TFR['AN']['2b'], axis = 0))
        trueCA = (np.mean(TFR['CA']['1a'], axis = 0) - np.mean(TFR['CA']['1b'], axis = 0)) -  (np.mean(TFR['CA']['2a'], axis = 0) - np.mean(TFR['CA']['2b'], axis = 0))
        true = trueAN - trueCA
        sig = eegAnalysis.permPValue(true, null)

        ## Save data
        np.save(fr'{path}\{subj}_{polarity}_polarity_subtraction_session_average_{data_type}_null', null)
        np.save(fr'{path}\{subj}_{polarity}_polarity_subtraction_session_average_{data_type}_true', true)
        np.save(fr'{path}\{subj}_{polarity}_polarity_subtraction_session_average_{data_type}_sig', sig)

##############################################################################################################

## Compute Session Permutations

data_splits = ['Choice', 'RT', 'Prev_Choice', 'Prev_RT', 'Rep']
polarities = ['AN', 'CA']
freqs = np.linspace(4, 30, 25)
t = np.linspace(0, .75, 384)
permutation_num = 1000
for axnum, data_type in enumerate(data_splits): 
    for subjnum, subj in enumerate(subj_keys):
        anodal_dates = pre_sessions[subj]['AN'].keys()
        cathodal_dates = pre_sessions[subj]['CA'].keys()
        combinations = list(product(anodal_dates, cathodal_dates))
        for combnum, combination in enumerate(combinations):
            if combination[0] in anodal_dates:
                an_pre_1, an_pre_2, _ = eegAnalysis.subtractTFRBehaviour(TFR = pre_sessions[subj]['AN'][combination[0]]['eeg'], 
                                    behave = pre_sessions[subj]['AN'][combination[0]]['behave'] , data_type = data_type, 
                                    subj = subj, date = combination[0], stat = 'all', plotting = False)

                an_post_1, an_post_2, _ = eegAnalysis.subtractTFRBehaviour(TFR = post_sessions[subj]['AN'][combination[0]]['eeg'], 
                                    behave = post_sessions[subj]['AN'][combination[0]]['behave'] , data_type = data_type, 
                                    subj = subj, date = combination[0], stat = 'all', plotting = False)
                
                ca_pre_1, ca_pre_2, _ = eegAnalysis.subtractTFRBehaviour(TFR = pre_sessions[subj]['CA'][combination[1]]['eeg'], 
                                    behave = pre_sessions[subj]['CA'][combination[1]]['behave'] , data_type = data_type, 
                                    subj = subj, date = combination[1], stat = 'all', plotting = False)

                ca_post_1, ca_post_2, _ = eegAnalysis.subtractTFRBehaviour(TFR = post_sessions[subj]['CA'][combination[1]]['eeg'], 
                                    behave = post_sessions[subj]['CA'][combination[1]]['behave'] , data_type = data_type, 
                                    subj = subj, date = combination[1], stat = 'all', plotting = False)
            else:
                ca_pre_1, ca_pre_2, _ = eegAnalysis.subtractTFRBehaviour(TFR = pre_sessions[subj]['CA'][combination[0]]['eeg'], 
                                    behave = pre_sessions[subj]['CA'][combination[0]]['behave'] , data_type = data_type, 
                                    subj = subj, date = combination[0], stat = 'all', plotting = False)

                ca_post_1, ca_post_2, _ = eegAnalysis.subtractTFRBehaviour(TFR = post_sessions[subj]['CA'][combination[0]]['eeg'], 
                                    behave = post_sessions[subj]['CA'][combination[0]]['behave'] , data_type = data_type, 
                                    subj = subj, date = combination[0], stat = 'all', plotting = False)
                
                an_pre_1, an_pre_2, _ = eegAnalysis.subtractTFRBehaviour(TFR = pre_sessions[subj]['AN'][combination[1]]['eeg'], 
                                    behave = pre_sessions[subj]['AN'][combination[1]]['behave'] , data_type = data_type, 
                                    subj = subj, date = combination[1], stat = 'all', plotting = False)

                an_post_1, an_post_2, _ = eegAnalysis.subtractTFRBehaviour(TFR = post_sessions[subj]['AN'][combination[1]]['eeg'], 
                                    behave = post_sessions[subj]['AN'][combination[1]]['behave'] , data_type = data_type, 
                                    subj = subj, date = combination[1], stat = 'all', plotting = False)
            
            ## Compute null distributions
            grandTFR = np.concatenate([an_pre_1, an_pre_2, an_post_1, an_post_2, ca_pre_1, ca_pre_2, ca_post_1, ca_post_2])
            null = eegAnalysis.computePolarityNull(grandTFR, permutation_num, band = 'All')

            # Compute true mean distribution
            trueAN = (np.mean(an_pre_1, axis = 0) - np.mean(an_pre_2, axis = 0)) -  (np.mean(an_post_1, axis = 0) - np.mean(an_post_2, axis = 0))
            trueCA = (np.mean(ca_pre_1, axis = 0) - np.mean(ca_pre_2, axis = 0)) -  (np.mean(ca_post_1, axis = 0) - np.mean(ca_post_2, axis = 0))
            true = trueAN - trueCA

            ## Compute p values
            sig = eegAnalysis.permPValue(true, null)

            np.save(fr'{path}\TFR_subtractions\{subj}_session_combination_{combnum}_{data_type}_null', null)
            np.save(fr'{path}\TFR_subtractions\{subj}_session_combination_{combnum}_{data_type}{data_type}_true', true)
            np.save(fr'{path}\TFR_subtractions\{subj}_session_combination_{combnum}_{data_type}{data_type}_sig', sig)
