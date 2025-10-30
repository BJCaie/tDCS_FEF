import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from scipy.signal import butter, lfilter
from meegkit.detrend import detrend
from scipy import signal
from itertools import compress
from Scripts import behaviour
import re
from sklearn.model_selection import train_test_split
import pickle 

def getSessions(eeg_path, behavioural_path, subj_keys, stim_type):
    """Returns individual participant EEG data according to condition and alignment type

    Args:
        eeg_path (str): path where eeg data is stored
        behavioural_path (str): string indicating the participant to be selected
        subj_keys (list): list of participant ids
        stim_type (str): 'PR'/'ST'/'PO' 

    Returns:
        sessions (dict): nested dictionary. if polarity_type is 'Split' ,of shape 'sessions[subj][polarity][date]'
                                            if polarity_type is of 'All', shape 'sessions[subj][date]'

    """
    eeg_directory = os.listdir(eeg_path)
    full_list = os.listdir(behavioural_path)
    sessions = dict()
    alignment = 'trial_aligned'
    freq_band = 'All'
    polarities = ['AN', 'CA']

    for subj in subj_keys:
        subj_list = [x for x in full_list if re.search(subj, x)]
        date_list = list(set([item[10:20] for item in subj_list]))
        sessions[subj] = dict()
        for polarity in polarities:
            sessions[subj][polarity] = dict()
            for date in date_list:
                session_list = [all([k in s for k in [date, subj]]) for s in full_list]
                session_list = list(compress(full_list, session_list))
                if polarity in session_list[0]:
                    sessions[subj][polarity][date] = dict()
                    sessions[subj][polarity][date]['behave'] = 0
                    sessions[subj][polarity][date]['eeg'] = 0
                    block_list = [all([k in s for k in [stim_type]]) for s in session_list]
                    block_list = list(compress(session_list, block_list))
                    for block in block_list:
                        for fname in eeg_directory:
                            if fname[0:20] in block:
                                if alignment in fname:
                                    eeg = pd.read_csv(eeg_path + "/" + fname, index_col= False)
                                    eeg = np.delete(eeg.to_numpy(), 0 , 1)
                                    TFR = computeTFR(eeg, plotting = False, freq_band = freq_band)
                                    behave = pd.read_csv(behavioural_path + '/' + block, index_col = False)
                                    ## NEED TO FILTER OUT BAD TRIALS IN EEG HERE
                                    if pd.read_csv(behavioural_path + '/' + block, index_col = False).shape[0] == TFR.shape[0]:
                                            if isinstance(sessions[subj][polarity][date]['behave'], int) is True:
                                                sessions[subj][polarity][date]['behave'] = behave
                                                sessions[subj][polarity][date]['eeg'] = TFR
                                            else:
                                                sessions[subj][polarity][date]['behave'] = pd.concat([sessions[subj][polarity][date]['behave'], behave])
                                                sessions[subj][polarity][date]['eeg'] = np.concatenate((sessions[subj][polarity][date]['eeg'], TFR))
                                    else:
                                        print('mismatchedtrial')
                    if isinstance(sessions[subj][polarity][date]['eeg'], int) is True:
                        print('huh')
    return sessions

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def computeTFR(eeg, freq_band, plotting):
    numfreqs = 25
    if freq_band == 'Alpha':
        freq = np.linspace(8, 12, numfreqs)
    elif freq_band == 'Beta':
        freq = np.linspace(15, 30, numfreqs)
    elif freq_band == 'Theta':
        freq = np.linspace(4, 7, numfreqs)
    else:
        freq = np.linspace(4, 30, numfreqs)

    fs = 512
    w = 5.
    widths = w*fs / (2*freq*np.pi)
    t = np.linspace(0, .75, 384)
    cwtms = np.zeros([eeg.shape[0], numfreqs, eeg.shape[1]])
    
    ## Compute wavelet transform for each trial
    for trial in range(eeg.shape[0]):
        sig, _, _ = detrend(eeg[trial,:], 3) # Detrend the data
        cwtm = signal.cwt(sig, signal.morlet2, widths, w=w) 
        cwtms[trial,:,:] = np.abs(cwtm)
        for freq in range(numfreqs):
            cwtms[trial,freq,:] = cwtms[trial,freq,:] - np.mean(cwtms[trial,freq,1:50])
            
    if plotting == True:
        plt.figure()
        plt.pcolormesh(t, freq, np.mean(cwtms, axis = 0), cmap='viridis', shading='gouraud')
        plt.colorbar()

    return cwtms

def subtractTFRBehaviour(TFR, behave, data_type, subj, date, plotting, stat):
    """
    Computes TFR subtraction (avg z score difference) for Choice, reaction time (median split), individually and in combination
    eeg_path (str): directory of eeg .csv files
    behavioural_path (str): directory of behaviour files
    subj _keys (dict): list of subjects
    data_type (str): 'RT for reaction times, 'Choice' for choice, 'Choice_RT' for combination
    """
    freq = np.linspace(4, 30, 25)
    t = np.linspace(0, .75, 384)

    if data_type == 'RT':
        RT = np.array(behave['Reaction Time: First Target'])
        ind = RT < np.nanpercentile(RT, 50)

    if data_type == 'Prev_RT':
        prevRT = np.array(behave['Reaction Time: First Target'].shift(1))
        ind = prevRT< np.nanpercentile(prevRT, 50)

    elif data_type == 'Choice':
        ind = behave['Choice'] == 0

    elif data_type == 'Prev_Choice':
        prevChoice = np.array(behave['Choice'].shift(1))
        ind = prevChoice == 0

    elif data_type == 'Rep':
        ind = behave['Rep Number'] == 1

    TFR1 = TFR[ind,:,:]
    TFR2 = TFR[~ind,:,:]
    TFR_subtraction = np.mean(TFR1, axis = 0) - np.mean(TFR2, axis = 0)

    if plotting is True:
        
        plt.pcolormesh(t, freq,  np.mean(TFR1, axis = 0), cmap='viridis', shading='gouraud')
        plt.title(f'{data_type} subj {subj} {date}')
        plt.colorbar()

        plt.figure()
        plt.pcolormesh(t, freq, np.mean(TFR2, axis = 0), cmap='viridis', shading='gouraud')
        plt.title(f'{data_type} subj {subj} {date}')
        plt.colorbar()

        plt.figure()
        plt.pcolormesh(t, freq, TFR_subtraction, cmap='viridis', shading='gouraud')
        plt.title(f'{data_type} subj {subj} {date} subtraction')
        plt.colorbar()

    if stat == 'mean':
        return np.mean(TFR1, axis = 0), np.mean(TFR2, axis = 0), TFR_subtraction
    else:
        return TFR1, TFR2, TFR_subtraction

def computeNull(TFR1a, TFR1b, TFR2a, TFR2b, permutation_num):
    """Inputs four time-frequency matrices of shape [trial x time x frequency], computes null distribution of double mean subtraction using 
    oermutation testing. TFRs are split into two classes 1/2 and a/b

    Args:
        TFR1a (arr): [trial x time x frequency]  
        TFR1b (arr): [trial x time x frequency]
        TFR2a (arr): [trial x time x frequency]
        TFR2b (arr): [trial x time x frequency]
        permutation_num (int): number of permutations

    Returns:
        null (arr): [perm x time x frequency] matrix of double subtractions
    """
    a = np.concatenate([TFR1a, TFR1b, TFR2a, TFR2b])
    indices = np.arange(0, a.shape[0], 1)
    null = np.zeros([permutation_num, TFR1a.shape[1], TFR1a.shape[2]])

    for perm in range(permutation_num):
        a1, b1, a2, b2 = [a[train_test_split(indices, test_size = .25)[1], :, :] for i in range(4)]
        
        s1 = np.mean(a1, axis = 0) - np.mean(b1, axis = 0)
        s2 = np.mean(a2, axis = 0) - np.mean(b2, axis = 0)
        null[perm, :, :] = s1 - s2
    
    return null

def computePolarityNull(TFR, permutation_num, band):
    """Inputs four time-frequency matrices of shape [trial x time x frequency], computes null distribution of double mean subtraction using 
    oermutation testing. TFRs are split into two classes 1/2 and a/b

    Args:
        TFR1a (arr): [trial x time x frequency]  
        TFR1b (arr): [trial x time x frequency]
        TFR2a (arr): [trial x time x frequency]
        TFR2b (arr): [trial x time x frequency]
        permutation_num (int): number of permutations

    Returns:
        null (arr): [perm x time x frequency] matrix of double subtractions
    """
    indices = np.arange(0, TFR.shape[0], 1)

    if band == 'All':
        null = np.zeros([permutation_num, TFR.shape[1], TFR.shape[2]])

        for perm in range(permutation_num):
            a1, b1, a2, b2, c1, d1, c2, d2 = [TFR[train_test_split(indices, test_size = .25)[1], :, :] for i in range(8)]

            s1 = np.mean(a1, axis = 0) - np.mean(a2, axis = 0)
            s2 = np.mean(b1, axis = 0) - np.mean(b2, axis = 0)
            s3 = np.mean(c1, axis = 0) - np.mean(c2, axis = 0)
            s4 = np.mean(d1, axis = 0) - np.mean(d2, axis = 0)

            null[perm, :, :] = (s1 - s2) - (s3 - s4)

    elif band == 'Alpha':
        null = np.zeros([permutation_num, TFR.shape[2]])

        for perm in range(permutation_num):
            a1, b1, a2, b2, c1, d1, c2, d2 = [TFR[train_test_split(indices, test_size = .25)[1], :, :] for i in range(8)]

            s1 = np.mean(np.mean(a1, axis = 0)[4:8,:], axis = 0)- np.mean(np.mean(a2, axis = 0)[4:8,:], axis = 0)
            s2 = np.mean(np.mean(b1, axis = 0)[4:8,:], axis = 0)- np.mean(np.mean(b2, axis = 0)[4:8,:], axis = 0)
            s3 = np.mean(np.mean(c1, axis = 0)[4:8,:], axis = 0)- np.mean(np.mean(c2, axis = 0)[4:8,:], axis = 0)
            s4 = np.mean(np.mean(d1, axis = 0)[4:8,:], axis = 0)- np.mean(np.mean(d2, axis = 0)[4:8,:], axis = 0)

            null[perm, :] = (s1 - s2) - (s3 - s4)

    elif band == 'Beta':
        null = np.zeros([permutation_num, TFR.shape[2]])

        for perm in range(permutation_num):
            a1, b1, a2, b2, c1, d1, c2, d2 = [TFR[train_test_split(indices, test_size = .25)[1], :, :] for i in range(8)]

            s1 = np.mean(np.mean(a1, axis = 0)[11:24,:], axis = 0)- np.mean(np.mean(a2, axis = 0)[11:24,:], axis = 0)
            s2 = np.mean(np.mean(b1, axis = 0)[11:24,:], axis = 0)- np.mean(np.mean(b2, axis = 0)[11:24,:], axis = 0)
            s3 = np.mean(np.mean(c1, axis = 0)[11:24,:], axis = 0)- np.mean(np.mean(c2, axis = 0)[11:24,:], axis = 0)
            s4 = np.mean(np.mean(d1, axis = 0)[11:24,:], axis = 0)- np.mean(np.mean(d2, axis = 0)[11:24,:], axis = 0)

            null[perm, :] = (s1 - s2) - (s3 - s4)
    return null

def permPValue(true, null):
    """ Calculates number of permutations per time-frequency point that the true double subtraction is as extreme as the null permutation

    Args:
        true (_type_): _description_
        null (_type_): _description_

    Returns:
        _type_: _description_
    """
    sig_low = np.zeros_like(true)
    sig_high = np.zeros_like(true)

    for (x, y), _ in np.ndenumerate(sig_low):
        for j in range(null.shape[0]):
            if true[x,y] < null[j,x,y]:
                sig_low[x,y] += 1

    for (x, y), _ in np.ndenumerate(sig_high):
         for j in range(null.shape[0]):
             if true[x,y] > null[j,x,y]:
                 sig_high[x,y] += 1

                 
    sig = np.fmax(sig_low, sig_high)
    return 1 - (sig / null.shape[0]) 

def sidakCorrection(alpha, m):
    alpha_corrected = 1 - (1 - alpha)**(1/m)
    return alpha_corrected

def plotPermutationAnalysis(averaging_type, subj_keys, data_splits, polarities, directory):
    """Plots significant voxels of EEG TFR permutation analysis according to averaging_type

    Args:
        averaging_type (str): if 'Group", plots group-averaged data, elif 'Individual', plots individual data averaged across 
        sessions, elif 'Session' plots individual sessions per participant

    """
    freqs = np.linspace(4, 30, 25)
    t = np.linspace(0, .75, 384)
    p = sidakCorrection(alpha = .05, m = freqs.shape[0] * t.shape[0])
    baseline = int(512 * .05) 
    folder = fr'{directory}\Data\EEG Permutations'

    if averaging_type == 'Group':
        for _, data_type in enumerate(data_splits): 
            fig = plt.figure(figsize=(10,10))
            for polnum, polarity in enumerate(polarities):

                subplot = 221 + polnum
                fig.add_subplot(subplot)
                true = np.load(fr'{folder}\{polarity}_group average_{data_type}_true.npy')
                sig = np.load(fr'{folder}\{polarity}_group average_{data_type}_sig.npy')
                plt.pcolormesh(t, freqs, true, vmin = -2, vmax = 2, cmap = 'viridis')
                plt.title(f'group average {data_type} {polarity}')
                plt.xlabel('Time from Trial Onset (s)')
                plt.ylabel('Frequency')
                cbar = plt.colorbar()
                cbar.set_label('Delta Power')


            subplot = 223
            fig.add_subplot(subplot)
            true_subtract = np.load(fr'{folder}\polarity_subtraction_group_average_{data_type}_true.npy')
            sig_subtract = np.load(fr'{folder}\polarity_subtraction_group_average_{data_type}_sig.npy')
            plt.pcolormesh(t, freqs, true_subtract, vmin = -2, vmax = 2, cmap = 'gray')

            sig_subtract[sig_subtract>.05] = np.nan
            true_subtract[:,0:baseline] = np.nan
            true_subtract[np.isnan(sig_subtract)] = np.nan
            plt.pcolormesh(t, freqs, true_subtract, vmin = -2, vmax = 2)
            plt.title(f'{data_type} polarity subtraction group average')
            plt.xlabel('Time from Trial Onset (s)')
            plt.ylabel('Frequency')
            cbar = plt.colorbar()
            cbar.set_label('Delta Power')

            subplot = 224
            fig.add_subplot(subplot)
            true_subtract = np.load(fr'{folder}\polarity_subtraction_group_average_{data_type}_true.npy')
            sig_subtract = np.load(fr'{folder}\polarity_subtraction_group_average_{data_type}_sig.npy')
            plt.pcolormesh(t, freqs, true_subtract, vmin = -2, vmax = 2, cmap = 'gray')
            plt.title(f'{data_type} polarity subtraction group average')
            plt.xlabel('Time from Trial Onset (s)')
            plt.ylabel('Frequency')
            cbar = plt.colorbar()
            cbar.set_label('Delta Power')

            sig_subtract[sig_subtract>p] = np.nan
            true_subtract[:,0:baseline] = np.nan
            true_subtract[np.isnan(sig_subtract)] = np.nan
            plt.pcolormesh(t, freqs, true_subtract, vmin = -2, vmax = 2)
            plt.title(f'{data_type} corrected polarity subtraction group average')
            plt.title(f'{data_type} polarity subtraction group average')
            plt.xlabel('Time from Trial Onset (s)')
            plt.ylabel('Frequency')
            fig.tight_layout()


    elif averaging_type == 'Individual':
        for _, data_type in enumerate(data_splits): 
            for subjnum, subj in enumerate(subj_keys): 
                fig = plt.figure(figsize=(10,10))
                fig.tight_layout()
                for polnum, polarity in enumerate(polarities):
                    true = np.load(fr'{folder}\{subj}_{polarity}_session average_{data_type}_true.npy')
                    sig = np.load(fr'{folder}\{subj}_{polarity}_session average_{data_type}_sig.npy')
                    subplot = 221 + polnum

                    fig.add_subplot(subplot)
                    plt.pcolormesh(t, freqs, true, vmin = -2, vmax = 2, cmap = 'gray')
                    plt.title(f'subj_{subjnum} {data_type} {polarity}')
                    plt.xlabel('Time from Trial Onset (s)')
                    plt.ylabel('Frequency')
                    cbar = plt.colorbar()
                    cbar.set_label('Delta Power')

                    plt.pcolormesh(t, freqs, true, vmin = -2, vmax = 2, cmap = 'viridis')
                    plt.title(f'subj_{subjnum} {data_type} {polarity} corrected sig voxels')
                    plt.xlabel('Time from Trial Onset (s)')
                    plt.ylabel('Frequency')


                true_subtract = np.load(fr'{folder}\{subj}_{polarity}_polarity_subtraction_session_average_{data_type}_true.npy')
                sig_subtract = np.load(fr'{folder}\{subj}_{polarity}_polarity_subtraction_session_average_{data_type}_sig.npy')
                
                subplot = 223
                fig.add_subplot(subplot)
                plt.pcolormesh(t, freqs, true_subtract, vmin = -2, vmax = 2, cmap = 'gray')
                plt.xlabel('Time from Trial Onset (s)')
                plt.ylabel('Frequency')
                cbar = plt.colorbar()
                cbar.set_label('Delta Power')

                sig_subtract[sig_subtract>.05] = np.nan
                true_subtract[:,0:baseline] = np.nan
                true_subtract[np.isnan(sig_subtract)] = np.nan

                plt.pcolormesh(t, freqs, true_subtract, vmin = -2, vmax = 2, cmap = 'viridis')
                plt.title(f'subj_{subjnum} {data_type} polarity subtraction')
                plt.xlabel('Time from Trial Onset (s)')
                plt.ylabel('Frequency')

                subplot = 224
                fig.add_subplot(subplot)

                true_subtract = np.load(fr'{folder}\{subj}_{polarity}_polarity_subtraction_session_average_{data_type}_true.npy')
                plt.pcolormesh(t, freqs, true_subtract, vmin = -2, vmax = 2, cmap = 'gray')

                sig_subtract[sig_subtract>p] = np.nan
                true_subtract[:,0:baseline] = np.nan
                true_subtract[np.isnan(sig_subtract)] = np.nan
                plt.pcolormesh(t, freqs, true_subtract, vmin = -2, vmax = 2, cmap = 'viridis')
                plt.title(f'corrected subj_{subjnum} {data_type} polarity subtraction')
                plt.xlabel('Time from Trial Onset (s)')
                plt.ylabel('Frequency')
                cbar = plt.colorbar()
                cbar.set_label('Delta Power')

                    
    elif averaging_type == 'Session':
        with open(fr'{directory}\Data\pre_sessions.pkl', 'rb') as f:
            pre_sessions = pickle.load(f)
        for _, data_type in enumerate(data_splits): 
            for subjnum, subj in enumerate(subj_keys): 
                for polarity in polarities:
                    fig = plt.figure(figsize=(15, 15))
                    for fignum, date in enumerate(pre_sessions[subj][polarity].keys()):
                        true = np.load(fr'{folder}\{subj}_{polarity}_{date}_{data_type}_true.npy')
                        sig = np.load(fr'{folder}\{subj}_{polarity}_{date}_{data_type}_sig.npy')
                        
                        subplot = 331 + fignum
                        fig.add_subplot(subplot)
                        plt.pcolormesh(t, freqs, true, cmap = 'gray', vmin = -2, vmax = 2)
                        plt.colorbar()
                        plt.title(f'subj_{subjnum} {data_type} {polarity} {date}')

                        
                        true[:,0:baseline] = np.nan
                        true[np.isnan(sig)] = np.nan

                        plt.pcolormesh(t, freqs, true, cmap = 'viridis', vmin = -2, vmax = 2)
                        plt.colorbar()
                        plt.title(f'subj_{subjnum} {data_type} {polarity} {date} significant')
                        plt.colorbar()
                        plt.xlabel('Time from Trial Onset (s)')
                        plt.ylabel('Frequency')
                        cbar = plt.colorbar()
                        cbar.set_label('Delta Power')


def plotBandTimeCourse(averaging_type, subj_keys, data_splits, directory):
    t = np.linspace(0, .75, 384)
    folder = fr'{directory}\Data\EEG Permutations'
    alpha = [5, 6]
    beta = [15, 16]
    p = sidakCorrection(alpha = .05, m = t.shape[0])

    if averaging_type == 'Group':
        for _, data_type in enumerate(data_splits): 
            ### Plot alpha band
            true = np.load(fr'{folder}\polarity_subtraction_group_average_{data_type}_true.npy')
            sig = np.load(fr'{folder}\polarity_subtraction_group_average_{data_type}_sig.npy')
            null = np.load(fr'{folder}\polarity_subtraction_group_average_{data_type}_null.npy')

            fig = plt.figure(figsize=(10,10))
            fig.tight_layout()
            subplot = 221 
            fig.add_subplot(subplot)
            
            plt.plot(t, np.mean(true[alpha[0]:alpha[1], :], axis = 0), 'k-')
            plt.fill_between(t, np.mean(true[alpha[0]:alpha[1], :], axis = 0) - np.std(null[alpha[0]:alpha[1], :]),
                            np.mean(true[alpha[0]:alpha[1], :], axis = 0) + np.std(null[alpha[0]:alpha[1], :]), color = 'grey')

            sig[sig>p] = np.nan
            true[np.isnan(sig)] = np.nan
            plt.fill_between(t, np.mean(true[alpha[0]:alpha[1], :], axis = 0) - np.std(null[alpha[0]:alpha[1], :]),
                            np.mean(true[alpha[0]:alpha[1], :], axis = 0) + np.std(null[alpha[0]:alpha[1], :]), color = 'blue')
            plt.title(f'{data_type} Alpha band modulation')
            plt.xlabel('Time from Trial Onset (s)')
            plt.ylabel('Delta Power')

            ### Plot beta band
            true = np.load(fr'{folder}\polarity_subtraction_group_average_{data_type}_true.npy')
            sig = np.load(fr'{folder}\polarity_subtraction_group_average_{data_type}_sig.npy')
            null = np.load(fr'{folder}\polarity_subtraction_group_average_{data_type}_null.npy')


            subplot = 222
            fig.add_subplot(subplot)
            
            plt.plot(t, np.mean(true[beta[0]:beta[1], :], axis = 0), 'k-')
            plt.fill_between(t, np.mean(true[beta[0]:beta[1], :], axis = 0) - np.std(null[beta[0]:beta[1], :]),
                            np.mean(true[beta[0]:beta[1], :], axis = 0) + np.std(null[beta[0]:beta[1], :]), color = 'grey')

            sig[sig>p] = np.nan
            true[np.isnan(sig)] = np.nan
            plt.fill_between(t, np.mean(true[beta[0]:beta[1], :], axis = 0) - np.std(null[beta[0]:beta[1], :]),
                            np.mean(true[beta[0]:beta[1], :], axis = 0) + np.std(null[beta[0]:beta[1], :]), color = 'blue')
            plt.title(f'Group {data_type} beta band modulation')
            plt.xlabel('Time from Trial Onset (s)')
            plt.ylabel('Delta Power')



    if averaging_type == 'Individual':
        for _, data_type in enumerate(data_splits): 
            for _, subj in enumerate(subj_keys): 
                true = np.load(fr'{folder}\{subj}_CA_polarity_subtraction_session_average_{data_type}_true.npy')
                sig = np.load(fr'{folder}\{subj}_CA_polarity_subtraction_session_average_{data_type}_sig.npy')
                null = np.load(fr'{folder}\{subj}_CA_polarity_subtraction_session_average_{data_type}_null.npy')

                fig = plt.figure(figsize=(10,10))
                fig.tight_layout()
                subplot = 221 
                fig.add_subplot(subplot)
                
                plt.plot(t, np.mean(true[alpha[0]:alpha[1], :], axis = 0), 'k-')
                plt.fill_between(t, np.mean(true[alpha[0]:alpha[1], :], axis = 0) - np.std(null[alpha[0]:alpha[1], :]),
                                np.mean(true[alpha[0]:alpha[1], :], axis = 0) + np.std(null[alpha[0]:alpha[1], :]), color = 'grey')

                sig[sig>p] = np.nan
                true[np.isnan(sig)] = np.nan
                plt.fill_between(t, np.mean(true[alpha[0]:alpha[1], :], axis = 0) - np.std(null[alpha[0]:alpha[1], :]),
                                np.mean(true[alpha[0]:alpha[1], :], axis = 0) + np.std(null[alpha[0]:alpha[1], :]), color = 'blue')
                plt.title(f'{subj} {data_type} alpha band modulation')
                plt.xlabel('Time from Trial Onset (s)')
                plt.ylabel('Delta Power')


                ### Plot beta band
                true = np.load(fr'{folder}\{subj}_CA_polarity_subtraction_session_average_{data_type}_true.npy')
                sig = np.load(fr'{folder}\{subj}_CA_polarity_subtraction_session_average_{data_type}_sig.npy')
                null = np.load(fr'{folder}\{subj}_CA_polarity_subtraction_session_average_{data_type}_null.npy')


                subplot = 222
                fig.add_subplot(subplot)
                
                plt.plot(t, np.mean(true[beta[0]:beta[1], :], axis = 0), 'k-')
                plt.fill_between(t, np.mean(true[beta[0]:beta[1], :], axis = 0) - np.std(null[beta[0]:beta[1], :]),
                                np.mean(true[beta[0]:beta[1], :], axis = 0) + np.std(null[beta[0]:beta[1], :]), color = 'grey')

                sig[sig>p] = np.nan
                true[np.isnan(sig)] = np.nan
                plt.fill_between(t, np.mean(true[beta[0]:beta[1], :], axis = 0) - np.std(null[beta[0]:beta[1], :]),
                                np.mean(true[beta[0]:beta[1], :], axis = 0) + np.std(null[beta[0]:beta[1], :]), color = 'blue')
                plt.title(f'{subj} {data_type} beta band modulation')
                plt.xlabel('Time from Trial Onset (s)')
                plt.ylabel('Delta Power')





