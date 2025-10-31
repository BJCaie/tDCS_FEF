import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pickle 
from itertools import product
import statsmodels.api as sm

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
    folder = fr'{directory}\TFR_Subtractions'

    if averaging_type == 'Group':
        for _, data_type in enumerate(data_splits): 
            fig = plt.figure(figsize=(10,10))
            fig.tight_layout()
            for polnum, polarity in enumerate(polarities):

                subplot = 221 + polnum
                fig.add_subplot(subplot)

                true = np.load(fr'{folder}\{polarity}_group average_{data_type}_true.npy')
                sig = np.load(fr'{folder}\{polarity}_group average_{data_type}_sig.npy')

                plt.pcolormesh(t, freqs, true, vmin = -2, vmax = 2, cmap = 'viridis')
                plt.title(f'group average {data_type} {polarity}')
                plt.colorbar()

                sig[sig>p] = np.nan
                true[:,0:baseline] = np.nan
                true[np.isnan(sig)] = np.nan

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
            plt.colorbar()

            subplot = 224
            fig.add_subplot(subplot)

            true_subtract = np.load(fr'{folder}\polarity_subtraction_group_average_{data_type}_true.npy')
            sig_subtract = np.load(fr'{folder}\polarity_subtraction_group_average_{data_type}_sig.npy')
            plt.pcolormesh(t, freqs, true_subtract, vmin = -2, vmax = 2, cmap = 'gray')

            sig_subtract[sig_subtract>p] = np.nan
            true_subtract[:,0:baseline] = np.nan
            true_subtract[np.isnan(sig_subtract)] = np.nan
            plt.pcolormesh(t, freqs, true_subtract, vmin = -2, vmax = 2)
            plt.title(f'{data_type} corrected polarity subtraction group average')
            plt.colorbar()


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

                    sig[sig>p] = np.nan
                    true[:,0:baseline] = np.nan
                    true[np.isnan(sig)] = np.nan

                    plt.pcolormesh(t, freqs, true, vmin = -2, vmax = 2, cmap = 'viridis')
                    plt.title(f'subj_{subjnum} {data_type} {polarity} corrected sig voxels')
                    plt.colorbar()

                true_subtract = np.load(fr'{folder}\{subj}_{polarity}_polarity_subtraction_session_average_{data_type}_true.npy')
                sig_subtract = np.load(fr'{folder}\{subj}_{polarity}_polarity_subtraction_session_average_{data_type}_sig.npy')
                
                subplot = 223
                fig.add_subplot(subplot)
                plt.pcolormesh(t, freqs, true_subtract, vmin = -2, vmax = 2, cmap = 'gray')

                sig_subtract[sig_subtract>.05] = np.nan
                true_subtract[:,0:baseline] = np.nan
                true_subtract[np.isnan(sig_subtract)] = np.nan

                plt.pcolormesh(t, freqs, true_subtract, vmin = -2, vmax = 2, cmap = 'viridis')
                plt.title(f'subj_{subjnum} {data_type} polarity subtraction')
                plt.colorbar()
                
                subplot = 224
                fig.add_subplot(subplot)

                true_subtract = np.load(fr'{folder}\{subj}_{polarity}_polarity_subtraction_session_average_{data_type}_true.npy')
                plt.pcolormesh(t, freqs, true_subtract, vmin = -2, vmax = 2, cmap = 'gray')
                sig_subtract[sig_subtract>p] = np.nan
                true_subtract[:,0:baseline] = np.nan
                true_subtract[np.isnan(sig_subtract)] = np.nan

                plt.pcolormesh(t, freqs, true_subtract, vmin = -2, vmax = 2, cmap = 'viridis')
                plt.title(f'corrected subj_{subjnum} {data_type} polarity subtraction')
                plt.colorbar()
                    
    elif averaging_type == 'Session':
        with open(fr'{directory}\Combined_Data\pre_sessions.pkl', 'rb') as f:
            pre_sessions = pickle.load(f)
        for _, data_type in enumerate(data_splits): 
            for subjnum, subj in enumerate(subj_keys): 
                for polarity in polarities:
                    fig = plt.figure(figsize=(15, 15))
                    for fignum, date in enumerate(pre_sessions[subj][polarity].keys()):
                        true = np.load(fr'{directory}\TFR_Subtractions\{subj}_{polarity}_{date}_{data_type}_true.npy')
                        sig = np.load(fr'{directory}\TFR_Subtractions\{subj}_{polarity}_{date}_{data_type}_sig.npy')
                        
                        subplot = 331 + fignum
                        fig.add_subplot(subplot)
                        plt.pcolormesh(t, freqs, true, cmap = 'gray', vmin = -2, vmax = 2)
                        plt.colorbar()
                        plt.title(f'subj_{subjnum} {data_type} {polarity} {date}')

                        
                        sig[sig>p] = np.nan
                        true[:,0:baseline] = np.nan
                        true[np.isnan(sig)] = np.nan

                        plt.pcolormesh(t, freqs, true, cmap = 'viridis', vmin = -2, vmax = 2)
                        plt.colorbar()
                        plt.title(f'subj_{subjnum} {data_type} {polarity} {date} significant')
                        
def plotPsycPermutations(true, null, plot_title):
    fig = plt.figure()
    params = ['threshold', 'slope', 'mean rt', 'std rt']
    sig = computeSignificance(true, null)

    for i in range(true.shape[0]):
        fig.add_subplot(221 + i)
        plt.hist(null[:,i], bins = 50)
        plt.vlines(true[i], ymin = 0, ymax = 20, colors = 'k')
        plt.title(params[i])
        plt.xlabel('Difference in Difference (ms)')
        plt.ylabel('Count')

    fig.suptitle(plot_title)
    fig.tight_layout()

def computeSignificance(true, null):
    sig_low = np.zeros_like(true)
    sig_high = np.zeros_like(true)
    sig = np.zeros_like(true)
    for i in range(true.shape[0]):
        for j in range(null.shape[0]):
            if true[i] < null[j,i]:
                sig_low[i] += 1
        for j in range(null.shape[0]):
            if true[i] > null[j,i]:
                sig_high[i] += 1
        sig[i] = (1 - (np.fmax(sig_low[i], sig_high[i]) / null.shape[0])) 
    return sig
    
def plotZscores(true, null, plot_title):
    x = [0,1,2,3]
    labels = ['threshold', 'slope', 'mean RT', 'std RT']
    fig, ax = plt.subplots()
    zscore = zScoreParams(true, null)
    plt.bar(np.arange(4), zscore)
    plt.title(plot_title)
    plt.xlabel('Parameter')
    plt.ylabel('Z Score')
    ax.set_xticks(x, labels)

def zScoreParams(true, null):
    zscore = np.zeros(4)
    for i in range(true.shape[0]):
        zscore[i] = (true[i] - np.mean(null[:,i])) / np.std(null[:,i])
    return zscore

def plotSessionCombinations(path, pre_sessions, subj_keys):
    params = ['Threshold', 'Slope', 'Mean RT', 'Std RT']
    for _, subj in enumerate(subj_keys):
        print(f'+++ {subj} psychometric permutation p values +++')
        anodal_dates = pre_sessions[subj]['AN'].keys()
        cathodal_dates = pre_sessions[subj]['CA'].keys()
        combinations = list(product(anodal_dates, cathodal_dates))
        zscore = np.zeros([4])
        for combnum, _ in enumerate(combinations):
            true = np.load(fr'{path}\Data\Psychometric Permutations\{subj}_session_combination_{combnum}_true.npy')
            null = np.load(fr'{path}\Data\Psychometric Permutations\{subj}_session_combination_{combnum}_null.npy')
            zscore = np.vstack((zscore, zScoreParams(true, null)))
            p = computeSignificance(true, null)
            print(f'combination {combnum}')
            print(p)
        zscore = np.delete(zscore, (0), axis = 0)

        ## Plot distribution
        fig = plt.figure()
        for i in range(4):
            subplot = 221 + i
            fig.add_subplot(subplot)
            plt.bar(np.arange(0, zscore.shape[0]),zscore[:,i])
            plt.ylim(bottom = plt.yticks()[0][0], top=plt.yticks()[0][-1])
            plt.xlabel('Session Combination')
            plt.ylabel('Z-score Difference')
            plt.title(f'{params[i]}')
        plt.suptitle(f'{subj} session combinations')
        plt.tight_layout()

def zScoreParams(true, null):
    zscore = np.zeros(4)
    for i in range(true.shape[0]):
        zscore[i] = (true[i] - np.mean(null[:,i])) / np.std(null[:,i])
    return zscore

def fitLinearModel(x, y):
    X2 = sm.add_constant(x)
    est = sm.OLS(y, X2)
    est2 = est.fit()
    return est2.pvalues[1], est2.rsquared


