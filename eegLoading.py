import numpy as np
import pandas as pd 
import scipy.io as sio
import os
from os.path import isfile, join
from os import listdir
import re
import glob
from collections import defaultdict
from scipy.signal import butter, lfilter
from scipy import signal
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # This is to suppress a warning from a future pandas version. just pip freeze it idc

class singleSubject:
    """ Class creates an object for each subject in the tDCS experiment to facilitate single subject analysis
        the inner functions combines the .csv file from the EEG data with the behavioural data from a matlab structure.
        It also compute the trial onsets and offsets in the EEG file from the trigger line, which allows the EEG data to be aligned with the correct
        trials.
    """
    def __init__(self, path, subjectname):
        self.path = path
        self.subjectname = subjectname
        self.sfreq = 512
        self.nchannels = 5
        self.sessiondates = get_immediate_subdirectories(self.path + '/' + self.subjectname)
        self.lowcut = .1
        self.highcut = 50
        self.notch = 60
        self.order = 4
        self.delaylength = .75
        self.winLength = int(self.sfreq*self.delaylength)
        self.laplacian = True
        self.filter = True
        self.tfr_file_path = r"C:\Users\Brandon\Desktop\PhD\FEF\TFR_Data" 
        self.behaviour_file_path = r"E:\PhD\FEF\Behavioural_Data" 
        self.eeg_file_path = r"E:\PhD\FEF\EEG_Data"

    def combinesubject(self):    
        def load_eeg(csvfilename, DMAT, session):
            eeg = {}
            try:
                EEG = pd.read_csv(csvfilename, 
                                skiprows=0) # Read in EEG data from .csv file
                eeg['eegdata'] = np.array([EEG['Channel 1'], EEG['Channel 2'], EEG['Channel 3'], EEG['Channel 4'], EEG['Channel 5']]) 
                eeg['trigger'] = np.array(EEG['Trigger line'])

            except:
                eeg['eegdata'] , eeg['trigger'] = realignCSV(EEG)
                                
            if any(np.isin([0,3], eeg['trigger'])) == False:
                try:
                    eeg['trigger'] = np.array(EEG['Trigger line.2']) # This is for when trigger line was printed to a second column for some reason
                except:
                    print('error with trigger renaming')

            if any(np.isin([0,3], eeg['trigger'])) == True: # This is to stop saving files where trigger is broken
                triggerIndex = np.array(np.nonzero(np.ediff1d(eeg['trigger'])))
                eeg['trialonsets'] = triggerIndex[np.ix_(*[range(0,i,2) for i in triggerIndex.shape])][0]
                eeg['trialoffsets'] = np.setdiff1d(triggerIndex, eeg['trialonsets'])
                
                ## Filter EEG data 
                filteredEEG, badChannelCount = filter_EEG(eeg['eegdata'], samplingFreq = self.sfreq, filter = self.filter)
                
                ## Compute Laplacian
                if self.laplacian == True:
                    laplacian = compute_laplacian(filteredEEG, badChannelCount)
                else:
                    laplacian = np.nanmean([filteredEEG['Channel 1'], filteredEEG['Channel 2'], filteredEEG['Channel 3'], filteredEEG['Channel 4'], filteredEEG['Channel 5']], axis = 0)

                ## Align data to trial onset and save data
                if eeg['trialonsets'].shape[0] == 90:
                    trial_aligned = alignTrialData(eeg = laplacian, trialOnsets = eeg['trialonsets'], winLength = self.winLength)
                    pd.DataFrame(trial_aligned).to_csv(self.eeg_file_path + '/' + DMAT[0:2] + '_' + DMAT[2:4] + '_' + DMAT[4:7] + '_' + session + '_' + 'trial_aligned') 
                else:
                    print('trial missing')
                    print(DMAT)
                    print(session)
                
            else:
                print('Trigger Error')
        
        def combinesession(self, session):
            """Combines all EEG and behavioural data from one session into a dictionary 'behaviour' and 'eeg'

            Args:
                sessionkey (_type_): string that gives the date of the experiment. Strings are derived from a subject folder
                                     and are in YYYY-MM-DD format
            """
            eeg = {}
            session_dir = self.path + '\\' + self.subjectname + '\\' + session
            dmat_dir = session_dir + '\DMAT'
            dmat_files = sorted_alphanumeric([f for f in 
                        listdir(dmat_dir) if isfile(join(dmat_dir, f))])
            num_pre_files = len(glob.glob(dmat_dir + '\\' "*PR*")) #This gives how many pre-stim files to know which EEG .csv corresponds to post-stim file 1
            eeg_files = sorted_alphanumeric([f for f in 
                        listdir(session_dir) if f.endswith('.csv')]) #Passes EEG File list into a function that reorders file based on timestamp (weird python doesn't do this natively)

            for i in range(len(dmat_files)):
                try:
                    behaviour = makeDataFrame(load_behaviour(dmat_dir + '/' + dmat_files[i]))
                    behavioural_filepath = self.behaviour_file_path + '\\' + dmat_files[i][0:2] + '_' + dmat_files[i][2:4] + '_' + dmat_files[i][4:7] + '_' + session + '.csv'
                    behaviour.to_csv(behavioural_filepath)
                    blocknum = re.findall(r'\d+', dmat_files[i]) # Finds all parts of the string that are numeric to return the blocknumber
                    assert len(blocknum) == 1 # Makes sure there isn't anything weird in the filename
                    blocknum = int(blocknum[0]) # Returns block n   umber as an integer for indexing
                    
                    if 'PR' in dmat_files[i]:
                        load_eeg(csvfilename = session_dir + '/' + eeg_files[blocknum], DMAT = dmat_files[i], session = session) #Load CSV file and compute trigger alignments 
                    elif 'PO' in dmat_files[i]:
                        load_eeg(csvfilename = session_dir + '/' + eeg_files[blocknum + num_pre_files - 1], DMAT = dmat_files[i], session = session) #Load CSV file and compute trigger alignments            
                except:
                    print('check corresponding error')
                
            return eeg
        
        ## Combine all sessions from one participant into dated behavioural and eeg         
        self.behaviour = {}
        for i in range(len(self.sessiondates)):
            self.behaviour[self.sessiondates[i]] = combinesession(self, session = self.sessiondates[i])      

def load_behaviour(filename):
    try:
        blockdata = sio.loadmat(file_name = filename)
        return blockdata['param']
    except:
        print('Behavioural data matrix not found')
        
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
    
def makeDataFrame(data):
    df = pd.DataFrame(
        {
            'Block Number': data[:,1],
            'Trial Number': data[:,2],
            'Good/Bad Trial': data[:,3],
            'Reaction Time: First Target': data[:,4],
            'Choice': data[:,8],
            'TOA': data[:,25],
            'First Target Onset': data[:,28],
            'Second Target Onset': data[:,29],
            'Relative RT': data[:,4] + data[:,28],
            'Saccade Endpoint': data[:,8],
            'Previous RT': np.insert(np.roll(data[:,4],1)[1:], 0, np.nan)
        }
    )

    df['Choice'][df['Choice']> 0] = 1 # Binarize Choice (right)
    df['Choice'][df['Choice']< 0] = 0 # Binarize Choice (left)
    df['TOA'] = df['TOA'].round(3) # Round TOA to discrete bins for averaging
    df['Rep Number'] = np.zeros(len(df))
    df['Alt Number'] = np.zeros(len(df))
    
    for i in range(len(df)):
        if df['Good/Bad Trial'][i] == 0:
            if df['Trial Number'][i] == df['Trial Number'].shift(periods=1)[i] + 1:
                if df['Choice'][i] == df['Choice'].shift(periods=1)[i]:
                    df['Rep Number'][i] = df['Rep Number'][i-1] + 1
            if df['Trial Number'][i] == df['Trial Number'].shift(periods=1)[i] + 1:
                if df['Choice'][i] != df['Choice'].shift(periods=1)[i]:
                    df['Alt Number'][i] = df['Alt Number'][i-1] + 1
    return df

def tryint(s):
    try:
        return int(s)
    except:
        return s
    
def sorted_alphanumeric(data):
    """Sorts file directory by all numbers in file name sequentially
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def notch_filter(data):
    b, a = signal.iirnotch(60, 30, 512)
    y = lfilter(b, a, data)
    return y

    
def filter_EEG(eeg, samplingFreq, filter):
    """
    Filters 4x1 HD-tDCS EEG center-surround electrode setup by calling filterEEG for each channel of data, and then performs a surface laplacian on the center electrode
    https://www.sciencedirect.com/science/article/pii/S0167876015001749
    Surface Laplacian is normalized sum of surround electrodes + center electrodes (since surround electrodes are equidistant)

    inputs:
    eeg: [channel x timestep]
    sampling_freq: 512 Hz in this experiment
    lowcut: lowest passing frequency for filtering
    highcut: highest passing frequency for filtering
    notch: utility frequency/power line noise cut (60Hz in Canada)
    order: order of the filters used (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html)

    outputs:
    laplacian: [1 x timestep] vector of surface laplacian for center electrode ('Channel 1')
    """
    channels = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5']
    filteredEEG = {}
    badChannelCount = 0

    for i in range(len(channels)):
        if filter == True:
            #notched_data = notch_filter(eeg[i,:])
            filteredEEG[channels[i]]  = butter_lowpass_filter(eeg[i,:], cutoff = 30, fs = samplingFreq, order=3)
            #filteredEEG[channels[i]] = filterSingleChannel(eeg[i,:], samplingFreq = samplingFreq, lowcut= lowcut, highcut = highcut, notch = notch, order = order) #Calls filterEEG from loading
        else:
            filteredEEG[channels[i]] = eeg[i,:]
        try:
            if np.isnan(filteredEEG[channels[i]][0]) == 1:
                badChannelCount = badChannelCount + 1
        except:
                badChannelCount = badChannelCount + 1
            
    return filteredEEG, badChannelCount

def compute_laplacian(filteredEEG, badChannelCount):
    """Compute surface laplacian for 4x1 EEG-tDCS setup

    Args:
        filteredEEG (dict): filteredEEG['Channel X'] corresponds to channel number
        badChannelCount (int): number of channels to discard for normalizing constant

    Returns:
        laplacian: np.array of surface laplacian computation
    """
    channels = ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5']
    normalizingConstant = 1 / (len(channels)- 1 - badChannelCount)
    for channel in channels:
        if filteredEEG[channel] is None:
            filteredEEG[channel] = np.nan
    laplacian = filteredEEG['Channel 1'] - .25 * np.nansum([filteredEEG['Channel 2'] + filteredEEG['Channel 3'] + filteredEEG['Channel 4'] + filteredEEG['Channel 5']], axis = 0)
    return laplacian

def alignTrialData(eeg, trialOnsets, winLength):
    """
    Transforms single column vector EEG data to matrix where alignedData = [trials x timesteps].
    alignedData[i, 0]: onset for trial i
    winLength: max number of timesteps following trialOnset (750 is first possible target, so should not be greater than .75*fs for trial aligned)
    """
    trialOnsets = np.array(trialOnsets)
    trialOnsets = trialOnsets.astype(int)
    winLength = int(winLength)

    alignedData = np.zeros([len(trialOnsets), winLength])
    try:
        for i in range(len(trialOnsets)):
            trialData = eeg[trialOnsets[i]:trialOnsets[i]+winLength]
            alignedData[i,0:winLength] = trialData
    except:
        print('check')
    return alignedData 

def alignTargetData(eeg, trialOnsets, behaviour, winLength, fs):
    """Align EEG data to onset of the first target onset

    Args:
        eeg (_type_): _description_
        trialOnsets (_type_): _description_
        behaviour (_type_): _description_
        winLength (_type_): _description_
        fs (_type_): _description_
    """
    preTime = 500
    postTime = 200

    trialOnsets = np.array(trialOnsets)
    trialOnsets = trialOnsets.astype(int)
    winLength = int(winLength)
    targetOnsets = np.rint(np.array((behaviour['First Target Onset'] / 1000) * fs))
    alignedData = np.zeros([len(trialOnsets), preTime + postTime])

    for i in range(len(trialOnsets)):
        currentOnset = int(trialOnsets[i] + targetOnsets[i])
        trialData = eeg[currentOnset-preTime:currentOnset + postTime]
        alignedData[i,:] = trialData
        
    return alignedData

def combineGroupEEGwithBehaviour(eeg_path, behavioural_path):
    """Function to append EEG data to each row of behavioural data matrix to do trial-wise analyses

    Args:
        eeg_path (str): path where processed eeg data is stored
        behavioural_path (str): path where behavioural data is stored

    Returns:
        combined: dataframe combining behavioural data with EEG surface laplacian along each trial row
    """
    eeg_directory = os.listdir(eeg_path)
    behavioural_directory = os.listdir(behavioural_path)

    combined = 0

    for b_name in behavioural_directory:
        block_string = b_name[0:20]
        behaviour = pd.read_csv(behavioural_path + "/" + b_name, index_col= False)
        for e_name in eeg_directory:
            if block_string in e_name:
                if 'trial_aligned' in e_name:
                    eeg = pd.read_csv(eeg_path + "/" + e_name, index_col= False)
                    if isinstance(combined, pd.DataFrame):
                        combined = pd.concat([combined, behaviour.merge(eeg)])
                    else:
                        combined = behaviour.merge(eeg)
    return combined

def individualEEGwithBehaviour(eeg_path, behavioural_path, participant):
    """Function to append subject-specific EEG data to each row of behavioural data matrix to do trial-wise analyses

    Args:
        eeg_path (str): path where processed eeg data is stored
        behavioural_path (str): path where behavioural data is stored
        participant (str): string matching participant ID

    Returns:
        combined (df): dataframe with EEG-behaviour single trials for chosen participant
    """
    eeg_directory = os.listdir(eeg_path)
    behavioural_directory = os.listdir(behavioural_path)

    combined = 0

    for b_name in behavioural_directory:
        if participant in b_name:
            block_string = b_name[0:20]
            behaviour = pd.read_csv(behavioural_path + "/" + b_name, index_col= False)
            for e_name in eeg_directory:
                if block_string in e_name:
                    if 'trial_aligned' in e_name:
                        eeg = pd.read_csv(eeg_path + "/" + e_name, index_col= False)
                        if isinstance(combined, pd.DataFrame):
                            combined = pd.concat([combined, behaviour.merge(eeg)])
                        else:
                            combined = behaviour.merge(eeg)
    return combined

def combineSubjects(path):
    """
    Args: 
        path: path directory where all data is stored.
    
    Outputs:
        all_subjects: dictionary with an entry for each subject. all_subjects['subject_name'] returns one instantiation of the singleSubject class 
                      for the corresponding subject initial code
        [pre/stim/post]_[anodal/cathodal]__eeg: dictionary containing combined EEG data for each participant, grouped according to the variable name (pre/stim/post x anodal/cathodal)

    """
    all_subjects = {} # Dictionary where each subject has a single entry
    eeg = defaultdict(dict)
    behaviour = {}

    for file in os.listdir(path):
        subjectdirectory = os.path.join(path, file)
        if os.path.isdir(subjectdirectory):
            subjectname = os.path.basename(os.path.normpath(subjectdirectory))
            all_subjects[subjectname] = singleSubject(path = path, subjectname = subjectname)
            all_subjects[subjectname].combinesubject()
            
    return all_subjects , behaviour


def realignCSV(EEG):
    timepoints = EEG[EEG.keys()[0]].shape[0]
    numChannels = 5
    channels = np.empty([timepoints, numChannels])
    trigger = np.empty([timepoints])

    for t in range(timepoints): # Iterate over each EEG timepoint

        s = re.split(r'-', EEG[EEG.keys()[0]][t])
        
        ## Fix last entry
        for c in range(6):
            s[c] = '-' + s[c][0:10]   

        channels[t, :] = s[1:6]

        ## Get trigger value
        if re.split(r'\D', s[-1])[-2][-1] == '3':
            trigger[t] = 3.0
        else:
            trigger[t] = 0
            
    return channels.T, trigger
