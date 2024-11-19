#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

from helper_code import *
import numpy as np, os, sys
import mne
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
from sklearn import preprocessing
import pandas as pd
import math
import antropy as ant
from scipy.signal import hilbert


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your model.

def train_challenge_model(data_folder, model_folder):
        # read files
        patient_ids = find_data_folders(data_folder)
        num_patients = len(patient_ids)
        if num_patients==0:
                raise FileNotFoundError('No data was provided.')

        # Create a folder for the model if it does not already exist.
        os.makedirs(model_folder, exist_ok=True)

        features = list()
        outcomes = list()
        cpcs = list()

        for patient_id in patient_ids:
                # Extract labels.
                patient_metadata = load_challenge_data(data_folder, patient_id)
                current_outcome = get_outcome(patient_metadata)
                current_cpc = get_cpc(patient_metadata)

                # Extract features (metadata and eeg)
                current_patient_features = get_features(data_folder, patient_id)
                
                if len(current_patient_features) == 0:
                    continue

                current_cpc = get_cpc(patient_metadata)
                current_outcome = get_outcome(patient_metadata)
                
                # Extract labels.
                for feature_per_rec in current_patient_features:
                    features.append(feature_per_rec)
                    outcomes.append(current_outcome)
                    cpcs.append(current_cpc)
                
                # for f_rec in current_patient_features:
                #         features.append(f_rec)
                #         outcomes.append(current_outcome)
                #         cpcs.append(current_cpc)



        features = np.vstack(features)
        outcomes = np.vstack(outcomes)
        cpcs = np.vstack(cpcs)

        print('Training the Challenge models on the Challenge data...')

        # Define parameters for random forest classifier and regressor.
        n_estimators   = 400  # Number of trees in the forest.
        # max_leaf_nodes = 456  # Maximum number of leaf nodes in each tree.
        random_state   = 42  # Random state; set for reproducibility.
        max_depth = None

        # Impute any missing features; use the mean value by default.
        imputer = SimpleImputer().fit(features)

        # Train the models.
        features = imputer.transform(features)
        outcome_model = RandomForestClassifier(
                criterion='entropy', n_estimators=n_estimators, max_depth = max_depth,
                max_features='sqrt', random_state=random_state).fit(features, outcomes.ravel())
        cpc_model = RandomForestRegressor(
                criterion='mse', n_estimators=n_estimators, max_depth = max_depth,
                max_features='sqrt', random_state=random_state).fit(features, cpcs.ravel())
        # Save the models.
        save_challenge_model(model_folder, imputer, outcome_model, cpc_model)
        
        print('Done.')

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def load_challenge_models(model_folder, verbose):
    filename = os.path.join(model_folder, 'models.sav')
    return joblib.load(filename)

# Run your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_challenge_models(models, data_folder, patient_id, verbose):
    imputer = models['imputer']
    outcome_model = models['outcome_model']
    cpc_model = models['cpc_model']

    # Extract features.
    features = get_features(data_folder, patient_id)
    features = features.reshape(1, -1)

    # Impute missing data.
    features = imputer.transform(features)

    # Apply models to features.
    outcome = outcome_model.predict(features)[0]
    outcome_probability = outcome_model.predict_proba(features)[0, 1]
    cpc = cpc_model.predict(features)[0]

    # Ensure that the CPC score is between (or equal to) 1 and 5.
    cpc = np.clip(cpc, 1, 5)

    return outcome, outcome_probability, cpc

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Save your trained model.
def save_challenge_model(model_folder, imputer, outcome_model, cpc_model):
    d = {'imputer': imputer, 'outcome_model': outcome_model, 'cpc_model': cpc_model}
    filename = os.path.join(model_folder, 'models.sav')
    joblib.dump(d, filename, protocol=0)

# Preprocess data.
def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.1, 45.0]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    if utility_frequency is not None and passband[0] <= utility_frequency <= passband[1]:
        data = mne.filter.notch_filter(data, sampling_frequency, utility_frequency, n_jobs=4, verbose='error')

    # Apply a bandpass filter.
    data = mne.filter.filter_data(data, sampling_frequency, passband[0], passband[1], n_jobs=4, verbose='error')

    # Resample the data.
    resampling_frequency = 128 if sampling_frequency % 2 == 0 else 125
    data = mne.filter.resample(data, down=int(sampling_frequency / resampling_frequency), npad='auto')

    # Scale the data (z-score normalization).
    # Convert to float32 after filtering to save memory
    data = data.astype(np.float32)
    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(data)
    
    return data, resampling_frequency

# Extract features.
def get_features(main_root, patient_id):
    # Load patient data.
    patient_metadata = load_challenge_data(main_root, patient_id)
    recording_ids = find_records_names_for_each_patient(main_root, patient_id)
    num_recordings = len(recording_ids)

    # Extract EEG features.
    eeg_channels = ['C3','P3','Cz','C4','T6',
                    'T3','T5','O1','Pz','O2','P4',
                    'F8','T4','F7','Fp1','F3','Fz','Fp2','F4']
    
    bipolar_channels = ['F3-C3','C3-P3','F4-C4','C4-P4','Fz-Cz','P3-O1','T5-O1','P4-O2','T6-O2',
                        'Fp1-Fp2','T3-T4','T5-T6','Fp1-T3','Fp2-T4','T4-O1','T3-O2','Fp1-Cz','Fp2-Cz',
                        'O1-O2','P3-P4','Cz-Pz']
    # group = 'EEG'
    min_length = 256
    list_features = []
    # stacked_eeg_features = list()
    if num_recordings > 0:
        records_name = find_records_names_for_each_patient(main_root, patient_id)
        record_files_path = [os.path.join(main_root, patient_id, record_name) for record_name in records_name]
        for recording_location in record_files_path:

            if os.path.exists(recording_location):
                data, channels, sampling_frequency = load_recording_data(recording_location)
                utility_frequency = get_utility_frequency(recording_location)

                if all(channel in channels for channel in eeg_channels) and all(len(channel) >= min_length for channel in data):
                    data, channels = reduce_channels(data, channels, eeg_channels) # It also fixes the order of channels
                    data = convert_to_bipolar(data, channels, bipolar_channels) # Convert to bipolar montage
                    data, sampling_frequency = preprocess_data(data, sampling_frequency, utility_frequency)
                    features = get_rec_features(patient_metadata, data).flatten()
                    list_features.append(features)
                    # stacked_eeg_features.append(eeg_features)
                # else:
                #     # eeg_features = float('nan') * np.ones(84) # 2 bipolar channels * 4 features / channel
                #     # stacked_eeg_features.append(eeg_features)
            # else:
            #     # eeg_features = float('nan') * np.ones(84) # 2 bipolar channels * 4 features / channel
            #     # stacked_eeg_features.append(eeg_features)

    # else:
    #     eeg_features = float('nan') * np.ones(84) # 2 bipolar channels * 4 features / channel
    #     stacked_eeg_features.append(eeg_features)

    # # all_stacked_features = list()
    # for eeg_f in stacked_eeg_features:
    #     each_recording_features = np.hstack((eeg_f))
    #     all_stacked_features.append(each_recording_features)

    # Extract features.
    return list_features

# # Extract patient features from the data.
# def get_patient_features(data):
#     age = get_age(data)
#     # sex = get_sex(data)
#     rosc = get_rosc(data)
#     ohca = get_ohca(data)
#     shockable_rhythm = get_shockable_rhythm(data)
#     # ttm = get_ttm(data)

#     # sex_features = np.zeros(2, dtype=int)
#     # if sex == 'Female':
#     #     female = 1
#     #     male   = 0
#     #     other  = 0
#     # elif sex == 'Male':
#     #     female = 0
#     #     male   = 1
#     #     other  = 0
#     # else:
#     #     female = 0
#     #     male   = 0
#     #     other  = 1

#     features = np.array((age, rosc, ohca, shockable_rhythm))

#     return features

# Extract features from the EEG data.
def get_rec_features(patient_metadata, data):
    # Extract patient features from metadata.
    # patient_features = get_patient_features(patient_metadata)
    age = get_age(patient_metadata)
    sex = get_sex(patient_metadata)
    rosc = get_rosc(patient_metadata)
    ohca = get_ohca(patient_metadata)
    # vfib = get_vfib(patient_metadata)
    ttm = get_ttm(patient_metadata)

    # Use one-hot encoding for sex; add more variables
    if sex == 'Female':
        female = 1
        male   = 0
        other  = 0
    elif sex == 'Male':
        female = 0
        male   = 1
        other  = 0
    else:
        female = 0
        male   = 0
        other  = 1
    
    # Combine the patient features.
    patient_features = np.array([age, female, male, other, rosc, ohca, ttm])

    temp1=[]
    f=list(fun(data[0],'0').values())+list(fun(data[0],'1').values())+list(fun(data[0],'2').values())+list(fun(data[0],'3').values())+list(fun(data[0],'4').values())+list(fun(data[0],'5').values())+list(fun(data[0],'6').values())+list(fun(data[0],'7').values())+list(fun(data[0],'8').values())+list(fun(data[0],'9').values())+list(fun(data[0],'10').values())+list(fun(data[0],'11').values())+list(fun(data[0],'12').values())+list(fun(data[0],'13').values())+list(fun(data[0],'14').values())+list(fun(data[0],'15').values())+list(fun(data[0],'16').values())+list(fun(data[0],'17').values())
    temp1.append(f)
    tt=pd.DataFrame(temp1).describe().loc[['mean','std']].T
    recording_features=np.hstack((tt['mean'],tt['std']))

    # Combine the features from the patient metadata and the recording data and metadata.
    features = np.hstack((patient_features, recording_features))

    return features

# Extract features from the ECG data.
def get_ecg_features(data):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        mean = np.mean(data, axis=1)
        std  = np.std(data, axis=1)
    elif num_samples == 1:
        mean = np.mean(data, axis=1)
        std  = float('nan') * np.ones(num_channels)
    else:
        mean = float('nan') * np.ones(num_channels)
        std = float('nan') * np.ones(num_channels)

    features = np.array((mean, std)).T

    return features

################################################################################
#
######################### ADDED FUNCTIONS ######################################
#
################################################################################

# Find the record names.
def find_records_names_for_each_patient(main_root, patient_id):
    records_names = []
    patient_folder = os.path.join(main_root, patient_id)
    for file_name in sorted(os.listdir(patient_folder)):
        if not file_name.startswith('.') and file_name.endswith('.hea'):
            records_names.append(file_name)
    return records_names

# Convert data to Bipolar mode
def convert_to_bipolar(data, channels, bipolar_channels):
    bipolar_data = []

    # Find the indices of the channels in the original data
    for i, ch_to_ch in enumerate(bipolar_channels):
        # Find the indices of the channels in the original data
        first_channel, second_channel = ch_to_ch.split("-")
        ch_index_1 = channels.index(first_channel)
        ch_index_2 = channels.index(second_channel)

        # Calculate the bipolar signal by subtracting one channel from the other
        bipolar_signal = data[ch_index_1, :] - data[ch_index_2, :]

        # Append the bipolar signal to the list
        bipolar_data.append(bipolar_signal)

    # Convert the list of bipolar signals to a 2D array
    bipolar_data = np.array(bipolar_data)

    return bipolar_data

#-----------------------------------------------------------------------------------------
##########################################################################################

class all_features:

    def __init__(self,array):
        self.array = array
        self.sum1 = 0
        self.expc = 0
        self.max_mod = 0
        self.differential1 = []
        self.differential2 = [] 
        self.F1 = 0
        self.F2 = 0
        self.F3 = 0
        self.F4 = 0
        self.F6 = 0
        self.F7 = 0
        self.F8 = 0
        self.F9 = 0
        self.F10 = 0
        self.F11 = 0
        self.F12 = 0
        self.F13 = 0
        self.F14 = 0
        self.F15 = 0
        self.F16 = 0
        self.F17 = 0
        self.F18 = 0
        self.F19 = 0
        self.F20 = 0
        self.F21 = []
        self.F24 = []
        self.F25 = []
        self.F26 = []
        self.F27 = []
        self.F28 = []
        self.F29 = []
    def variance(self,a):
        l=[]
        for i in range(len(a)):
            s=0
            for j in range(len(a[i])):
                s+=math.sqrt(abs(a[i][j]))
            l.append((s/len(a[i]))**2)
        return l
        
    def sum_array(self):
        self.sum1 = sum(list(self.array))
        return 

    def expectation(self):
        s=0
        for j in range(len(self.array)):
            s+=self.array[j]*(j+1)
        self.expc = s
        return     
    def maximum_mod_value(self):
        self.max_mod = abs(max(self.array,key=abs))
        return 
    
    def first_order_differential(self):
        for i in range(len(self.array)):
            s=[]
            for j in range(len(self.array[i])-1):
                s.append(self.array[i][j+1]-self.array[i][j])
            self.differential1.append(s)
        return
    def second_order_differential(self):
        for i in range(len(self.array)):
            s=[]
            for j in range(len(self.differential1[i])-1):
                s.append(self.differential1[i][j+1]-self.differential1[i][j])
            self.differential2.append(s)
        return  
        
    def maximum_value(self):
        self.F1 = max(self.array)
        return self.F1

    def max_index(self):
        self.F2=list(self.array).index(max(list(self.array)))+1
        return self.F2

    def equivalent_width(self):
        if self.F1 == 0:
            return None
        self.F3 = self.expc / self.F1
        return self.F3

    def centroid(self):
        if self.sum1 == 0:
            return None
        self.F4 = self.expc / self.sum1
        return self.F4
        
    def root_mean_square_width(self):
        if self.sum1 == 0:
            return None
        s = sum(self.array[j] * (j+1) ** 2 for j in range(len(self.array)))
        self.F6 = math.sqrt(abs(s / self.sum1))
        return self.F6
        
    def mean_value(self):
        self.F7 = np.mean(self.array)
        return self.F7
    
    def standard_deviation(self):
        s = 0
        for j in range(len(self.array)):
            s+= (self.array[j] - self.F7)**2
        self.F8 = math.sqrt(s/len(self.array))
        return self.F8
    
    def skewness(self):
        if self.F8 == 0:
            return None
        s = sum((self.array[j] - self.F7) ** 3 for j in range(len(self.array)))
        self.F9 = s / ((self.F8 ** 3) * len(self.array))
        return self.F9
                           
    def kurtosis(self):
        if self.F8 == 0:
            return None
        s = sum((self.array[j] - self.F7) ** 4 for j in range(len(self.array)))
        self.F10 = s / ((self.F8 ** 4) * len(self.array))
        return self.F10
            
    def median(self):
        self.F11 = np.median(self.array)
        return self.F11
    
    def rms_value(self):
        s=0
        for j in range(len(self.array)):
            s+=self.array[j]**2
        self.F12 = math.sqrt(s/len(self.array))
        return self.F12
    
    def sqrt_amp(self):
        s=0
        for j in range(len(self.array)):
            s+=math.sqrt(abs(self.array[j]))
        self.F13 = (s/len(self.array))**2
        return self.F13
    
    def peaktopeak(self):
        self.F14 = self.F1 - min(self.array)
        return self.F14
    
    def var(self):
        self.F15 = self.F8**2
        return self.F15
        
    def crest_factor(self):
        if self.F12 == 0:
            return None
        self.F16 = self.max_mod / self.F12
        return self.F16

    
    def shape_factor(self):
        s=0
        for j in range(len(self.array)):
            s+=abs(self.array[j])
        self.F17 = (len(self.array)*self.F12)/s
        return self.F17
        
    def impulse_factor(self):
        s=0
        for j in range(len(self.array)):
            s+=abs(self.array[j])
        self.F18 = (len(self.array)*self.max_mod)/s
        return self.F18
        
    def margin_factor(self):
        if self.F15 == 0:
            return None
        self.F19 = self.max_mod / self.F15
        return self.F19
    
    def form_factor(self):
        if self.F12 == 0 or self.F7 == 0:
            return None
        self.F20 = self.F12 / self.F7
        return self.F20
    
    def clearance_factor(self):
        if self.F13 == 0:
            return None
        self.F21 = self.max_mod / self.F13
        return self.F21
    
    def peak_index(self):
        if self.F8 == 0:
            return None
        self.F24 = self.max_mod / self.F8
        return self.F24
    
    def skewness_index(self):
        if self.F15 == 0:
            return None
        self.F25 = self.F9 / (math.sqrt(self.F15) ** 3)
        return self.F25
    
    def first_quartile(self):
        self.F26 = np.quantile(self.array, .25)
        return self.F26
    
    def third_quartile(self):
        self.F27 = np.quantile(self.array, .75) 
        return self.F27
    
    def waveform_length(self):
        s=0
        for j in range(len(self.array)-1):
            s+=abs(self.array[j+1]-self.array[j])
        self.F28 = s
        return self.F28
    
    def wilson_amplitude(self):
        s=0
        for j in range(len(self.array)-1):
            x=abs(self.array[j+1]-self.array[j])
            if x>=0.5:
                s+=1
            else:
                s+=0
        self.F29 = s
        return self.F29

def get_hurst_exponent(time_series, max_lag=20):
    """Returns the Hurst Exponent of the time series"""
    
    lags = range(2, max_lag)

    # variances of the lagged differences
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]

    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]

def fun(data,k):
    feature=all_features(data)
    feature.sum_array()
    feature.expectation()
    F1 = feature.maximum_value()
    F2 = feature.max_index()
    F3 = feature.equivalent_width()
    F4 = feature.centroid()
    F6 = feature.root_mean_square_width()
    F7 = feature.mean_value()
    F8 = feature.standard_deviation()
    F9 = feature.skewness()
    F10 = feature.kurtosis()
    F11 = feature.median()
    F12 = feature.rms_value()
    F13 = feature.sqrt_amp()
    F14 = feature.peaktopeak()
    F15 = feature.var()
    F16 = feature.crest_factor()             ##here max is 1 for all signals
    # feature.maximum_mod_value()  
    F17 = feature.shape_factor()
    F18 = feature.impulse_factor()
    F19 = feature.margin_factor()
    F20 = feature.form_factor()
    F21 = feature.clearance_factor()
    F24 = feature.peak_index()
    F25 = feature.skewness_index()
    F26 = feature.first_quartile()
    F27 = feature.third_quartile()
    F28 = feature.waveform_length()
    F29 = feature.wilson_amplitude()
    sp_entropy=ant.spectral_entropy(data, sf=128, method='welch', normalize=True)
    permutation_entropy=ant.perm_entropy(data, normalize=True)
    entropy_svd=ant.svd_entropy(data, normalize=True)
    # Hjorth mobility and complexity
    hjorth_mob=ant.hjorth_params(data)[0]
    hjorth_comp=ant.hjorth_params(data)[1]
    zcr=ant.num_zerocross(data)
    delta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=128,  fmin=0.5,  fmax=8.0, verbose=False)
    theta_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=128,  fmin=4.0,  fmax=8.0, verbose=False)
    alpha_psd, _ = mne.time_frequency.psd_array_welch(data, sfreq=128,  fmin=8.0, fmax=12.0, verbose=False)
    beta_psd,  _ = mne.time_frequency.psd_array_welch(data, sfreq=128, fmin=12.0, fmax=30.0, verbose=False)
    delta_psd_mean = np.nanmean(delta_psd)
    theta_psd_mean = np.nanmean(theta_psd)
    alpha_psd_mean = np.nanmean(alpha_psd)
    beta_psd_mean  = np.nanmean(beta_psd)
    delta_psd_std = np.nanstd(delta_psd)
    theta_psd_std = np.nanstd(theta_psd)
    alpha_psd_std = np.nanstd(alpha_psd)
    beta_psd_std  = np.nanstd(beta_psd)
    delta_psd_max = np.nanmax(delta_psd)
    theta_psd_max = np.nanmax(theta_psd)
    alpha_psd_max = np.nanmax(alpha_psd)
    beta_psd_max  = np.nanmax(beta_psd)
    delta_psd_freq = np.nanargmax(delta_psd)
    theta_psd_freq = np.nanargmax(theta_psd)
    alpha_psd_freq = np.nanargmax(alpha_psd)
    beta_psd_freq  = np.nanargmax(beta_psd)
    IF_max= max((np.diff(np.unwrap(np.angle(hilbert(data))))/(2.0*np.pi) * 100))
    IF_min= min((np.diff(np.unwrap(np.angle(hilbert(data))))/(2.0*np.pi) * 100))
    hurst_exp = get_hurst_exponent(data)
    power=(1/len(data))*(np.sum(np.abs(data)**2))
    # Petrosian fractal dimension
    pfd=ant.petrosian_fd(data)
    # Katz fractal dimension
    kfd=ant.katz_fd(data)
    # Higuchi fractal dimension
    hfd=ant.higuchi_fd(data)
    # Detrended fluctuation analysis
    dfd=ant.detrended_fluctuation(data)
    TD_features = {'Max_val'+k:F1,'Max_index'+k:F2,'Equi_width'+k:F3,'Centroid'+k:F4,'RMS_width'+k:F6,'Mean'+k:F7,'SD'+k:F8,'Skewness'+k:F9,'Kurtosis'+k:F10,'Median'+k:F11,'SQRT'+k:F13,'p-p'+k:F14,'Crest_factor'+k:F16,'Shape_factor'+k:F17,'Impulse_factor'+k:F18,'Margin_factor'+k:F19,'Form_factor'+k:F20,'Clearance_factor'+k:F21,'Peak_index'+k:F24,'Skewness_index'+k:F25,'1st-Q'+k:F26,'3rd-Q'+k:F27,'Waveform_length'+k:F28,'Wilson_amp'+k:F29,'sp_entropy'+k:sp_entropy,'permutation_entropy'+k:permutation_entropy,'entropy_svd'+k:entropy_svd,'hjorth_mob'+k:hjorth_mob,'hjorth_comp'+k:hjorth_comp,'zcr'+k:zcr,'delta_psd_mean'+k:delta_psd_mean,'theta_psd_mean'+k:theta_psd_mean,'alpha_psd_mean'+k:alpha_psd_mean,'beta_psd_mean'+k:beta_psd_mean,'delta_psd_max'+k:delta_psd_max,'theta_psd_max'+k:theta_psd_max,'alpha_psd_max'+k:alpha_psd_max,'beta_psd_max'+k:beta_psd_max,'delta_psd_freq'+k:delta_psd_freq,'theta_psd_freq'+k:theta_psd_freq,'beta_psd_freq'+k:beta_psd_freq,'alpha_psd_freq'+k:alpha_psd_freq,'delta_psd_std'+k:delta_psd_std,'theta_psd_std'+k:theta_psd_std,'beta_psd_std'+k:beta_psd_std,'alpha_psd_std'+k:alpha_psd_std,'IF_max'+k:IF_max,'IF_min'+k:IF_min,'power'+k:power,'Hurst_exp'+k:hurst_exp,'pfd'+k:pfd,'kfd'+k:kfd,'hfd'+k:hfd,'dfd'+k:dfd}
    return TD_features
