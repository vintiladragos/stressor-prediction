# function that does all data preprocessing of the above cells
import wfdb
import numpy as np
import pandas as pd


def preprocess_data(record_1_name, record_2_name):
    """
    :param record_1_name: SpO2HR
    :param record_2_name: AccTempEDA
    :return:
    phase_data structure:
    list(tuple(str, pd.DataFrame))
    tuple: (phase_name, phase_data)
    phase_name: label of phase (interval)
    dataframe columns: hr, EDA, acceleration
    """
    phase_names = ["relaxation", "physical", "relaxation", "emotional", "cognitive", "recovery", "emotional(horror)", "relaxation"]
    annotation = wfdb.rdann(record_2_name, 'atr')
    phases = annotation.sample / annotation.fs
    # format time of annotations to be in seconds
    phases = phases.astype(int)
    # relaxation = until phases[1]
    # physical = phases[1] to phases[2]
    # relaxation = phases[2] to phases[3]
    # emotional = phases[3] to phases[4]
    # cognitive =  phases[4] to phases[5]
    # recovery = phases[5] to phases[6]
    # emotional(horror) = phases[6] to phases[7]
    # relaxation = phases[7] to end
    record_1 = wfdb.rdrecord(record_1_name)
    record_2 = wfdb.rdrecord(record_2_name)
    # we want to downsample the data of record_2 to match the data of record_1. record_1 is 1Hz, record_2 is 8Hz
    record_2_downsampled = record_2.p_signal[::annotation.fs]
    record_1_name_to_index = {name: index for index, name in enumerate(record_1.sig_name)}
    record_2_name_to_index = {name: index for index, name in enumerate(record_2.sig_name)}
    number_of_columns = len(record_1.sig_name) + len(record_2.sig_name)
    # remove the last few data points of record_1 so that it matches the length of record_2
    record_1.p_signal = record_1.p_signal[:record_2_downsampled.shape[0], :]
    # stack the data of record_1 and record_2
    stacked_data = np.hstack((record_1.p_signal, record_2_downsampled))
    # add column names to stacked data
    stacked_data = pd.DataFrame(stacked_data, columns=record_1.sig_name + record_2.sig_name)
    # export to csv
    stacked_data.to_csv('outputs/stacked_data.csv', index=False)
    # get data
    stacked_preprocessed_data = pd.read_csv('outputs/stacked_data.csv')
    # merge ax, ay, az into one column based on acceleration
    stacked_preprocessed_data['acceleration'] = np.sqrt(stacked_preprocessed_data['ax']**2 + stacked_preprocessed_data['ay']**2 + stacked_preprocessed_data['az']**2)
    stacked_preprocessed_data = stacked_preprocessed_data.drop(columns=['ax', 'ay', 'az'])
    # remove temperature data because in my humble opinion, after looking at this dataset,
    # the variability of temperature is unreliable
    stacked_preprocessed_data = stacked_preprocessed_data.drop(columns=['temp'])
    # More expert information is needed about SpO2's importance, for now we will remove it
    stacked_preprocessed_data = stacked_preprocessed_data.drop(columns=['SpO2'])
    # define baseline data
    baseline_data = stacked_preprocessed_data[:phases[1]]
    # make data represent the variation from baseline data
    stacked_preprocessed_data = stacked_preprocessed_data - baseline_data.median()
    # change EDA values to instead be the difference from the previous value
    stacked_preprocessed_data['EDA'] = stacked_preprocessed_data['EDA'].diff(periods=10)
    # refill the first 10 values with 0
    stacked_preprocessed_data['EDA'].iloc[:10] = 0
    # remove negative values from EDA differences, as we only care about positive changes in EDA at this moment
    stacked_preprocessed_data['EDA'] = stacked_preprocessed_data['EDA'].clip(lower=0)
    # change acceleration values to instead be the difference from the previous value
    stacked_preprocessed_data['acceleration'] = stacked_preprocessed_data['acceleration'].diff()
    # fill the first value with 0
    stacked_preprocessed_data['acceleration'].iloc[0] = 0
    # since we care about when acceleration happens, not when it stops, we will remove negative values
    stacked_preprocessed_data['acceleration'] = stacked_preprocessed_data['acceleration'].clip(lower=0)
    # re-define baseline data
    baseline_data = stacked_preprocessed_data[:phases[1]]

    # split data into phases
    phase_data = []
    for index, phase in enumerate(phases):
        if index == len(phases) - 1:
            phase_data.append((str(phase_names[index]), stacked_preprocessed_data[phase:]))
        else:
            phase_data.append((str(phase_names[index]), stacked_preprocessed_data[phase:phases[index + 1]]))

    # unions of baseline and each phase
    # unions = []
    # for index, (name, phase) in enumerate(phase_data):
    #     union_df = pd.concat([baseline_data, phase])
    #     unions.append((name, union_df))

    return phase_data
