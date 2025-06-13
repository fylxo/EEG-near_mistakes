import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
import mne

def create_summary_dataframe(all_data, verbose=True):
    """
    Create a session-level summary DataFrame and print key stats.
    Returns the summary DataFrame and grouped stats.
    """
    # Build session-level DataFrame
    df_summary = pd.DataFrame([{
        'rat_id': d['rat_id'],
        'session_date': d['session_date'],
        'eeg_len': d['eeg'].shape[1],
        'nm_count': len(d['nm_sizes']),
        'iti_count': len(d['iti_sizes']),
        'file': d['file_path']
    } for d in all_data])

    # Group by rat
    grouped = df_summary.groupby('rat_id').agg({
        'eeg_len': 'mean',
        'nm_count': 'sum',
        'iti_count': 'sum',
        'session_date': 'count'
    }).rename(columns={'session_date': 'session_count'})

    # Longest and shortest sessions
    longest = df_summary.loc[df_summary['eeg_len'].idxmax()]
    shortest = df_summary.loc[df_summary['eeg_len'].idxmin()]

    if verbose:
        print("First 5 session summaries:")
        print(df_summary.head())

        print("\nEEG length stats:")
        print(df_summary['eeg_len'].describe())

        print("\nNM/ITI sizes per session:")
        print(df_summary[['nm_count', 'iti_count']].describe())

        print("\nPer-rat summary:")
        print(grouped)

        print("\nLongest EEG session:")
        print(longest[['rat_id', 'session_date', 'eeg_len']])

        print("\nShortest EEG session:")
        print(shortest[['rat_id', 'session_date', 'eeg_len']])

    return df_summary, grouped, longest, shortest

def convert_to_raw(data, sfreq=200):
    eeg = data['eeg']  # shape: (32, n_samples)
    n_channels, n_times = eeg.shape

    # Create channel names and types
    ch_names = [f"Ch{i}" for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Convert to µV (MNE expects Volts, so we divide by 1e6)
    raw = mne.io.RawArray(eeg / 1e6, info)
    return raw

def plot_overlayed_psd(data, sfreq=200, picks='all', fmin=1, fmax=90, average=False, show=True, **kwargs):
    """
    Plot overlayed PSDs for all or a subset of channels using MNE-Python.
    """
    raw = convert_to_raw(data)
    psd = raw.compute_psd(fmin=fmin, fmax=fmax, picks=picks)
    fig = psd.plot(average=average, picks='all', show=show, **kwargs)
    return fig