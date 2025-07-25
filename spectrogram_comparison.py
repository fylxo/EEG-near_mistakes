import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import spectrogram
import warnings
warnings.filterwarnings('ignore')


try:
    import mplcursors
    MPLCURSORS_AVAILABLE = True
except ImportError:
    MPLCURSORS_AVAILABLE = False
    print("mplcursors not available. Using built-in matplotlib hover functionality.")

def generate_synthetic_eeg(duration=60, fs=1000, n_events=20):
    """Generate synthetic EEG data with events"""
    t = np.arange(0, duration, 1/fs)
    
    # Base EEG signal with multiple frequency components
    eeg = (np.random.randn(len(t)) * 0.5 + 
           np.sin(2 * np.pi * 10 * t) * 0.3 +  # Alpha
           np.sin(2 * np.pi * 20 * t) * 0.2 +  # Beta
           np.sin(2 * np.pi * 4 * t) * 0.4)    # Theta
    
    # Add event-related activity (brief gamma bursts)
    event_times = np.random.uniform(5, duration-5, n_events)
    event_samples = (event_times * fs).astype(int)
    
    for event_sample in event_samples:
        # Add gamma burst around event
        event_window = slice(max(0, event_sample-int(0.2*fs)), 
                           min(len(eeg), event_sample+int(0.2*fs)))
        gamma_burst = np.sin(2 * np.pi * 40 * t[event_window]) * 0.8
        eeg[event_window] += gamma_burst
    
    return eeg, t, event_samples

def approach1_global_spectrogram(eeg, fs, event_samples):
    """Approach 1: Compute spectrogram over entire signal, then window and average"""
    # Compute spectrogram over entire signal
    f, t_spec, Sxx = spectrogram(eeg, fs, nperseg=int(0.5*fs), noverlap=int(0.4*fs))
    
    # Use raw power values (no dB conversion)
    Sxx_power = Sxx
    
    # Window around events and average
    window_size = 2.0  # -1 to +1 seconds
    window_samples = int(window_size * len(t_spec) / (len(eeg) / fs))
    
    event_spectrograms = []
    
    for event_sample in event_samples:
        # Convert event sample to spectrogram time index
        event_time = event_sample / fs
        event_idx = int(event_time * len(t_spec) / (len(eeg) / fs))
        
        # Extract window around event
        start_idx = max(0, event_idx - window_samples//2)
        end_idx = min(len(t_spec), event_idx + window_samples//2)
        
        if end_idx - start_idx >= window_samples//2:  # Ensure minimum window size
            event_spec = Sxx_power[:, start_idx:end_idx]
            event_spectrograms.append(event_spec)
    
    if event_spectrograms:
        # Average all event spectrograms
        avg_spectrogram = np.mean(event_spectrograms, axis=0)
        # Create time vector for the windowed spectrogram
        time_vector = np.linspace(-1, 1, avg_spectrogram.shape[1])
        return f, time_vector, avg_spectrogram
    else:
        return f, np.array([]), np.array([])

def approach2_windowed_spectrogram(eeg, fs, event_samples):
    """Approach 2: Window time domain first, compute spectrograms, clip and average"""
    window_size_samples = int(4 * fs)  # -2 to +2 seconds
    clip_size_samples = int(2 * fs)    # -1 to +1 seconds
    
    event_spectrograms = []
    
    for event_sample in event_samples:
        # Extract time domain window (-2 to +2 seconds)
        start_idx = max(0, event_sample - window_size_samples//2)
        end_idx = min(len(eeg), event_sample + window_size_samples//2)
        
        if end_idx - start_idx >= window_size_samples//2:  # Ensure minimum window size
            windowed_eeg = eeg[start_idx:end_idx]
            
            # Compute spectrogram for this window
            f, t_spec, Sxx = spectrogram(windowed_eeg, fs, nperseg=int(0.5*fs), noverlap=int(0.4*fs))
            
            # Use raw power values (no dB conversion)
            Sxx_power = Sxx
            
            # Clip to -1 to +1 seconds (center portion)
            total_time = len(windowed_eeg) / fs
            center_idx = len(t_spec) // 2
            clip_samples = int(clip_size_samples * len(t_spec) / len(windowed_eeg))
            
            start_clip = max(0, center_idx - clip_samples//2)
            end_clip = min(len(t_spec), center_idx + clip_samples//2)
            
            clipped_spec = Sxx_power[:, start_clip:end_clip]
            event_spectrograms.append(clipped_spec)
    
    if event_spectrograms:
        # Average all event spectrograms
        avg_spectrogram = np.mean(event_spectrograms, axis=0)
        # Create time vector for the clipped spectrogram
        time_vector = np.linspace(-1, 1, avg_spectrogram.shape[1])
        return f, time_vector, avg_spectrogram
    else:
        return f, np.array([]), np.array([])

def add_hover_functionality(ax, t, f, data, unit=""):
    """Add hover functionality to a plot using matplotlib's built-in events"""
    def on_hover(event):
        if event.inaxes == ax:
            # Get the data coordinates
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                # Find the closest data point
                t_idx = np.argmin(np.abs(t - x))
                f_idx = np.argmin(np.abs(f - y))
                
                if t_idx < len(t) and f_idx < len(f) and t_idx < data.shape[1] and f_idx < data.shape[0]:
                    value = data[f_idx, t_idx]
                    ax.set_title(f"{ax.get_title().split('|')[0]} | Time: {x:.2f}s, Freq: {y:.1f}Hz, Value: {value:.2e}{unit}")
    
    # Store original title
    original_title = ax.get_title()
    
    def on_leave(event):
        if event.inaxes == ax:
            ax.set_title(original_title)
    
    ax.figure.canvas.mpl_connect('motion_notify_event', on_hover)
    ax.figure.canvas.mpl_connect('axes_leave_event', on_leave)

def plot_comparison(f1, t1, spec1, f2, t2, spec2):
    """Plot comparison of the two approaches with interactive hover - Power and dB plots"""
    
    # Create power plots
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig1.suptitle('Power Spectrograms', fontsize=16)
    
    # Plot approach 1 - Power
    if spec1.size > 0:
        im1 = ax1.pcolormesh(t1, f1, spec1, shading='gouraud', cmap='viridis')
        ax1.set_title('Approach 1: Global Spectrogram\n(Compute full signal, then window)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Frequency (Hz)')
        ax1.set_ylim(0, 50)
        plt.colorbar(im1, ax=ax1, label='Power')
        if MPLCURSORS_AVAILABLE:
            cursor1 = mplcursors.cursor(im1, hover=True)
            cursor1.connect("add", lambda sel: sel.annotation.set_text(
                f"Time: {sel.target[0]:.2f}s\nFreq: {sel.target[1]:.1f}Hz\nPower: {sel.target[2]:.2e}"))
    
    # Plot approach 2 - Power
    if spec2.size > 0:
        im2 = ax2.pcolormesh(t2, f2, spec2, shading='gouraud', cmap='viridis')
        ax2.set_title('Approach 2: Windowed Spectrogram\n(Window first, then compute)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_ylim(0, 50)
        plt.colorbar(im2, ax=ax2, label='Power')
        if MPLCURSORS_AVAILABLE:
            cursor2 = mplcursors.cursor(im2, hover=True)
            cursor2.connect("add", lambda sel: sel.annotation.set_text(
                f"Time: {sel.target[0]:.2f}s\nFreq: {sel.target[1]:.1f}Hz\nPower: {sel.target[2]:.2e}"))
    
    # Plot difference - Power
    if spec1.size > 0 and spec2.size > 0:
        # Ensure same dimensions for subtraction
        min_time = min(spec1.shape[1], spec2.shape[1])
        min_freq = min(spec1.shape[0], spec2.shape[0])
        
        diff = spec1[:min_freq, :min_time] - spec2[:min_freq, :min_time]
        im3 = ax3.pcolormesh(t1[:min_time], f1[:min_freq], diff, shading='gouraud', cmap='RdBu_r')
        ax3.set_title('Difference\n(Approach 1 - Approach 2)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Frequency (Hz)')
        ax3.set_ylim(0, 50)
        plt.colorbar(im3, ax=ax3, label='Power Difference')
        if MPLCURSORS_AVAILABLE:
            cursor3 = mplcursors.cursor(im3, hover=True)
            cursor3.connect("add", lambda sel: sel.annotation.set_text(
                f"Time: {sel.target[0]:.2f}s\nFreq: {sel.target[1]:.1f}Hz\nDiff: {sel.target[2]:.2e}"))
    
    plt.tight_layout()
    plt.show()
    
    # Create decibel plots
    fig2, (ax4, ax5, ax6) = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle('Decibel Spectrograms', fontsize=16)
    
    # Plot approach 1 - dB
    if spec1.size > 0:
        spec1_db = 10 * np.log10(spec1 + 1e-12)  # Add small epsilon to avoid log(0)
        im4 = ax4.pcolormesh(t1, f1, spec1_db, shading='gouraud', cmap='viridis')
        ax4.set_title('Approach 1: Global Spectrogram\n(Compute full signal, then window)')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Frequency (Hz)')
        ax4.set_ylim(0, 50)
        plt.colorbar(im4, ax=ax4, label='Power (dB)')
        if MPLCURSORS_AVAILABLE:
            cursor4 = mplcursors.cursor(im4, hover=True)
            cursor4.connect("add", lambda sel: sel.annotation.set_text(
                f"Time: {sel.target[0]:.2f}s\nFreq: {sel.target[1]:.1f}Hz\nPower: {sel.target[2]:.1f} dB"))
    
    # Plot approach 2 - dB
    if spec2.size > 0:
        spec2_db = 10 * np.log10(spec2 + 1e-12)  # Add small epsilon to avoid log(0)
        im5 = ax5.pcolormesh(t2, f2, spec2_db, shading='gouraud', cmap='viridis')
        ax5.set_title('Approach 2: Windowed Spectrogram\n(Window first, then compute)')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Frequency (Hz)')
        ax5.set_ylim(0, 50)
        plt.colorbar(im5, ax=ax5, label='Power (dB)')
        if MPLCURSORS_AVAILABLE:
            cursor5 = mplcursors.cursor(im5, hover=True)
            cursor5.connect("add", lambda sel: sel.annotation.set_text(
                f"Time: {sel.target[0]:.2f}s\nFreq: {sel.target[1]:.1f}Hz\nPower: {sel.target[2]:.1f} dB"))
    
    # Plot difference - dB
    if spec1.size > 0 and spec2.size > 0:
        diff_db = spec1_db[:min_freq, :min_time] - spec2_db[:min_freq, :min_time]
        im6 = ax6.pcolormesh(t1[:min_time], f1[:min_freq], diff_db, shading='gouraud', cmap='RdBu_r')
        ax6.set_title('Difference\n(Approach 1 - Approach 2)')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Frequency (Hz)')
        ax6.set_ylim(0, 50)
        plt.colorbar(im6, ax=ax6, label='Power Difference (dB)')
        if MPLCURSORS_AVAILABLE:
            cursor6 = mplcursors.cursor(im6, hover=True)
            cursor6.connect("add", lambda sel: sel.annotation.set_text(
                f"Time: {sel.target[0]:.2f}s\nFreq: {sel.target[1]:.1f}Hz\nDiff: {sel.target[2]:.1f} dB"))
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Generate synthetic EEG data
    print("Generating synthetic EEG data...")
    eeg, t, event_samples = generate_synthetic_eeg(duration=60, fs=1000, n_events=20)
    fs = 1000
    
    print(f"Generated {len(eeg)} samples over {len(t)/fs:.1f} seconds")
    print(f"Number of events: {len(event_samples)}")
    
    # Approach 1: Global spectrogram
    print("\nApproach 1: Computing global spectrogram...")
    f1, t1, spec1 = approach1_global_spectrogram(eeg, fs, event_samples)
    
    # Approach 2: Windowed spectrogram
    print("Approach 2: Computing windowed spectrograms...")
    f2, t2, spec2 = approach2_windowed_spectrogram(eeg, fs, event_samples)
    
    # Plot results
    print("\nPlotting comparison...")
    plot_comparison(f1, t1, spec1, f2, t2, spec2)
    
    # Print summary statistics
    if spec1.size > 0 and spec2.size > 0:
        print(f"\nSummary Statistics:")
        print(f"Approach 1 - Mean power: {np.mean(spec1):.2e}, Std: {np.std(spec1):.2e}")
        print(f"Approach 2 - Mean power: {np.mean(spec2):.2e}, Std: {np.std(spec2):.2e}")
        
        # Calculate difference metrics
        min_time = min(spec1.shape[1], spec2.shape[1])
        min_freq = min(spec1.shape[0], spec2.shape[0])
        diff = spec1[:min_freq, :min_time] - spec2[:min_freq, :min_time]
        print(f"Difference - Mean: {np.mean(diff):.2e}, Std: {np.std(diff):.2e}")
        print(f"Max absolute difference: {np.max(np.abs(diff)):.2e}")