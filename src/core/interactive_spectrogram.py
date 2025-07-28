#!/usr/bin/env python3
"""
Interactive Spectrogram Visualization

Simple function to create interactive spectrograms with hover values using Plotly.
Just hover your mouse over the spectrogram to see exact time, frequency, and power values.
"""

import numpy as np
import plotly.graph_objects as go
import os


def create_interactive_spectrogram(frequencies, times, spectrogram_data, 
                                 title="Interactive Spectrogram", 
                                 baseline_window=None,
                                 save_path=None):
    """
    Create an interactive spectrogram with hover values.
    
    Parameters:
    -----------
    frequencies : np.ndarray
        Frequency values for y-axis
    times : np.ndarray
        Time values for x-axis  
    spectrogram_data : np.ndarray
        2D spectrogram data (n_freqs, n_times)
    title : str
        Plot title
    baseline_window : tuple, optional
        (start_time, end_time) for baseline period highlighting
    save_path : str, optional
        Path to save HTML file
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Interactive Plotly figure
    """
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        x=times,
        y=frequencies,
        z=spectrogram_data,
        colorscale='Viridis',  # You can change this to match your Parula colormap
        hovertemplate='<b>Time:</b> %{x:.3f}s<br>' +
                     '<b>Frequency:</b> %{y:.1f}Hz<br>' +
                     '<b>Z-score:</b> %{z:.3f}<extra></extra>',
        colorbar=dict(title="Z-score")
    ))
    
    # Add event marker at t=0
    fig.add_vline(x=0, line_dash="dash", line_color="white", line_width=2,
                  annotation_text="Event", annotation_position="top")
    
    # Add baseline window if provided
    if baseline_window is not None:
        start_time, end_time = baseline_window
        fig.add_vrect(x0=start_time, x1=end_time, 
                     fillcolor="red", opacity=0.2,
                     line_color="red", line_width=1,
                     annotation_text="Baseline", annotation_position="top left")
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        yaxis_type="log",  # Log scale for frequencies
        width=900,
        height=500,
        font=dict(size=12)
    )
    
    # Save HTML file if path provided
    if save_path:
        html_file = os.path.join(save_path, f'interactive_{title.lower().replace(" ", "_")}.html')
        fig.write_html(html_file)
        print(f"✓ Interactive spectrogram saved to: {html_file}")
        print(f"  Open in browser to hover and see values!")
    
    return fig


def add_interactive_to_results(results, save_path, nm_sizes=None):
    """
    Create interactive spectrograms for all NM sizes in results.
    
    Parameters:
    -----------
    results : dict
        Analysis results dictionary
    save_path : str
        Directory to save HTML files
    nm_sizes : list, optional
        Specific NM sizes to plot (default: all)
    """
    
    if nm_sizes is None:
        nm_sizes = list(results['averaged_windows'].keys())
    
    frequencies = results['frequencies']
    baseline_window = results['analysis_parameters'].get('baseline_window', None)
    
    interactive_figs = {}
    
    for nm_size in nm_sizes:
        window_data = results['averaged_windows'][nm_size]
        avg_spectrogram = window_data['avg_spectrogram']
        window_times = window_data['window_times']
        
        n_rats = results['n_rats']
        total_events = window_data['total_events_all_rats']
        
        title = f'NM Size {nm_size} - {n_rats} Rats ({total_events} events)'
        
        fig = create_interactive_spectrogram(
            frequencies=frequencies,
            times=window_times, 
            spectrogram_data=avg_spectrogram,
            title=title,
            baseline_window=baseline_window,
            save_path=save_path
        )
        
        interactive_figs[nm_size] = fig
    
    return interactive_figs


# Example usage
if __name__ == "__main__":
    """
    Example of how to use with your analysis results
    """
    
    # After running your analysis (e.g., nm_theta_cross_rats.py):
    # results = run_cross_rats_analysis(roi="11", ...)
    
    # Create interactive spectrograms:
    # interactive_figs = add_interactive_to_results(results, "path/to/save")
    
    # Show in browser:
    # for nm_size, fig in interactive_figs.items():
    #     fig.show()
    
    print("To use interactive spectrograms:")
    print("1. Import: from interactive_spectrogram import add_interactive_to_results")
    print("2. After your analysis: interactive_figs = add_interactive_to_results(results, save_path)")
    print("3. View: fig.show() opens in browser with hover values!")