#!/usr/bin/env python3
"""
Example: How to add interactive spectrograms to your existing analysis scripts

Add these lines to your main analysis files to get interactive hover spectrograms.
"""

# Add this import at the top of your analysis files
from interactive_spectrogram import add_interactive_to_results

# Example modification for nm_theta_cross_rats.py (or any of your 4 main files)
def example_integration():
    """
    Show how to integrate interactive spectrograms into existing analysis
    """
    
    # Your existing analysis code...
    # results = run_cross_rats_analysis(roi="11", ...)
    
    # ADD THESE LINES after your analysis completes:
    
    print("Creating interactive spectrograms...")
    interactive_figs = add_interactive_to_results(
        results=results,           # Your analysis results
        save_path=save_path       # Same path where static plots are saved
    )
    
    # Optional: Show immediately in browser
    print("Opening interactive spectrograms in browser...")
    for nm_size, fig in interactive_figs.items():
        fig.show()  # Opens each spectrogram in browser tab
    
    print("✓ Interactive spectrograms created!")
    print("  - HTML files saved alongside your static plots")
    print("  - Hover mouse over spectrograms to see exact values")
    

# For your 4 main files, you would add:

"""
=============== FOR nm_theta_cross_rats.py ===============
At the end of run_cross_rats_analysis(), add:

    # Create interactive spectrograms  
    from interactive_spectrogram import add_interactive_to_results
    print("Creating interactive spectrograms...")
    interactive_figs = add_interactive_to_results(results, save_path)
    
    return aggregated_results, interactive_figs  # Optional: return both

=============== FOR global_normalization.py ===============
Same integration pattern

=============== FOR pre_event_normalization.py =============== 
Same integration pattern

=============== FOR nm_theta_cross_rats_global.py ===============
Same integration pattern
"""

if __name__ == "__main__":
    print("This is an example file showing how to integrate interactive spectrograms.")
    print("Add the import and function calls to your main analysis scripts.")
    print("Then you'll get interactive HTML spectrograms with hover values!")