import numpy as np
import matplotlib.pyplot as plt
import os

if __name__=='__main__':
    file_name = 'datac.npz'
    data_archive = np.load(file_name)

    # Extract the arrays using their defined keys
    k_data = data_archive['k']
    ktext_data = data_archive['ktext']

    data_archive.close()
    
    # --- 3. PLOT THE DATA AND THE y=x LINE ---
    
    # Use a square figure size for a non-distorted comparison to the y=x line
    plt.figure(figsize=(8, 8))

    # Plot 1: The experimental data (ktext against k)
    plt.plot(ktext_data, k_data, 
             'o-', 
             label='k_text vs k', 
             color='blue', 
             markersize=6)
    
    # Plot 2: Add the y=x line for comparison
    # Determine the plotting range based on the minimum and maximum data values
    min_val = min(np.min(k_data), np.min(ktext_data))
    max_val = max(np.max(k_data), np.max(ktext_data))
    
    # Create a line that spans slightly beyond the data range
    line_range = np.linspace(min_val, max_val, 2)

    plt.plot(line_range, line_range, 
             color='red', 
             linestyle='--', 
             linewidth=2,
             label='Line (y = x)') 
    
    # Set labels and title
    plt.xlabel('k_text=Gd^4/(8ND^3) (N/m)', fontsize=14)
    plt.ylabel('k (N/m)', fontsize=14)
    plt.title('k_text vs k Comparison to y=x', fontsize=16)

    # Add legend
    plt.legend(fontsize=12)
    
    # Ensure the axes have the same scale for a true 1:1 comparison
    plt.axis('equal') 
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
