import numpy as np
import matplotlib.pyplot as plt

def smooth_time_series(series: np.ndarray, window_size: int) -> np.ndarray:
    """
    Smoothes a 1D time series using a simple moving average.

    Args:
        series: The input 1D numpy array (time series).
        window_size: The size of the moving average window. Must be an odd integer
                     to ensure the output is centered.

    Returns:
        A new numpy array containing the smoothed time series.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd number to be centered.")

    # Create the averaging window
    window = np.ones(window_size) / window_size
    
    smoothed_series = np.convolve(series, window, mode='same')

    edge_width = window_size // 2
    smoothed_series[:edge_width] = series[:edge_width]
    smoothed_series[-edge_width:] = series[-edge_width:]
    
    return smoothed_series

def main():
    time = np.linspace(0, 1000, 5000)

    a=np.load("result2.npz")
    original_series = a['u'][:,2]
    # Define smoothing parameters
    window_size = 111  # Must be an odd number

    # Smooth
    try:
        smoothed_series = smooth_time_series(original_series, window_size)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # 4. Plot the original and smoothed time series
    plt.figure(figsize=(12, 6))
    plt.plot(time, original_series, label='Original Series', alpha=0.6, color='blue')
    plt.plot(time, smoothed_series, label=f'Smoothed (Window={window_size})', color='red', linewidth=2.5)
    
    plt.title('Time Series Smoothing')
    plt.xlabel('Time')
    plt.ylabel('Control Input Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Show the plot
    plt.show()

if __name__ == "__main__":   
    main()