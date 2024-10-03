import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def generate_data_stream(length, seasonality, random_noise):
    """
    Generates a simulated data stream with seasonal patterns and random noise.

    Args:
        length: Length of the data stream.
        seasonality: Seasonality factor.
        random_noise: Random noise level.

    Returns:
        A numpy array representing the data stream.
    """

    time_series = np.zeros(length)

    for i in range(length):
        time_series[i] = np.sin(i / seasonality) + np.random.normal(0, random_noise)
        #print(f"Index: {i}, Value: {time_series[i]}")
    return time_series

def detect_anomalies(data_stream, window_size, threshold):
    """
    Detects anomalies in the data stream using a simple moving average approach.

    Args:
        data_stream: The input data stream.
        window_size: Size of the moving average window.
        threshold: Anomaly threshold.

    Returns:
        A list of indices where anomalies were detected.
    """

    anomalies = []
    moving_average = np.convolve(data_stream, np.ones(window_size) / window_size, mode='valid')

    for i in range(len(moving_average)):
        if abs(data_stream[i + window_size - 1] - moving_average[i]) > threshold:
            anomalies.append(i + window_size - 1)
            #print(f"Anomaly detected at index: {i + window_size - 1}")

    return anomalies

def detect_anomalies_vectorized(data_stream, window_size, threshold):
    """
    Utilize NumPy's vectorized operations: 
    Instead of iterating over the data stream, perform calculations on entire arrays. 
    This can significantly improve performance.
    """
    moving_average = np.convolve(data_stream, np.ones(window_size) / window_size, mode='valid')
    anomaly_indices = np.where(np.abs(data_stream[window_size - 1:] - moving_average) > threshold)[0]
    
    return anomaly_indices

def detect_anomalies_incremental(data_stream, window_size, threshold):
    """
    Incremental update: 
    For each new data point, update the moving average incrementally instead of recalculating it from scratch. 
    This reduces computational overhead.
    """
    moving_average = np.sum(data_stream[:window_size]) / window_size
    anomalies = []

    for i in range(window_size, len(data_stream)):
        moving_average = (moving_average * window_size - data_stream[i - window_size] + data_stream[i]) / window_size
        if abs(data_stream[i] - moving_average) > threshold:
            anomalies.append(i)

    return anomalies

def visualize_data(data_stream, anomalies):
    """
    Visualizes the data stream and detected anomalies.

    Args:
        data_stream: The input data stream.
        anomalies: A list of indices where anomalies were detected.
    """

    plt.figure(figsize=(12, 6))
    plt.plot(data_stream, label='Data Stream')
    plt.scatter(anomalies, data_stream[anomalies], color='red', label='Anomalies')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Data Stream Anomaly Detection')
    plt.show()

def visualize_data_animated(data_stream, anomalies):
    """
    Animated visualization of the data stream and detected anomalies.

    Args:
        data_stream: The input data stream.
        anomalies: A list of indices where anomalies were detected.
    """

    fig, ax = plt.subplots(figsize=(12, 6))
    line, = ax.plot([], [], label='Data Stream')
    scatter, = ax.plot([], [], 'ro', label='Anomalies')
    ax.set_xlim(0, len(data_stream) / seasonality)
    ax.set_ylim(min(data_stream), max(data_stream))
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Data Stream Anomaly Detection')

    def update(frame):
        line.set_data(range(frame + 1), data_stream[:frame + 1])
        scatter.set_data(anomalies[:frame + 1], data_stream[anomalies[:frame + 1]])

        #if frame + 1 > window_size: #Off-screen follow
            #ax.set_xlim(frame + 1 - window_size, frame + 1)

        return line, scatter

    ani = FuncAnimation(fig, update, frames=len(data_stream), interval=100)
    plt.show()

if __name__ == '__main__':
    data_stream_length = 10000
    seasonality = 100
    random_noise = 0.5
    window_size = 50
    threshold = 2

    # Generate data stream
    data_stream = generate_data_stream(data_stream_length, seasonality, random_noise)

    # Detect anomalies
    #anomalies = detect_anomalies(data_stream, window_size, threshold)
    #anomalies = detect_anomalies_vectorized(data_stream, window_size, threshold)
    anomalies = detect_anomalies_incremental(data_stream, window_size, threshold)

    # Visualize data stream
    visualize_data(data_stream, anomalies)
    #visualize_data_animated(data_stream, anomalies)