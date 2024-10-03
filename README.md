# Data Stream Anomaly Detection
The process of identifying unusual patterns or behaviours within a data set. These irregularities in the data are considered anomalies or outliers; and could indicate an inconsistency or a malfunction in the system being monitored.

### Methods used for anomaly detection includes:
<ol>
<li>Statistical: Searches for anomalies by analysing a time series’ mean, variance, or distribution. (e.q. Z-score which measures the distance between a data point and the mean by standard deviation).</li>
<li>Machine Learning: By training a machine learning model to detect anomalies in time series dataset (e.q. Isolation forest & Support Vector Machines (SVMs)).</li>
<li>Signal Processing: Identifies changes in signal patterns generated by the monitored system (e.q. Fourier transformation).</li>
<li>Hybrid: Combining multiple techniques for accuracy improvement.</li>
</ol>

## Algorithm Selection
The chosen method is a simple and efficient implementation using NumPy's vectorized operations.

This approach involves calculating the moving average using a convolution operation, which is optimized for performance in NumPy. The convolution efficiently computes the average of a fixed number of data points within a sliding window. This method is suitable for data streams where the computational cost is not a major concern and where the underlying data distribution is relatively stable.

While moving average may not be as effective as more sophisticated algorithms for handling concept drift and seasonal variations, it can be a good choice for basic anomaly detection tasks, especially when combined with other techniques or when the data stream is relatively small.

## Data Stream Simulation
Generating a simulated data stream with seasonal patterns and random noise:
```
def generate_data_stream(length, seasonality, random_noise):
    time_series = np.zeros(length)
    for i in range(length):
        time_series[i] = np.sin(i / seasonality) + np.random.normal(0, random_noise)
    return time_series
```

Debug: [length = 10000, seasonality = 100, random_noise = 0.5]
<br /><br />
![image](https://github.com/user-attachments/assets/5b5bcbb8-bdd3-481b-916c-2370beba1bd6)

## Anomaly Detection
Detecting anomalies in the data stream using a simple moving average approach:
```
def detect_anomalies(data_stream, window_size, threshold):
    anomalies = []
    moving_average = np.convolve(data_stream, np.ones(window_size) / window_size, mode='valid')
    for i in range(len(moving_average)):
        if abs(data_stream[i + window_size - 1] - moving_average[i]) > threshold:
            anomalies.append(i + window_size - 1)
    return anomalies
```

Debug: [random_noise = 0.6, window_size = 50, threshold = 2]
<br /><br />
![image](https://github.com/user-attachments/assets/06e087b1-c336-4c27-aab6-c4126e66a1ce)

## Optimization
The current function uses a simple moving average approach to identify anomalies. While effective for basic scenarios, it can be computationally expensive for large data streams.

### Optimization Strategies:

1. Vectorization:
Utilizing NumPy’s vectorized operations to perform calculations on entire arrays instead of iterating over the data stream can significantly improve performance.
```
def detect_anomalies_vectorized(data_stream, window_size, threshold):

    moving_average = np.convolve(data_stream, np.ones(window_size) / window_size, mode='valid')
    anomaly_indices = np.where(np.abs(data_stream[window_size - 1:] - moving_average) > threshold)[0]
    
    return anomaly_indices
```

2. Incremental update: 
Reducing computational overhead by updating the moving average incrementally for each new data point instead of recalculating it from scratch.
```
def detect_anomalies_incremental(data_stream, window_size, threshold):
    
    moving_average = np.sum(data_stream[:window_size]) / window_size
    anomalies = []

    for i in range(window_size, len(data_stream)):
        moving_average = (moving_average * window_size - data_stream[i - window_size] + data_stream[i]) / window_size
        if abs(data_stream[i] - moving_average) > threshold:
            anomalies.append(i)

    return anomalies
```

## Visualization
Adjust the parameters [stream length, seasonality, random noise, window size, and threshold] to the desired values and run the visualization (Static or animated):
```
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
```

![image](https://github.com/user-attachments/assets/b96e0166-9862-4401-922d-fadbe7aa02b8)
<br />
![image](https://github.com/user-attachments/assets/d2e05f85-5a26-45eb-8666-9bd2e0eba99e)

