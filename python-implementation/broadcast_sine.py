import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from simple_broadcast_srnn import SimpleBroadcastSrnn
from local_broadcast_srnn import LocalBroadcastSrnn
from tqdm import tqdm

def generate_sine_wave_sequences(n_samples=1000, seq_length=50, dt=0.1, noise_std=0, n_components=5):
    """
    Generate sine wave sequences for next-value prediction.
    
    Args:
        n_samples: Number of training sequences
        seq_length: Length of each input sequence
        dt: Time step
        noise_std: Standard deviation of noise
    
    Returns:
        inputs: Input sequences (n_samples, seq_length, 1)
    """
    np.random.seed(102)
    inputs = []
    for _ in range(n_samples):
        
        # Generate longer time series to extract sequence + next value
        t = np.arange(seq_length) * dt
        wave = t*0
        for i in range(n_components):
            # Random starting phase and frequency for variety
            phase = np.random.uniform(0, 2*np.pi)
            amplitude = np.random.uniform(0.01, 1.0)
            frequency = np.random.uniform(0.01, 0.5)
            sine_wave = amplitude * np.sin(2 * np.pi * frequency * t  + phase)
            wave += sine_wave

        # sine_wave[len(sine_wave)//2:] -= 10
        # Add noise
        if noise_std > 0:
            wave += np.random.normal(0, noise_std, seq_length)
        
        wave -= wave[0]
        wave /= abs(wave).max()

        # Input is sequence of length seq_length, target is the next value
        inputs.append(wave.reshape(-1, 1))  # Input sequence
    
    return inputs

def encode_continuous_to_spikes(input_dim, values, duration=10, encoding_type='rate'):
    """
    Convert continuous values to spike trains.
    
    Args:
        values: Array of continuous values (sequence_length, input_dim)
        duration: Number of time steps for encoding each value
        encoding_type: 'rate' or 'temporal'
    
    Returns:
        spike_trains: Binary spike trains (total_time_steps, input_dim)
    """
    if values.ndim != 1:
        values = values.reshape(-1)
    
    spike_probs = np.zeros_like(values)
    col_min, col_max = values.min(), values.max()
    spike_probs = (values - col_min) / (col_max - col_min)

    spikes = np.zeros((duration, input_dim))
    for t in range(duration):
        spikes[t] = np.random.rand(input_dim) < spike_probs[t]
        # for i in range(input_dim):
        #    spikes[t, i] = spike_probs[t] > i/input_dim

    return spikes


def run_sequence(input_dim, sequence, target, srnn: LocalBroadcastSrnn):
    """Run a single sequence through the network for next-value prediction"""
    # Reset for next sequence
    srnn.reset()
    duration = len(sequence)
    # Encode sequence to spikes
    spikes = encode_continuous_to_spikes(input_dim=input_dim, values=sequence, duration=duration, encoding_type='rate')

    total_loss = 0
    predictions = []

    # Initialize
    total_size = srnn.input_size + srnn.total_neurons

    for t in range(duration):
        # Create input vector (sequence spikes + recurrent spikes)
        input_row = np.zeros((1, total_size))
        # condition = t < duration//2
        # if condition: 
        input_row[0, :srnn.input_size] = fixed_noise[t]
        # else:
        #     noise = (np.random.rand(input_row.shape[0], input_row.shape[1]) < 0.1)
        #     input_row = np.maximum(input_row, noise)
        input_csr = sp.csr_matrix(input_row)
        # Step the network
        srnn.input(input_csr)
        output = srnn.output()
    
        predictions.append(output)
        if target is not None:
            total_loss += srnn.feedback(target[t])

    return total_loss, predictions

def train_regression(input_dim, srnn: LocalBroadcastSrnn, sequences, epochs=50, batch_size=10):
    """Train the next-value prediction network"""
    n_samples = len(sequences)
    
    train_losses = []

    for epoch in range(epochs):
        total_loss = 0
        mse_error = 0
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        
        for i, idx in enumerate(tqdm(indices, desc=f"Epoch {epoch+1}")):
            seq = sequences[idx]
            seq, target_seq = seq, seq #seq[:-1], seq[1:]
            
            # Run sequence
            loss, predictions = run_sequence(input_dim, seq, target_seq, srnn)
            
            total_loss += loss
            mse_error += (predictions - target_seq) ** 2
            
            # Update parameters every batch_size samples
            if (i + 1) % batch_size == 0:
                srnn.update_parameters()
                srnn.update_outweights()
        
        # Final update if remaining samples
        if n_samples % batch_size != 0:
            srnn.update_parameters()
            srnn.update_outweights()
        
        # Print statistics
        avg_loss = total_loss / n_samples
        rmse = np.sqrt(mse_error / n_samples)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} - Loss: {float(avg_loss):.4f}, MSE: {np.mean(rmse):.4f}")
    
    return train_losses


def plot_simple_example(sequences, srnn, seq_length=20):
    """Plot a simple complete example to see if the model fits the pattern"""

    # true_future = sine_wave[1:]
    true_future = sequences[np.random.randint(len(sequences))]

    for i in range(10):
        _, predictions = run_sequence(input_dim, true_future, None, srnn)

    # Plot the complete example
    plt.figure(figsize=(12, 6))
    indices = list(range(0, seq_length))
    
    plt.plot(indices, true_future, 'g-', linewidth=3, label='True Future')
    
    # Plot predictions
    plt.plot(indices, predictions, 'r--', linewidth=2, label='Model Predictions')
    

    plt.xlabel('Time Step')
    plt.ylabel('Amplitude')
    plt.title('Sine Wave Prediction: Does the Model Fit the Pattern?')
    # plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print some metrics
    mse = np.mean((np.array(predictions) - true_future) ** 2)

    # correlation = np.corrcoef(true_future, np.array(predictions).reshape(-1))[0, 1] if len(predictions) > 1 else 0
    print(f"Prediction MSE: {mse:.4f}")
    # print(f"Correlation with true future: {correlation:.3f}")


# Main execution
if __name__ == "__main__":
    print("Generating sine wave sequences for next-value prediction...")
    
    # Generate data with shorter sequences for better learning
    seq_length = 250  # Length of input sequence
    
    train_sequences = generate_sine_wave_sequences(
        n_samples=1, seq_length=seq_length,
    )
    
    # Build broadcasting SRNN
    print("Building broadcasting SRNN network...")
    batch_size = 1
    input_dim = 20
    n_hidden = 200
    num_layers = 15
    num_neurons_list = [n_hidden for _ in range(num_layers)]
    output_size = 1
    input_connectivity = 0.3
    hidden_connectivity = 0.05
    output_connectivity = 0.3
    local_connectivity = 0.1
    # srnn = SimpleBroadcastSrnn(
    srnn = LocalBroadcastSrnn(
        num_neurons_list=num_neurons_list, 
        input_size=input_dim, 
        output_size=output_size,
        input_connectivity=input_connectivity,
        hidden_connectivity=hidden_connectivity,
        output_connectivity=output_connectivity,
        local_connectivity=local_connectivity,
        output_activation_function="linear",
        # target_firing_rate=13,
        # self_predict=True,
        # tau_out=20e-3,
        # unary_weights=True,
    )
    
    # Train network
    print("Training network...")

    fixed_noise = (np.random.rand(seq_length, input_dim) < 0.25)
    
    train_losses = train_regression(
        input_dim,
        srnn,
        train_sequences,
        epochs=100,
        batch_size=batch_size
    )
    
    # Test with a simple complete example visualization
    print("Testing with complete example visualization...")
    while input("Visualize?"):
        plot_simple_example(train_sequences, srnn, seq_length)
