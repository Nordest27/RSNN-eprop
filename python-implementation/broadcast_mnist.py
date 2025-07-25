import numpy as np
import scipy.sparse as sp
from simple_broadcast_srnn import SimpleBroadcastSrnn
from local_broadcast_srnn import LocalBroadcastSrnn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import json
import time
import random


def poisson_encode(image: np.ndarray, duration=100, max_prob=0.25):
    """
    Convert an image to a binary spike train over time.
    image: 28x28 pixel values [0, 1]
    returns: (duration, 784) spike train
    """
    flat_image = image.flatten()
    spike_probs = max_prob*flat_image / (max(flat_image))
    spikes = np.random.rand(duration, 784) < spike_probs
    return spikes

def freq_encode(image: np.ndarray, duration=100):
    """
    Deterministic frequency-based spike encoding.
    Each pixel spikes periodically over time with frequency proportional to brightness.
    
    Args:
        image: 28x28 grayscale image with values in [0, 1]
        duration: Number of time steps to encode
    
    Returns:
        spike_train: Binary array of shape (duration, 784)
    """
    flat_image = image.flatten()
    spike_train = np.zeros((duration, 784), dtype=np.uint8)
    
    for i, intensity in enumerate(flat_image):
        if intensity > 0:
            freq = int(intensity * duration)
            if freq >= duration:
                spike_train[:, i] = 1  # always spikes
            else:
                spike_indices = np.round(np.linspace(0, duration - 1, freq)).astype(int)
                spike_train[spike_indices, i] = 1
                
    return spike_train

def bucket_encode(image: np.ndarray, duration=100, discretization=5):
    flat_image = image.flatten()
    spike_train = np.zeros((duration, 784), dtype=np.uint8)

    # Assign each pixel to a bucket (0 = brightest, discretization-1 = dimmest)
    bucket_indices = (discretization - 1 - (flat_image * (discretization - 1)).astype(int)).clip(0, discretization - 1)

    for t in range(discretization-1):
        spike_train[t, bucket_indices == t] = 1

    return spike_train

def bucket_cycle_encode(image: np.ndarray, duration=100, discretization=5):
    """
    Cyclical rank-based spike encoding.
    
    Args:
        image: 28x28 grayscale image with values in [0, 1]
        duration: Number of time steps to encode
        discretization: Number of brightness buckets
        
    Returns:
        spike_train: Binary array of shape (duration, 784)
    """
    flat_image = image.flatten()
    spike_train = np.zeros((duration, 784), dtype=np.uint8)

    # Assign each pixel to a bucket (0 = brightest, discretization-1 = dimmest)
    bucket_indices = (discretization - 1 - (flat_image * (discretization - 1)).astype(int)).clip(0, discretization - 1)

    for t in range(duration):
        if t % discretization != discretization-1:
            active_bucket = t % discretization
            spike_train[t, bucket_indices == active_bucket] = 1

    return spike_train


# Define dataset loader
def load_mnist(n_samples=1000):
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    images = []
    labels = []
    for i in range(n_samples if n_samples != -1 else len(mnist)):
        img, lbl = mnist[i]
        images.append(img.squeeze().numpy())
        labels.append(lbl)
    return np.array(images), np.array(labels)

# --- New run_single_image for broadcasting SRNN ---
def run_single_image_broadcast(image, label, srnn: LocalBroadcastSrnn | SimpleBroadcastSrnn, duration):
    srnn.reset()
    spikes = poisson_encode(image, duration=duration, max_prob=0.01)
    # spikes = freq_encode(image, duration=duration)
    # spikes = bucket_encode(image, duration=duration)

    # noise = (np.random.rand(spikes.shape[0], spikes.shape[1]) < 0.01)
    # spikes = spikes | noise

    input_size = srnn.input_size
    total_size = srnn.input_size + srnn.total_neurons
    for t in range(duration):
        # Prepare input vector: put spikes in the first input_size positions
        input_row = np.zeros((1, total_size))
        input_row[0, :input_size] = spikes[t]
        input_csr = sp.csr_matrix(input_row)
        # Step the network
        srnn.input(input_csr)
        output = srnn.output()

    # Prediction: class with most spikes
    pred = np.argmax(output)
    loss = srnn.feedback(label)
    return loss, pred

# --- New train/test loops for broadcasting SRNN ---
def train_broadcast(srnn: LocalBroadcastSrnn | SimpleBroadcastSrnn, images, labels, epochs=1, duration=10):
    for epoch in range(epochs):
        correct = 0
        total_loss = 0
        b = 0
        for i, (img, lbl) in enumerate(tqdm(zip(images, labels), total=len(images), desc=f"Epoch {epoch+1}")):
            loss, pred = run_single_image_broadcast(img, lbl, srnn, duration)
            if pred == lbl:
                correct += 1
            if loss is not None:
                total_loss += loss
            
            b += 1
            if b == batch_size:
                srnn.update_parameters()
                srnn.update_outweights()
                b = 0

        acc = correct / len(images)
        avg_loss = total_loss / len(images)
        print(f"Epoch {epoch+1} - Accuracy: {acc:.3f}, Average Loss: {avg_loss:.3f}")

def test_broadcast(srnn: LocalBroadcastSrnn | SimpleBroadcastSrnn, images, labels, duration):
    correct = 0
    for img, lbl in tqdm(zip(images, labels), total=len(images), desc="Testing"):
        _, pred = run_single_image_broadcast(img, None, srnn, duration)
        if pred == lbl:
            correct += 1
    acc = correct / len(images)
    print(f"Test Accuracy: {acc:.3f}")
    return acc

# Main execution
if __name__ == "__main__":
    # Load data
    print("Loading MNIST dataset...")
    duration = 100
    batch_size = 1
    n_hidden = 64
    images, labels = load_mnist(n_samples=15)
    train_images, train_labels = images[:int(len(images)*0.8)], labels[:int(len(images)*0.8)]
    test_images, test_labels = images[int(len(images)*0.8):], labels[int(len(images)*0.8):]
    print(f"Loaded {len(train_images)} training samples")
    print(f"Label distribution: {np.bincount(train_labels)}")

    # Build broadcasting SRNN
    print("Building broadcasting SRNN network...")
    num_layers = 8
    num_neurons_list = [n_hidden for _ in range(num_layers)]
    input_size = 784
    output_size = 10
    input_connectivity = 0.3
    hidden_connectivity = 0.025
    output_connectivity = 0.3
    local_connectivity = 0.13
    # srnn = SimpleBroadcastSrnn(
    srnn = LocalBroadcastSrnn(
        num_neurons_list=num_neurons_list, 
        input_size=input_size, 
        output_size=output_size,
        input_connectivity=input_connectivity,
        hidden_connectivity=hidden_connectivity,
        output_connectivity=output_connectivity,
        local_connectivity=local_connectivity
    )

    print(f"Number of neurons: {sum(num_neurons_list)}")
    complete_connections = (
        input_size * sum(num_neurons_list) 
        + sum(num_neurons_list)**2
        + sum(num_neurons_list) * output_size
    )
    print(f"Total number of possible connections: {complete_connections}")

    estimated_connections = (
        input_size * sum(num_neurons_list) * input_connectivity
        + sum(n**2 for n in num_neurons_list) * local_connectivity 
        + sum(num_neurons_list)**2 * hidden_connectivity
        - sum(n**2 for n in num_neurons_list) * hidden_connectivity 
        + sum(num_neurons_list) * output_size * output_connectivity
    )
    print(f"Estimated number of connections: {int(estimated_connections)}")
    n = sum(num_neurons_list)  # total number of hidden neurons
    p = hidden_connectivity
    prob_connected = max(0.0, 1 - n * (1 - p) ** (n - 1))
    print(f"Probability of connected hidden layer (lower bound): {prob_connected:.4f}")
    if prob_connected < 0.01:
        print("Warning: The hidden layer is almost certainly disconnected at this connectivity.")

    # Check local (internal) connectivity for each layer
    for i, n in enumerate(num_neurons_list):
        p_local = local_connectivity
        prob_connected_local = max(0.0, 1 - n * (1 - p_local) ** (n - 1))
        print(f"Layer {i+1}: n={n}, local_connectivity={p_local}, Probability of connected local subgraph: {prob_connected_local:.4f}")
        if prob_connected_local < 0.01:
            print(f"  Warning: Layer {i+1} is almost certainly disconnected internally at this local connectivity.")
    # print(f"Actual number of connections {json.dumps(srnn.get_n_connections(), indent=2)}")
    
    # Train network (no learning yet, just demo)
    print("Training network...")
    train_broadcast(srnn, train_images, train_labels, epochs=100, duration=duration)

    # Test on test data
    print("Testing network...")
    test_broadcast(srnn, test_images, test_labels, duration)

    # Shutdown processes
    srnn.shutdown()
