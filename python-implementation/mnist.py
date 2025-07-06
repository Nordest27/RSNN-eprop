import numpy as np
import scipy.sparse as sp
from srnn import LIFLayer, SoftmaxOutputLayer  # Updated import
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

def poisson_encode(image: np.ndarray, duration=100):
    """
    Convert an image to a binary spike train over time.
    image: 28x28 pixel values [0, 1]
    returns: (duration, 784) spike train
    """
    flat_image = image.flatten()
    spike_probs = flat_image / (max(flat_image)*duration)
    spikes = np.random.rand(duration, 784) < spike_probs
    return spikes

# Define dataset loader
def load_mnist(n_samples=1000):
    transform = transforms.Compose([transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    images = []
    labels = []
    for i in range(n_samples):
        img, lbl = mnist[i]
        images.append(img.squeeze().numpy())
        labels.append(lbl)
    return np.array(images), np.array(labels)

def build_hidden_layer(num_inputs=784, num_hidden=100, connection_density=0.05):
    """
    Build a hidden layer using LIFLayer instead of individual LIF neurons.
    Total input size includes both external inputs and recurrent connections.
    """
    total_input_size = num_inputs + num_hidden  # External + recurrent
    
    hidden_layer = LIFLayer(
        num_inputs=total_input_size,
        num_neurons=num_hidden,
        learning_rate=1e-2,
        connection_density=connection_density,
        output_size=10
    )
    
    return hidden_layer

def build_output_layer(num_outputs=10, num_hidden=100, connection_density=0.05):
    return SoftmaxOutputLayer(
        num_hidden=num_hidden,
        num_outputs=num_outputs,
        learning_rate=1e-2,
        connection_density=connection_density
    )

def run_single_image(image, label, hidden_layer, output_layer, duration):
    """
    Run a single image through the network.
    """
    spikes = poisson_encode(image, duration=duration)
    spikes_shape = spikes.shape
    spikes = sp.csr_matrix(spikes, shape=spikes_shape)
    # Initialize output spike counts
    spike_counts = np.zeros(output_layer.num_outputs)
    hidden_output = hidden_layer.get_output_spikes()
    for t in range(duration):
        input_spike_vector = sp.hstack([
            spikes.getrow(t), 
            hidden_output.reshape(1, -1)
        ])
        # Feed input to hidden layer
        hidden_layer.receive_pulse(input_spike_vector)
        hidden_layer.next_time_step()
        
        # Get hidden layer output and feed to output layer
        hidden_output = hidden_layer.get_output_spikes()
        if hidden_output.nnz > 0:  # Only if there are spikes
            output_layer.receive_pulse(hidden_output)
        
        output_layer.update()
        
    loss = None
    if label is not None:
        # Compute error and backpropagate
        error_signal = output_layer.compute_error(label)
        loss = output_layer.compute_loss(label)
        
        # Backpropagate to hidden layer
        hidden_layer.receive_error(error_signal)
        
        # Accumulate gradients for output layer
        output_layer.accumulate_gradient(error_signal)

    output = output_layer.output()
    # Reset layers for next image
    hidden_layer.reset()
    output_layer.reset()
    
    return loss, output

def train(hidden_layer, output_layer, images, labels, epochs=1, batch_size=10, duration=10):
    """
    Training loop with batch updates.
    """
    for epoch in range(epochs):
        correct = 0
        total_loss = 0
        
        for i, (img, lbl) in enumerate(tqdm(zip(images, labels), total=len(images), desc=f"Epoch {epoch+1}")):
            # Run single image
            loss, output = run_single_image(img, lbl, hidden_layer, output_layer, duration)
            
            # Make prediction
            pred = np.argmax(output)
            if pred == lbl:
                correct += 1
            
            # Compute loss for monitoring
            total_loss += loss
            
            # Update parameters every batch_size samples
            if (i + 1) % batch_size == 0:
                output_layer.update_parameters()
                hidden_layer.update_parameters()
                # out_w = output_layer.weights.T
                # hidden_layer.loss_weights[out_w.indices] = out_w[out_w.indices]

        # Final parameter update if there are remaining samples
        if len(images) % batch_size != 0:
            output_layer.update_parameters()
            hidden_layer.update_parameters()
            # out_w = output_layer.weights.T
            # hidden_layer.loss_weights[out_w.indices] = out_w[out_w.indices]

        
        # Print epoch statistics
        acc = correct / len(images)
        avg_loss = total_loss / len(images)
        print(f"Epoch {epoch+1} - Accuracy: {acc:.3f}, Average Loss: {avg_loss:.3f}")

def test(hidden_layer, output_layer, images, labels, duration):
    """
    Test the trained network.
    """
    correct = 0

    for img, lbl in tqdm(zip(images, labels), total=len(images), desc="Testing"):
        _, output = run_single_image(img, None, hidden_layer, output_layer, duration)
        pred = np.argmax(output)
        if pred == lbl:
            correct += 1
    
    acc = correct / len(images)
    print(f"Test Accuracy: {acc:.3f}")
    return acc

# Main execution
if __name__ == "__main__":
    # Load data
    print("Loading MNIST dataset...")
    duration = 20
    train_images, train_labels = load_mnist(n_samples=100)
    print(f"Loaded {len(train_images)} training samples")
    print(f"Label distribution: {np.bincount(train_labels)}")
    
    # Build network
    print("Building network...")
    hidden_layer = build_hidden_layer(num_hidden=50, connection_density=0.5)
    output_layer = build_output_layer(num_hidden=50, connection_density=0.5)
    
    # Train network
    print("Training network...")
    train(hidden_layer, output_layer, train_images, train_labels, 
          epochs=50, batch_size=2, duration=duration)
    
    # Test on training data (you should use separate test data in practice)
    print("Testing network...")
    test(hidden_layer, output_layer, train_images, train_labels, duration)
