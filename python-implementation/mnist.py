import numpy as np
from srnn import LIF, SoftmaxOutputLayer
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

def poisson_encode(image: np.ndarray, max_rate=100, duration=100):
    """
    Convert an image to a binary spike train over time.
    image: 28x28 pixel values [0, 1]
    returns: (duration, 784) spike train
    """
    flat_image = image.flatten()
    spike_probs = flat_image * max_rate / 1000  # per ms
    return np.random.rand(duration, 784) < spike_probs


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

def build_hidden_layer(num_inputs=784, num_hidden=100):
    hidden_neurons = []
    for i in range(num_hidden):
        connections = {
            j: np.random.normal(0, 0.1)
            for j in range(num_inputs)
        }
        neuron = LIF(
            firing_threshold=1.0,
            leak_alpha=0.9,
            output_leak_alpha=0.9,
            connections=connections,
            output_size=10, 
            learning_rate=0.01
        )
        hidden_neurons.append(neuron)
    return hidden_neurons

def build_output_layer(num_outputs=10):
    return SoftmaxOutputLayer(
        num_hidden=100,
        num_outputs=num_outputs,
        learning_rate=0.01,
        leak_alpha=0.9
    )

def run_single_image(image, label, hidden, output_layer, duration=100):
    spikes = poisson_encode(image, duration=duration)
    spike_counts = np.zeros(output_layer.num_outputs)

    for t in range(duration):
        # Feed input spikes to hidden layer
        if t < duration:
            for pre_idx in range(784):
                if spikes[t, pre_idx]:
                    for h in hidden:
                        h.recieve_pulse(pre_idx)

        # Hidden neuron update
        for i, h in enumerate(hidden):
            h.next_time_step()
            if h.sending_pulse:
                for j in range(output_layer.num_outputs):
                    output_layer.receive_pulse(i)

        # Update output layer
        output_layer.update()

        # Compute error (cross-entropy gradient)
        error_signal = output_layer.compute_error(label)
        # Feedback error to hidden layer
        for h in hidden:
            h.recieve_error(error_signal)
        output_layer.accumulate_gradient(error_signal)

        # print(output_layer.membrane_potentials)
        # print("Label", label,", p: ", output_layer.output()[label])

    # Accumulate spike count for output prediction
    spike_counts += output_layer.output()

    # Update LIF weights
    for h in hidden:
        h.reset()
    output_layer.reset()

    return spike_counts

# Full training loop
def train(hidden, output, images, labels, epochs=1):
    for epoch in range(epochs):
        correct = 0
        i = 0
        for img, lbl in tqdm(zip(images, labels), total=len(images)):
            spike_counts = run_single_image(img, lbl, hidden, output, duration=10)
            pred = np.argmax(spike_counts)
            i += 1
            if i%10== 0:
                output.update_parameters()
                for h in hidden:
                    h.update_parameters()
            if pred == lbl:
                correct += 1
        acc = correct / len(images)
        print(f"Epoch {epoch+1} accuracy: {acc:.3f}")


images, labels = load_mnist(n_samples=1000)
# images, labels = images[6:7], labels[6:7]
print(labels)
hidden = build_hidden_layer()
output = build_output_layer()
train(hidden, output, images, labels, epochs=500)
