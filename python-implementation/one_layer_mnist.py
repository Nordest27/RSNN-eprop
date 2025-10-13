import numpy as np
import scipy.sparse as sp
from srnn import ALIFLayer, OutputLayer, ActorCriticOutputLayer
from visualize import SRNNVisualizer
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from line_profiler import profile

TAU_OUT = 3e-3

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
    discretization = min(discretization, duration)

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

def bucket_multiple_encode(images: list[np.ndarray], duration=100, discretization=5):
    spike_train = np.zeros((duration, 784), dtype=np.uint8)

    for i, image in enumerate(images):
        flat_image = image.flatten()
        # Assign each pixel to a bucket (0 = brightest, discretization-1 = dimmest)y
        bucket_indices = (discretization - 1 - (flat_image * (discretization - 1)).astype(int)).clip(0, discretization - 1)
                
        for t in range(duration):
            if t % discretization == discretization-1:
                continue
            spike_train[t, bucket_indices == (t - discretization * i)] = 1

    return spike_train

def get_input(image, duration):
    spikes = poisson_encode(image, duration=duration, max_prob=0.1)
    # spikes = freq_encode(image, duration=duration)
    # spikes = bucket_encode(image, duration)
    # spikes = bucket_cycle_encode(image, duration)
    # spikes = bucket_multiple_encode(image, duration=duration)
    # noise = (np.random.rand(spikes.shape[0], spikes.shape[1]) < 0.01)
    # spikes = spikes | noise
    # spikes = np.ones(spikes.shape) - spikes
    return spikes


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

def build_hidden_layers(
    num_inputs=784, 
    num_hidden=100,
    num_outputs=10,
    input_connection_density=0.1,
    local_connection_density=0.05, 
):
    """
    Build a hidden layer using LIFLayer instead of individual LIF neurons.
    Total input size includes both external inputs and recurrent connections.
    """
    hidden_layer = ALIFLayer(
        just_input_size=num_inputs,
        num_inputs=num_inputs + num_hidden,
        num_neurons=num_hidden,
        learning_rate=1e-3,
        tau_out=TAU_OUT,
        input_connection_density=input_connection_density,
        local_connection_density=local_connection_density,
        output_sizes={
            "base": num_outputs,
            "policy": num_outputs,
            "value": 1
        },
        target_firing_rate=50,
        firing_threshold=1.0,
        beta="sparse_adaptive",
        beta_params={
            "lif_fraction": 0.6,  # Fraction of LIF neurons
            "exp_scale": 0.2,    # Scale parameter for exponential distribution
            "max_beta": 2.0      # Maximum beta value
        }
    )
    teacher_input = num_inputs + num_hidden
    teacher_layer = ALIFLayer(
        just_input_size=teacher_input,
        num_inputs=teacher_input + num_hidden,
        num_neurons=num_hidden,
        learning_rate=1e-3,
        tau_out=TAU_OUT,
        input_connection_density=input_connection_density,
        local_connection_density=local_connection_density,
        output_sizes={
            "policy": 2*num_hidden,
            "value": 1,
        },
        firing_threshold=1.0,
        beta="sparse_adaptive",
        beta_params={
            "lif_fraction": 0.6,  # Fraction of LIF neurons
            "exp_scale": 0.2,    # Scale parameter for exponential distribution
            "max_beta": 2.0      # Maximum beta value
        }
    )

    print(teacher_layer.analyze_beta_distribution())
    return teacher_layer, hidden_layer

def build_output_layers(num_outputs=10, num_hidden=100, connection_density=0.05):
    return {
        "base": OutputLayer(
            num_hidden=num_hidden,
            num_outputs=num_outputs,
            learning_rate=1e-3,
            tau_out=TAU_OUT,
            connection_density=connection_density,
            activation_function="softmax"
        ),
        "ac_number_predict": ActorCriticOutputLayer(
            num_hidden=num_hidden,
            num_outputs=num_outputs,
            learning_rate=1e-3,
            tau_out=TAU_OUT,
            connection_density=connection_density,
            policy_activation_function="softmax"
        ),
        "ac_learned_losses": ActorCriticOutputLayer(
            num_hidden=num_hidden,
            num_outputs=num_hidden,
            learning_rate=1e-2,
            tau_out=TAU_OUT,
            connection_density=connection_density,
            policy_activation_function="linear"
        )
    }

@profile
def run_single_image(
    image, 
    label, 
    teacher_layer: ALIFLayer,
    hidden_layer: ALIFLayer,
    output_layers: list[OutputLayer | ActorCriticOutputLayer], 
    duration
):  
    # teacher_layer.reset()
    # hidden_layer.full_reset()
    hidden_layer.reset()
    for output_layer in output_layers.values():
        output_layer.reset()

    spikes = get_input(image, duration)
    label = [label] if not isinstance(label, list) else label

    spikes_shape = spikes.shape
    spikes = sp.csr_array(spikes, shape=spikes_shape)

    # teacher_output = teacher_layer.get_output_spikes()
    hidden_output = hidden_layer.get_output_spikes()
    outputs = []
    sequence = []
    
    for t in range(duration):
        sequence.append(spikes._getrow(t))
        input_spike_vector = sp.hstack([
            spikes._getrow(t) if t < duration-1 else 0 * spikes._getrow(t), 
            hidden_output.reshape(1, -1)
        ])
        hidden_layer.receive_pulse(input_spike_vector)
        hidden_layer.next_time_step()

        # teacher_input_spike_vector = sp.hstack([
        #     spikes._getrow(t), 
        #     hidden_output.reshape(1, -1),
        #     teacher_output.reshape(1, -1)
        # ])
        # teacher_layer.receive_pulse(teacher_input_spike_vector)
        # teacher_layer.next_time_step()

        # teacher_output = teacher_layer.get_output_spikes()
        hidden_output = hidden_layer.get_output_spikes()

        # output_layers["base"].receive_pulse(hidden_output)
        output_layers["ac_number_predict"].receive_pulse(hidden_output)
        # output_layers["ac_learned_losses"].receive_pulse(teacher_output)
        
        # output_layers["base"].next_time_step()
        output_layers["ac_number_predict"].next_time_step()
        # output_layers["ac_learned_losses"].next_time_step()

        # learned_losses = output_layers["ac_learned_losses"].action()
        # prev_weights = hidden_layer.weights
        # hidden_layer.receive_loss_signal(learned_losses)
        # print(sum(sum(abs(prev_weights-hidden_layer.weights))))

        # action = output_layers["base"].output()
        
        action = output_layers["ac_number_predict"].action()
        action_label = np.argmax(action)
        reward = 0
        this_state_td_error = 0
        if label[0] is not None:
            aux_label = label
            if isinstance(label, list):
                aux_label = label[0]
            reward = float(action_label==aux_label) * ((t+1)/duration)**2
            # print(reward)

            # supervised_error = output_layers["base"].compute_error(aux_label)
            # output_layers["base"].receive_error(supervised_error)
            # reward = -sum(abs(supervised_error))
            # hidden_layer.receive_error(supervised_error)

            policy_gradient, this_state_td_error, previous_policy_grads, previous_state_td_error = (
                output_layers["ac_number_predict"].td_error_update(reward)
            )
            hidden_layer.receive_rl_gradient(
                t,
                policy_gradient, this_state_td_error, 
                previous_policy_grads, previous_state_td_error
            )
            # policy_gradient, advantage = (
            #     output_layers["ac_learned_losses"].receive_reward(reward)
            # )
            # teacher_layer.receive_rl_gradient(
            #     policy_gradient, advantage,
            # )

    if label[0] is not None:
        hidden_layer.update_parameters()
        # teacher_layer.update_parameters()
        # output_layers["base"].update_parameters()
        output_layers["ac_number_predict"].update_parameters()
        # output_layers["ac_learned_losses"].update_parameters()

    outputs.append(action)
    avg_firing_rate = hidden_layer.firing_rate
    return reward, outputs, avg_firing_rate, output_layers["ac_number_predict"].value_output.output()[0], this_state_td_error


@profile
def train(
    teacher_layer, hidden_layer, 
    output_layers, images, 
    labels, 
    epochs=1, 
    batch_size=10, 
    duration=10
):
    """
    Training loop with batch updates.
    """
    for epoch in range(epochs):
        correct = 0
        total_loss = 0
        total_firing_rate = 0
        total_adv = 0
        total_value = 0
        
        for i in tqdm(range(len(images)), total=len(images), desc=f"Epoch {epoch+1}"):
            img = images[i]
            lbl = [labels[i]]*duration
            # Run single image
            loss, outputs, avg_firing_rate, value, adv = run_single_image(
                img, lbl, teacher_layer, hidden_layer, output_layers, duration
            )
            total_adv += adv

            for o in range(len(outputs)):
                pred = np.argmax(outputs[o])
                if pred == lbl[o]:
                    correct += 1/len(outputs)
            
            total_loss += loss
            total_firing_rate += avg_firing_rate
            total_value += value
 
        acc = correct / len(images)
        avg_loss = total_loss / len(images)
        avg_fr = total_firing_rate / len(images)
        avg_adv = total_adv / len(images)
        avg_value = total_value / len(images)
        print(
            f"Epoch {epoch+1} - Acc: {acc:.3f}, "
            f"AvgReward: {avg_loss:.3f}, "
            f"FR: {1000*avg_fr/hidden_layer.num_neurons:.3f} Hz, "
            f"Value Estimate: {avg_value:.2f}, "
            f"Value Error: {avg_adv:.2f}"
        )

def test(teacher_layer, hidden_layer, output_layers, images, labels, duration):
    """
    Test the trained network.
    """
    correct = 0

    for img, lbl in tqdm(zip(images, labels), total=len(images), desc="Testing"):
        output = run_single_image(img, None, teacher_layer, hidden_layer, output_layers, duration)[1]
        pred = np.argmax(output[0])
        if pred == lbl:
            correct += 1
    
    acc = correct / len(images)
    print(f"Test Accuracy: {acc:.3f}")
    return acc

def visualize_mnist(image, label, hidden_layer, output_layers, duration):
    print(label)
    spikes = get_input(image, duration)

    visualizer = SRNNVisualizer(hidden_layer, spikes, output_layers["ac_number_predict"].policy_output)
    visualizer.animate(interval=10)

# Main execution
if __name__ == "__main__":
    # Load data
    print("Loading MNIST dataset...")
    duration = 10
    batch_size = 1
    n_hidden = 128
    images, labels = load_mnist(n_samples=100)
    train_images, train_labels = images[:int(len(images)*0.8)], labels[:int(len(images)*0.8)]
    test_images, test_labels = images[int(len(images)*0.8):], labels[int(len(images)*0.8):]
    print(f"Loaded {len(train_images)} training samples")
    print(f"Label distribution: {np.bincount(train_labels)}")

    # Build network
    print("Building network...")
    teacher_layer, hidden_layer = build_hidden_layers(
        num_hidden=n_hidden, 
        input_connection_density=0.3,
        local_connection_density=0.3
    )
    output_layers = build_output_layers(num_hidden=n_hidden, connection_density=0.3)
    # print(output_layer.weights.nnz)
    # Train network
    print("Training network...")
    train(teacher_layer, hidden_layer, output_layers, train_images, train_labels, 
          epochs=1000, batch_size=batch_size, duration=duration)
    
    # Test on training data (you should use separate test data in practice)
    print("Testing network...")
    test(teacher_layer, hidden_layer, output_layers, test_images, test_labels, duration) 

    # print("Visualizing")
    while input("Visualize?"):
        i = np.random.randint(len(test_images))
        visualize_mnist(
            test_images[i],
            test_labels[i],
            hidden_layer, 
            output_layers, 
            1000
        )
