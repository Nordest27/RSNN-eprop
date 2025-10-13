import numpy as np
import scipy.sparse as sp
from srnn import AdamOptimizer


class ActorCriticMLPOutputLayer:
    """
    Actor-Critic output layer that aggregates spikes, passes them through
    dense ReLU layers, and outputs policy/value estimates.
    Uses deferred parameter updates for consistency with other layers.
    """

    def __init__(
        self,
        num_hidden: int,
        num_outputs: int,
        hidden_layers: list[int] = [128],
        learning_rate: float = 1e-3,
        tau_out: float = 20e-3,
        dt: float = 1e-3,
        policy_activation_function: str = "softmax",
        gamma: float = 0.99,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
    ):
        assert policy_activation_function in ["softmax", "linear"]

        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.policy_activation_function = policy_activation_function

        # Temporal filtering (low-pass of incoming spikes)
        self.kappa = np.exp(-dt / tau_out)
        self.low_pass_input = np.zeros(num_hidden)

        # Build MLP architecture
        layer_sizes = [num_hidden] + hidden_layers + [
            2 * num_outputs if policy_activation_function == "linear" else num_outputs
        ]
        self.weights = []
        self.biases = []
        self.optimizers = []
        self.accumulated_dW = []
        self.accumulated_db = []

        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i + 1], layer_sizes[i]) / np.sqrt(layer_sizes[i])
            b = np.zeros(layer_sizes[i + 1])
            self.weights.append(w)
            self.biases.append(b)
            self.optimizers.append(AdamOptimizer(w.shape, adam_beta1, adam_beta2, adam_epsilon))
            self.accumulated_dW.append(np.zeros_like(w))
            self.accumulated_db.append(np.zeros_like(b))

        # Value head
        self.value_w = np.random.randn(1, layer_sizes[-2]) / np.sqrt(layer_sizes[-2])
        self.value_b = np.zeros(1)
        self.value_opt = AdamOptimizer(self.value_w.shape, adam_beta1, adam_beta2, adam_epsilon)
        self.accumulated_value_dW = np.zeros_like(self.value_w)
        self.accumulated_value_db = np.zeros_like(self.value_b)

        # RL state
        self.previous_value_estimate = 0.0
        self.reward_trace = 0.0
        self.policy_grad = None

        # Forward cache
        self.activations = []
        self.z_values = []

        self.reset()

    def reset(self):
        self.low_pass_input *= 0
        self.previous_value_estimate = 0
        self.reward_trace = 0
        for i in range(len(self.accumulated_dW)):
            self.accumulated_dW[i].fill(0)
            self.accumulated_db[i].fill(0)
        self.accumulated_value_dW.fill(0)
        self.accumulated_value_db.fill(0)

    def receive_pulse(self, sparse_spike_vector: sp.csr_array):
        """Receive spikes (sparse) and update filtered trace."""
        spike_dense = np.zeros(self.num_hidden)
        spike_dense[sparse_spike_vector.indices] = 1.0
        self.low_pass_input = self.kappa * self.low_pass_input + spike_dense
    
    def next_time_step():
        pass

    def _forward(self):
        """Forward through all MLP layers."""
        a = self.low_pass_input
        self.activations = [a]
        self.z_values = []
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            z = W @ a + b
            a = np.maximum(0, z)
            self.z_values.append(z)
            self.activations.append(a)
        # Last (linear) layer
        z = self.weights[-1] @ a + self.biases[-1]
        self.z_values.append(z)
        self.activations.append(z)
        return z

    def _policy_output(self):
        """Compute policy output (softmax or Gaussian parameters)."""
        z = self._forward()
        if self.policy_activation_function == "softmax":
            exps = np.exp(z - np.max(z))
            return exps / np.sum(exps)
        else:
            means = z[:self.num_outputs]
            log_stds = z[self.num_outputs:]
            return means, log_stds

    def _value_output(self):
        """Compute scalar value from last hidden layer."""
        h = self.activations[-2]
        return (self.value_w @ h + self.value_b)[0]

    def action(self):
        """Sample action and compute log-prob gradients."""
        if self.policy_activation_function == "softmax":
            probs = self._policy_output()
            action_idx = np.random.choice(len(probs), p=probs)
            action_onehot = np.zeros_like(probs)
            action_onehot[action_idx] = 1.0
            self.policy_grad = probs - action_onehot
            return action_onehot
        else:
            means, log_stds = self._policy_output()
            stds = np.exp(log_stds)
            action = np.random.normal(means, stds)
            self.policy_grad = np.concatenate([
                -(action - means) / (stds**2),
                1 - ((action - means)**2 / (stds**2)),
            ])
            return action

    def _backward(self, grad_output: np.ndarray):
        """Backpropagate through MLP."""
        grads = []
        da = grad_output
        for i in reversed(range(len(self.weights))):
            a_prev = self.activations[i]
            z = self.z_values[i]
            dz = da * (z > 0 if i < len(self.weights) - 1 else 1)
            dW = np.outer(dz, a_prev)
            db = dz
            da = self.weights[i].T @ dz
            grads.append((dW, db))
        grads.reverse()
        return grads

    def td_error_update(self, reward: float):
        """Compute TD error and accumulate parameter gradients."""
        value_est = self._value_output()
        td_error = reward + self.gamma * value_est - self.previous_value_estimate
        self.previous_value_estimate = value_est

        # === accumulate gradients for policy ===
        grads = self._backward(self.policy_grad * td_error)
        for i, (dW, db) in enumerate(grads):
            self.accumulated_dW[i] += dW
            self.accumulated_db[i] += db

        # === accumulate gradients for value ===
        h = self.activations[-2]
        self.accumulated_value_dW += td_error * h
        self.accumulated_value_db += td_error

        return td_error

    def update_parameters(self):
        """Apply accumulated updates using Adam optimizers."""
        for i, (W, b, opt, dW, db) in enumerate(
            zip(self.weights, self.biases, self.optimizers, self.accumulated_dW, self.accumulated_db)
        ):
            update = opt.get_update(sp.csr_array(dW))
            W -= self.learning_rate * update.toarray()
            b -= self.learning_rate * db
            dW.fill(0)
            db.fill(0)

        # Value head update
        val_update = self.value_opt.get_update(sp.csr_array(self.accumulated_value_dW))
        self.value_w -= self.learning_rate * val_update.toarray()
        self.value_b -= self.learning_rate * self.accumulated_value_db
        self.accumulated_value_dW.fill(0)
        self.accumulated_value_db.fill(0)
