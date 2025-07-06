#Default Parameters gotten from https://github.com/ChFrenkel/eprop-PyTorch/blob/main/main.py
import numpy as np
import scipy.sparse as sp

import numpy as np
import scipy.sparse as sp

class LIFLayer:
    def reset(self):
        self.membrane_potentials = np.zeros(self.num_neurons)
        self.accumulated_inputs = np.zeros(self.num_neurons)
        self.sending_pulses = np.zeros(self.num_neurons, dtype=bool)
        self.refractory_periods = np.zeros(self.num_neurons, dtype=int)
        
        self.eligibility_vectors = np.zeros((self.num_neurons, self.num_inputs))
        self.low_pass_eligibility_traces = np.zeros((self.num_neurons, self.num_inputs))
        self.el_vec_inputs = np.zeros((self.num_neurons, self.num_inputs))
        self.learning_signals = np.zeros(self.num_neurons)
        self.time_step = 0
        
        # Reset weight updates
        self.weight_updates = sp.csr_matrix((self.num_neurons, self.num_inputs))
        # self.batch_size += 1

    def __init__(
        self,
        num_inputs: int,
        num_neurons: int,
        firing_threshold: float = 0.6,
        learning_rate: float = 0.01,
        pseudo_derivative_slope: float = 0.3,
        connection_density: float = 0.1,
        tau=2000e-3,
        tau_out=20e-2,
        dt=1e-3,
        output_size: int = 1
    ):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.firing_threshold = firing_threshold
        self.learning_rate = learning_rate
        self.pseudo_derivative_slope = pseudo_derivative_slope
        self.alpha = np.exp(-dt / tau)
        self.kappa = np.exp(-dt / tau_out)
        
        # Initialize sparse weight matrix
        weight_mask = sp.random(num_neurons, num_inputs, density=connection_density, format='csr')
        weight_values = np.random.normal(0, 0.1, size=weight_mask.nnz)
        self.weights = sp.csr_matrix((weight_values, weight_mask.indices, weight_mask.indptr),
                                   shape=(num_neurons, num_inputs))
        
        self.loss_weights = sp.random(num_neurons, output_size, density=0.1, format='csr')
        # Initialize accumulated weight updates as sparse matrix
        self.accumulated_weight_updates = sp.csr_matrix((num_neurons, num_inputs))
        # self.batch_size = -1
        
        self.reset()

    def heavy_side_step_function(self, x: np.ndarray) -> np.ndarray:
        return (x >= 0).astype(float)

    def h_pseudo_derivative(self) -> np.ndarray:
        """Compute pseudo-derivative for all neurons"""
        # Set derivative to 0 for neurons in refractory period
        mask = self.refractory_periods <= 0
        
        derivatives = np.zeros(self.num_neurons)
        active_neurons = mask
        
        if np.any(active_neurons):
            derivatives[active_neurons] = (
                1 / self.firing_threshold
                * self.pseudo_derivative_slope
                * np.maximum(
                    0,
                    1 - np.abs(
                        (self.membrane_potentials[active_neurons] 
                        - self.firing_threshold) 
                        / self.firing_threshold
                    )
                )
            )
        
        return derivatives

    def receive_pulse(self, spike_vector: np.ndarray):
        """
        Accept a sparse (or dense) binary vector of incoming spikes
        spike_vector shape: [num_inputs]
        """
        # Ensure spike_vector is a row vector
        if spike_vector.shape[0] != 1:
            spike_vector = spike_vector.reshape(1, -1)
        # Matrix multiplication: weights @ spike_vector.T
        input_effects = self.weights @ spike_vector.T
        input_effects = np.asarray(input_effects.todense()).reshape(-1) 
        # Get active indices
        active_indices = spike_vector.nonzero()[1]
        
        self.accumulated_inputs += input_effects
        self.el_vec_inputs[:, active_indices] = 1

    def next_time_step(self):
        """Update all neurons for one time step"""
        # Update membrane potentials
        self.membrane_potentials = (
            self.membrane_potentials * self.alpha
            + self.accumulated_inputs
            - self.firing_threshold * self.sending_pulses
        )
        
        # Determine which neurons are firing
        self.sending_pulses = self.heavy_side_step_function(
            self.membrane_potentials - self.firing_threshold
        )
        
        # Update eligibility vectors
        self.eligibility_vectors = (
            self.eligibility_vectors * self.alpha
            + self.el_vec_inputs
        )
        
        # Update low-pass eligibility traces
        pseudo_derivatives = self.h_pseudo_derivative()
        self.low_pass_eligibility_traces = (
            self.low_pass_eligibility_traces * self.kappa
            + self.eligibility_vectors * pseudo_derivatives[:, np.newaxis]
        )
        
        # Reset accumulated inputs and eligibility inputs
        self.accumulated_inputs[:] = 0
        self.el_vec_inputs[:] = 0
        
        # Update refractory periods
        self.refractory_periods = np.where(
            self.sending_pulses, 
            3,  # Set refractory period to 3 for firing neurons
            np.maximum(0, self.refractory_periods - 1)  # Decrement for others
        )
        
        self.time_step += 1

    def receive_error(self, errors: np.ndarray):
        """
        Receive error signals and compute learning signals
        errors shape: [output_size]
        """
        # Compute learning signal for each neuron
        self.learning_signals = self.loss_weights @ errors
        
        # Accumulate weight updates using eligibility traces
        weight_updates_dense = self.learning_signals[:, np.newaxis] * self.low_pass_eligibility_traces
        
        # Convert to sparse and accumulate
        weight_updates_sparse = sp.csr_matrix(weight_updates_dense)
        # Only update weights that exist (multiply by weight mask)
        self.accumulated_weight_updates += weight_updates_sparse.multiply(self.weights != 0)

    def update_parameters(self):
        """Apply accumulated weight updates"""
        # if self.batch_size == 0:
        #     return
            
        lr_scaled = self.learning_rate 
        # / self.batch_size

        
        # Update weights
        self.weights = self.weights - self.accumulated_weight_updates.multiply(lr_scaled)
        
        # Reset accumulated updates
        self.accumulated_weight_updates = sp.csr_matrix((self.num_neurons, self.num_inputs))
        # self.batch_size = 0

    def get_output_spikes(self):
        """Return current spike outputs as sparse matrix"""
        if np.any(self.sending_pulses):
            # Create sparse matrix with spikes
            spike_indices = np.where(self.sending_pulses)[0]
            spike_data = np.ones(len(spike_indices))
            return sp.csr_matrix((spike_data, (np.zeros(len(spike_indices)), spike_indices)), 
                               shape=(1, self.num_neurons))
        else:
            return sp.csr_matrix((1, self.num_neurons))

    def get_output_spikes_dense(self):
        """Return current spike outputs as dense array"""
        return self.sending_pulses.astype(float)
    
# class ReadoutNeuron:
#     def __init__(self, output_bias: float, leak_alpha: float):
#         self.leak_alpha = leak_alpha
#         self.output_bias = output_bias
#         self.membrane_potential = 0
#         self.accumulated_input = 0

#     def receive_pulse(self, pulse_weight: float):
#         self.accumulated_input += pulse_weight

#     def update(self):
#         self.membrane_potential = (
#             self.membrane_potential * self.leak_alpha
#             + self.accumulated_input
#             + self.output_bias
#         )
#         self.accumulated_input = 0

#     def output(self):
#         return self.membrane_potential


# class SoftmaxOutputLayer:

#     def reset(self):
#         self.membrane_potentials = np.zeros(self.num_outputs)
#         self.accumulated_input = np.zeros(self.num_outputs)

#     def __init__(self, num_outputs: int, leak_alpha: float = 1.0, output_bias: float = 0.0):
#         self.num_outputs = num_outputs
#         self.leak_alpha = leak_alpha
#         self.output_bias = output_bias
#         self.reset()

#     def receive_pulse(self, neuron_idx: int, pulse_weight: float):
#         self.accumulated_input[neuron_idx] += pulse_weight

#     def update(self):
#         self.membrane_potentials = (
#             self.membrane_potentials * self.leak_alpha
#             + self.accumulated_input
#             + self.output_bias
#         )
#         self.accumulated_input = np.zeros_like(self.accumulated_input)

#     def output(self):
#         # Softmax activation
#         exps = np.exp(self.membrane_potentials - np.max(self.membrane_potentials))
#         return exps / np.sum(exps)

#     def compute_loss(self, target_class: int):
#         probs = self.output()
#         # Cross-entropy loss
#         return -np.log(probs[target_class] + 1e-9)  # add epsilon for numerical stability

#     def compute_error(self, target_class: int):
#         # Gradient of cross-entropy loss w.r.t. pre-softmax potentials
#         probs = self.output()
#         probs[target_class] -= 1
#         return probs



class SoftmaxOutputLayer:
    def reset(self):
        self.input_spikes = sp.csr_matrix((1, self.num_hidden))
        self.membrane_potentials = np.zeros(self.num_outputs)
        self.last_spikes = np.zeros(self.num_hidden)

    def __init__(
        self, 
        num_hidden: int, 
        num_outputs: int, 
        learning_rate: float = 0.01,
        connection_density: float = 0.05,  # Fraction of connections present
        tau_out=20e-2,
        dt=1e-3
    ):
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.kappa = np.exp(-dt / tau_out)

        # Sparse initialization
        weight_mask = sp.random(num_outputs, num_hidden, density=connection_density, format='csr')
        weight_values = np.random.normal(0, 0.1, size=weight_mask.nnz)
        self.weights = sp.csr_matrix((weight_values, weight_mask.indices, weight_mask.indptr),
                                     shape=(num_outputs, num_hidden))

        self.bias = np.zeros(num_outputs)

        # Accumulate gradients using dense matrix for simplicity
        self.accumulated_gradients = sp.csr_matrix((num_outputs, num_hidden))
        self.accumulated_bias_gradients = np.zeros_like(self.bias)
        self.batch_size = 0

        self.reset()

    def receive_pulse(self, sparse_spike_vector: sp.csr_matrix):
        assert sparse_spike_vector.shape == (1, self.num_hidden)
        self.input_spikes += sparse_spike_vector

    def update(self):
        # Convert to dense for single timestep usage
        self.last_spikes = np.zeros(self.num_hidden)
        if self.input_spikes.nnz > 0:
            self.last_spikes[self.input_spikes.indices] = 1

        # Sparse matrix-vector multiplication
        self.membrane_potentials = (
            self.kappa * self.membrane_potentials
            + self.weights @ self.last_spikes
            + self.bias
        )

        # Reset input
        self.input_spikes = sp.csr_matrix((1, self.num_hidden))

    def output(self):
        exps = np.exp(self.membrane_potentials - np.max(self.membrane_potentials))
        return exps / np.sum(exps)

    def compute_loss(self, target_class: int):
        probs = self.output()
        return -np.log(probs[target_class] + 1e-9)

    def compute_error(self, target_class: int):
        probs = self.output()
        probs[target_class] -= 1
        return probs

    def accumulate_gradient(self, error_signal: np.ndarray):
        """
        Only accumulate gradients for non-zero weights and active spikes.
        """
        # Outer product: error_signal (num_outputs) x last_spikes (num_hidden)
        grad_dense = np.outer(error_signal, self.last_spikes)

        # Mask to only keep gradients for existing weights
        grad_sparse = sp.csr_matrix(grad_dense)  # full sparse gradient

        # Only accumulate updates on the current sparse structure of weights
        self.accumulated_gradients += grad_sparse.multiply(self.weights != 0)
        self.accumulated_bias_gradients += error_signal
        self.batch_size += 1

    def update_parameters(self):
        if self.batch_size == 0:
            return

        lr_scaled = self.learning_rate / self.batch_size

        # Update only the non-zero weights
        self.weights = self.weights - self.accumulated_gradients.multiply(lr_scaled)

        self.bias -= lr_scaled * self.accumulated_bias_gradients

        # Reset
        self.accumulated_gradients = sp.csr_matrix(self.weights.shape)
        self.accumulated_bias_gradients[:] = 0
        self.batch_size = 0

# if __name__ == "__main__":
#     n = LIF(0.5, 0.95)
#     o = LIO(0.05, 0.5)
#     for i in range(100):
#         o.recieve_pulse(0.3)
#         print(o.membrane_potential)
#         o.update()
