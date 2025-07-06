#Default Parameters gotten from https://github.com/ChFrenkel/eprop-PyTorch/blob/main/main.py
import numpy as np


def heavy_side_step_function(x: float) -> float:
    return 1 if x >= 0 else 0


class LIF:
    
    def reset(self):
        self.membrane_potential = 0
        self.accumulated_input = 0
        self.sending_pulse = 0
        self.refractory_period = 0

        self.eligibility_vector = np.zeros_like(self.connections_w)
        self.low_pass_eligibility_traces = np.zeros_like(self.connections_w)
        self.el_vec_input = np.zeros_like(self.connections_w)
        self.learning_signal = 0
        self.time_step = 0

    def __init__(
            self,
            connections: dict,
            output_size: int,
            firing_threshold: float = 0.6, 
            learning_rate: float = 0.01,
            pseudo_derivative_slope: float = 0.3,
            connected_to_output: bool = True,
            tau=2000e-3, # 'Membrane potential leakage time constant in the recurrent layer (in seconds)'
            tau_out=20e-2, # 'Membrane potential leakage time constant in the output layer (in seconds)'
            dt=1e-3
    ):
        self.firing_threshold = firing_threshold
        self.alpha    = np.exp(-dt/tau)
        self.kappa    = np.exp(-dt/tau_out)
        self.learning_rate = learning_rate
        self.pseudo_derivative_slope = pseudo_derivative_slope
        self.connected_to_output = connected_to_output
        
        sorted_keys = sorted(connections.keys())
        self.connections_idx = np.array(sorted_keys)
        self.connections_w = np.array([connections[k] for k in sorted_keys])
        self.connections_w_update = np.zeros_like(self.connections_w)

        # self.batch_size = 0

        self.loss_weights = abs(np.random.random(output_size))
        self.other_loss_weights = abs(np.random.random(output_size))
        self.reset()

    def h_pseudo_derivative(self) -> float:
        if self.refractory_period > 0:
            return 0.0
        return (
            1 / self.firing_threshold
            * self.pseudo_derivative_slope
            * max(
                0,
                1
                - abs(
                    (self.membrane_potential - self.firing_threshold)
                    / self.firing_threshold
                ),
            )
        )

    def recieve_pulse(self, idx: int):
        if idx not in self.connections_idx:
            return
        arr_idx = self.connections_idx.searchsorted(idx)
        self.accumulated_input += self.connections_w[arr_idx]
        self.el_vec_input[arr_idx] = 1
    
    def next_time_step(self):
        self.membrane_potential = (
            self.membrane_potential * self.alpha
            + self.accumulated_input
            - self.firing_threshold * self.sending_pulse
        )
        self.sending_pulse = heavy_side_step_function(
            self.membrane_potential - self.firing_threshold
        )
        self.eligibility_vector = (
            self.eligibility_vector * self.alpha 
            + self.el_vec_input
        )
        self.low_pass_eligibility_traces = (
            self.low_pass_eligibility_traces * self.kappa
            + self.eligibility_vector * self.h_pseudo_derivative()
        )
        self.accumulated_input = 0
        self.el_vec_input[:] = 0
        # self.batch_size += 1
        self.time_step += 1
        if self.sending_pulse:
            self.refractory_period = 3
        else:
            self.refractory_period -= 1

    def recieve_error(self, errors):
        self.learning_signal = np.dot(errors, self.loss_weights)
        self.connections_w_update += (
            self.learning_signal
            * self.low_pass_eligibility_traces
        )
    
    def update_parameters(self):
        self.connections_w -= (
            self.learning_rate
            * self.connections_w_update
            # / self.batch_size
        )
        self.connections_w_update[:] = 0
        # self.batch_size = 0
    
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
        # Input and state
        self.input = np.zeros(self.num_outputs)
        self.input_spikes = np.zeros(self.num_hidden)
        self.membrane_potentials = np.zeros(self.num_outputs)
        self.last_spikes = np.zeros(self.num_hidden)

    def __init__(
            self, 
            num_hidden: int, 
            num_outputs: int, 
            learning_rate: float = 0.01,
            tau_out=20e-2, # 'Membrane potential leakage time constant in the output layer (in seconds)'
            dt=1e-3
        ):
        self.num_outputs = num_outputs
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        self.kappa    = np.exp(-dt/tau_out)

        # Weight matrix and bias
        self.weights = np.random.normal(0, 0.1, size=(num_outputs, num_hidden))
        self.bias = np.zeros(num_outputs)

        # Accumulated gradients
        self.accumulated_gradients = np.zeros_like(self.weights)
        self.accumulated_bias_gradients = np.zeros_like(self.bias)
        self.batch_size = 0

        self.reset()

    def receive_pulse(
        self, 
        hidden_idx: int
        # out_idx: int,
        # v: float
    ):
        self.input_spikes[hidden_idx] = 1
        # self.input[out_idx] += v

    def update(self):
        # Leaky integration of input
        self.membrane_potentials = (
            self.kappa * self.membrane_potentials 
            # + self.input
            + self.weights @ self.input_spikes +
            + self.bias
        )

        # Save spikes for gradient use
        self.last_spikes = self.input_spikes.copy()

        # Reset input spikes
        self.input_spikes[:] = 0
        self.input[:] = 0

    def output(self):
        # Softmax activation
        exps = np.exp(self.membrane_potentials - np.max(self.membrane_potentials))
        return exps / np.sum(exps)

    def compute_loss(self, target_class: int):
        probs = self.output()
        return -np.log(probs[target_class] + 1e-9)

    def compute_error(self, target_class: int):
        # Cross-entropy derivative
        probs = self.output()
        probs[target_class] -= 1
        return probs

    def accumulate_gradient(self, error_signal: np.ndarray):
        # Vectorized accumulation
        self.accumulated_gradients += np.outer(error_signal, self.last_spikes)
        self.accumulated_bias_gradients += error_signal
        self.batch_size += 1

    def update_parameters(self):
        # Apply updates
        self.weights -= (
            self.learning_rate 
            * self.accumulated_gradients 
            / self.batch_size
        )
        self.bias -= (
            self.learning_rate 
            * self.accumulated_bias_gradients 
            / self.batch_size
        )
        # Reset accumulators
        self.accumulated_gradients[:] = 0
        self.accumulated_bias_gradients[:] = 0
        self.batch_size = 0

# if __name__ == "__main__":
#     n = LIF(0.5, 0.95)
#     o = LIO(0.05, 0.5)
#     for i in range(100):
#         o.recieve_pulse(0.3)
#         print(o.membrane_potential)
#         o.update()
