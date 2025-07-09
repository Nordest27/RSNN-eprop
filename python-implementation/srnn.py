#Default Parameters gotten from https://github.com/ChFrenkel/eprop-PyTorch/blob/main/main.py
import numpy as np
import scipy.sparse as sp
import multiprocessing as mp

import numpy as np
import scipy.sparse as sp

class ALIFLayer:
    def reset(self):
        self.membrane_potentials = np.zeros(self.num_neurons)
        self.sending_pulses = np.zeros(self.num_neurons, dtype=bool)
        self.refractory_periods = np.zeros(self.num_neurons, dtype=int)
        self.input_spikes = sp.csr_matrix((1, self.num_inputs))
        
        # ALIF-specific: adaptive threshold components
        self.adaptive_thresholds = np.zeros(self.num_neurons)
        
        # Learning-related variables
        self.low_pass_active_connections = sp.csr_matrix((self.num_neurons, self.num_inputs))
        self.adaptative_eligibility_vector = sp.csr_matrix((self.num_neurons, self.num_inputs))
        self.low_pass_eligibility_traces = sp.csr_matrix((self.num_neurons, self.num_inputs))
        self.learning_signals = np.zeros(self.num_neurons)
        self.time_step = 0

    def calculate_weight_mask(self):
        """Same as LIF layer - creates sparse connectivity pattern"""
        external_size_1 = self.recurrent_start - self.just_input_size
        local_size = self.num_neurons
        external_size_2 = self.num_inputs - (self.recurrent_start + self.num_neurons)

        # Input mask
        input_mask = (np.random.rand(self.num_neurons, self.just_input_size) < self.input_connection_density)
        input_mask = sp.csr_matrix(input_mask)

        # External 1 (before recurrent block)
        external_mask_1 = (np.random.rand(self.num_neurons, external_size_1) < self.hidden_connection_density)
        external_mask_1 = sp.csr_matrix(external_mask_1)

        # Local recurrent block
        local_mask = (np.random.rand(self.num_neurons, local_size) < self.local_connection_density)
        local_mask = sp.csr_matrix(local_mask)

        # External 2 (after recurrent block)
        external_mask_2 = (np.random.rand(self.num_neurons, external_size_2) < self.hidden_connection_density)
        external_mask_2 = sp.csr_matrix(external_mask_2)

        # Stack them in the correct order
        return sp.hstack([input_mask, external_mask_1, local_mask, external_mask_2], format='csr')

    def __init__(
        self,
        just_input_size: int,
        num_inputs: int,
        num_neurons: int,
        batch_size: int,
        recurrent_start: int | None = None,
        firing_threshold: float = 0.6,
        learning_rate: float = 0.01,
        pseudo_derivative_slope: float = 0.3,
        input_connection_density: float = 0.1,
        hidden_connection_density: float = 0.1,
        local_connection_density: float = 0.1,
        tau=2000e-3,
        tau_out=20e-2,
        dt=1e-3,
        output_size: int = 1,
        # ALIF-specific parameters
        tau_adaptation=200e-3,  # Adaptation time constant
        beta=0.07,  # Adaptation coupling strength
        # Adam optimizer parameters
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        assert just_input_size + num_neurons <= num_inputs
        self.just_input_size = just_input_size
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.recurrent_start = recurrent_start or (num_inputs - num_neurons)

        self.firing_threshold = firing_threshold
        self.learning_rate = learning_rate
        self.pseudo_derivative_slope = pseudo_derivative_slope
        self.alpha = np.exp(-dt / tau)
        self.kappa = np.exp(-dt / tau_out)
        
        # ALIF-specific parameters
        self.rho = np.exp(-dt / tau_adaptation)  # Adaptation decay factor
        self.beta = beta  # Adaptation coupling strength

        self.input_connection_density = input_connection_density
        self.hidden_connection_density = hidden_connection_density
        self.local_connection_density = local_connection_density
        
        # Adam optimizer parameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Initialize weights
        weight_mask = self.calculate_weight_mask()
        weight_values = np.random.normal(0, 0.1, size=weight_mask.nnz)
        self.weights = sp.csr_matrix((weight_values, weight_mask.indices, weight_mask.indptr),
                                   shape=(num_neurons, self.num_inputs))

        self.loss_weights = abs(sp.random(num_neurons, output_size, density=1.0, format='csr'))
        
        # Initialize accumulated weight updates as sparse matrix
        self.accumulated_weight_updates = sp.csr_matrix((num_neurons, self.num_inputs))
        self.batch_size = batch_size
        
        # Initialize Adam optimizer state
        self.adam_step = 0
        self.m_weights = sp.csr_matrix((num_neurons, self.num_inputs))
        self.v_weights = sp.csr_matrix((num_neurons, self.num_inputs))
        
        self.reset()

    def heavy_side_step_function(self, x: np.ndarray) -> np.ndarray:
        return (x >= 0).astype(float)

    def h_pseudo_derivative(self) -> np.ndarray:
        """Compute pseudo-derivative for ALIF neurons as per e-prop paper"""
        # Set derivative to 0 for neurons in refractory period
        mask = self.refractory_periods <= 0
        
        derivatives = np.zeros(self.num_neurons)
        active_neurons = mask
        
        if np.any(active_neurons):
            # ψ_j^t = (1/v_th) * γ_pd * max(0, 1 - |v_j - A_j^t|/v_th)

            derivatives[active_neurons] = (
                (1 / self.firing_threshold) 
                * self.pseudo_derivative_slope  # γ_pd = 0.3 in paper
                * np.maximum(
                    0,
                    1 - np.abs(
                        (self.membrane_potentials[active_neurons] - self.effective_thresholds[active_neurons]) 
                        / self.firing_threshold
                    )
                )
            )
        
        return derivatives

    def receive_pulse(self, spike_vector):
        """Receive input spikes"""
        self.input_spikes = self.input_spikes.maximum(spike_vector)

    def next_time_step(self):
        """Update all neurons for one time step - ALIF dynamics"""
        # Compute effective threshold (base + adaptive component)
        # A_j^t = v_th + βa_t^j
        self.effective_thresholds = self.firing_threshold + self.beta * self.adaptive_thresholds
        
        # Update membrane potentials
        self.membrane_potentials = np.asarray(
            self.membrane_potentials * self.alpha
            + (self.weights * self.input_spikes.T).T
            - self.effective_thresholds * self.sending_pulses
        ).reshape(-1)

        # Determine which neurons are firing (using effective threshold)
        self.sending_pulses = np.multiply(
            self.heavy_side_step_function(self.membrane_potentials - self.effective_thresholds),
            self.refractory_periods == 0
        )
        
        # Update adaptive thresholds: decay + spike-triggered increase
        self.adaptive_thresholds = (
            self.adaptive_thresholds * self.rho + self.sending_pulses
        )

        # Update eligibility vectors
        pseudo_derivatives = self.h_pseudo_derivative()[:, np.newaxis]
        self.low_pass_active_connections = (
            self.low_pass_active_connections * self.alpha
            + (self.weights != 0).multiply(self.input_spikes)
        )
        self.adaptative_eligibility_vector = (
            self.adaptative_eligibility_vector
            .multiply((self.rho - self.beta * pseudo_derivatives))
            + self.low_pass_active_connections.multiply(pseudo_derivatives)
        )
        # Update low-pass eligibility traces
        self.low_pass_eligibility_traces = (
            self.low_pass_eligibility_traces * self.kappa 
            + (
              self.low_pass_active_connections 
              - self.beta * self.adaptative_eligibility_vector
            )
            .multiply(pseudo_derivatives) 
        )

        # Reset accumulated inputs
        self.input_spikes = sp.csr_matrix((1, self.num_inputs))

        # Update refractory periods
        self.refractory_periods = np.where(
            self.sending_pulses, 
            3,  # Set refractory period to 3 for firing neurons
            np.maximum(0, self.refractory_periods - 1)  # Decrement for others
        )

        self.time_step += 1

    def receive_error(self, errors):
        """Receive error signals and compute learning signals"""
        # Compute learning signal for each neuron
        self.learning_signals = self.loss_weights @ errors

        # Compute updates only for existing weights
        rows, cols = self.weights.nonzero()
        weight_update_values = (
            self.learning_signals[rows] * 
            self.low_pass_eligibility_traces[rows, cols].A1
        )
        weight_updates_sparse = sp.csr_matrix(
            (weight_update_values, (rows, cols)), 
            shape=(self.num_neurons, self.num_inputs)
        )
        self.accumulated_weight_updates += weight_updates_sparse

    def update_parameters(self):
        """Apply accumulated weight updates using Adam optimizer"""
        if self.accumulated_weight_updates.nnz == 0:
            return
            
        # Scale gradients by batch size
        gradients = self.accumulated_weight_updates / self.batch_size
        
        # Increment Adam step counter
        self.adam_step += 1
        
        # Update biased first moment estimate (momentum)
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * gradients
        
        # Update biased second moment estimate (RMSprop)
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * gradients.multiply(gradients)
        
        # Bias correction
        m_hat = self.m_weights / (1 - self.beta1 ** self.adam_step)
        v_hat = self.v_weights / (1 - self.beta2 ** self.adam_step)
        
        # Convert v_hat to dense for sqrt operation, then back to sparse
        v_hat_dense = v_hat.toarray()
        v_hat_sqrt = np.sqrt(v_hat_dense) + self.epsilon
        
        # Compute Adam update
        m_hat_dense = m_hat.toarray()
        update_dense = self.learning_rate * m_hat_dense / v_hat_sqrt
        update_sparse = sp.csr_matrix(update_dense)
        
        # Only update weights that exist (multiply by weight mask)
        masked_update = update_sparse.multiply(self.weights != 0)
        
        # Update weights
        self.weights = self.weights - masked_update
        
        # Reset accumulated updates
        self.accumulated_weight_updates = sp.csr_matrix((self.num_neurons, self.num_inputs))

    def get_output_spikes(self):
        """Return current spike outputs as sparse matrix"""
        if np.any(self.sending_pulses):
            spike_indices = np.where(self.sending_pulses)[0]
            spike_data = np.ones(len(spike_indices))
            return sp.csr_matrix((spike_data, (np.zeros(len(spike_indices)), spike_indices)), 
                               shape=(1, self.num_neurons))
        else:
            return sp.csr_matrix((1, self.num_neurons))

    def get_output_spikes_dense(self):
        """Return current spike outputs as dense array"""
        return self.sending_pulses.astype(float)
    
    def get_adaptive_thresholds(self):
        """Return current adaptive threshold values"""
        return self.adaptive_thresholds.copy()
    
    def get_effective_thresholds(self):
        """Return current effective thresholds (base + adaptive)"""
        return self.firing_threshold + self.beta * self.adaptive_thresholds
    
    def get_adam_stats(self):
        """Return current Adam optimizer statistics for debugging"""
        return {
            'step': self.adam_step,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'lr': self.learning_rate,
            'm_weights_nnz': self.m_weights.nnz,
            'v_weights_nnz': self.v_weights.nnz
        }
    
    def get_alif_stats(self):
        """Return ALIF-specific statistics"""
        return {
            'rho': self.rho,
            'beta': self.beta,
            'mean_adaptive_threshold': np.mean(self.adaptive_thresholds),
            'max_adaptive_threshold': np.max(self.adaptive_thresholds),
            'mean_effective_threshold': np.mean(self.get_effective_thresholds()),
            'active_neurons': np.sum(self.adaptive_thresholds > 0.01)
        }

class SoftmaxOutputLayer:
    def reset(self):
        self.input_spikes = sp.csr_matrix((1, self.num_hidden))
        self.membrane_potentials = np.zeros(self.num_outputs)
        self.last_spikes = np.zeros(self.num_hidden)

    def __init__(
        self, 
        num_hidden: int, 
        num_outputs: int,
        input_offset: int = 0,
        learning_rate: float = 0.01,
        connection_density: float = 0.05,  # Fraction of connections present
        tau_out=20e-2,
        dt=1e-3
    ):
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.kappa = np.exp(-dt / tau_out)

        # Sparse initialization with input_offset: zero out columns < input_offset
        weight_mask = sp.random(num_outputs, num_hidden, density=connection_density, format='csr')
        if input_offset > 0:
            # Zero out all columns < input_offset
            mask = weight_mask.toarray()
            mask[:, :input_offset] = 0
            weight_mask = sp.csr_matrix(mask)

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
        self.input_spikes = self.input_spikes.maximum(sparse_spike_vector)

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


class SimpleBroadcastSrnn:
    def __init__(
        self, 
        num_neurons_list, 
        input_size, output_size, 
        input_connectivity=0.1,
        hidden_connectivity=0.01,
        local_connectivity=0.1,
        output_connectivity=0.03
    ):
        self.num_layers = len(num_neurons_list)
        self.num_neurons_list = num_neurons_list
        self.input_size = input_size
        self.output_size = output_size

        self.input_queues = [mp.Queue() for _ in range(self.num_layers)]
        self.output_queues = [mp.Queue() for _ in range(self.num_layers)]
        self.processes = []

        # Assign layer IDs and neuron index ranges
        self.layer_ids = list(range(self.num_layers))
        self.layers_offsets = []
        offset = input_size
        for n in num_neurons_list:
            self.layers_offsets.append(offset)
            offset += n
        self.total_neurons = offset - input_size
        self.global_size = input_size + self.total_neurons
        
        # Default LIFLayer parameters
        default_lif_kwargs = dict(
            just_input_size=input_size,
            num_inputs=self.global_size,
            firing_threshold=0.6,
            learning_rate=0.001,
            input_connection_density=input_connectivity,
            hidden_connection_density=hidden_connectivity,
            local_connection_density=local_connectivity,
            batch_size=1,
            output_size=output_size
        )

        for i in range(self.num_layers):
            lif_kwargs = dict(default_lif_kwargs)
            lif_kwargs['num_inputs'] = input_size + self.total_neurons
            lif_kwargs['num_neurons'] = num_neurons_list[i]
            p = mp.Process(
                target=self.lif_worker,
                args=(
                    i, 
                    lif_kwargs, 
                    self.input_queues[i], 
                    self.output_queues[i], 
                    self.layers_offsets[i], 
                    self.global_size
                )
            )
            p.start()
            self.processes.append(p)

        num_hidden = sum(num_neurons_list)
        self.output_layer = SoftmaxOutputLayer(
            num_hidden=input_size + num_hidden,
            num_outputs=output_size,
            learning_rate=0.001,
            connection_density=output_connectivity,
            # input_offset=self.layers_offsets[-1]
        )

    @staticmethod
    def lif_worker(
            layer_id, 
            config, 
            input_queue, 
            output_queue, 
            layer_offset, 
            global_size
        ):
        layer = ALIFLayer(**config, recurrent_start=layer_offset, beta=abs(np.random.random()))
        while True:
            instruction, data = input_queue.get()
            match instruction:
                case "STOP":
                    break
                case "PULSE":
                    # print("PULSE")
                    layer.receive_pulse(data)  # expects csr_matrix
                case "NEXT STEP":
                    layer.next_time_step()
                    # Get local spikes
                    local_spikes = layer.get_output_spikes()
                    output_spikes = sp.hstack([
                        sp.csr_matrix((1, layer_offset)),
                        local_spikes,
                        sp.csr_matrix((1, global_size - layer_offset - layer.num_neurons))
                    ])
                    output_queue.put((layer_id, output_spikes))
                case "FEEDBACK":
                    # print("FEEDBACK")
                    layer.receive_error(data)
                case "UPDATE":
                    # print("UPDATE")
                    layer.update_parameters()
                case "OUTWEIGHTS":
                    # print("OUTWEIGHTS")
                    layer.loss_weights[data.indices] = data[data.indices]
                case "RESET":
                    # print("RESET")
                    layer.reset()
                case "N_CONNECTIONS":
                    # Return the number of nonzero weights
                    output_queue.put(("N_CONNECTIONS", layer.weights.nnz))
            # time.sleep(0.001)

    def input(self, input_data):
        # input_data: csr_matrix of shape (1, input_size + total_neurons)
        for q in self.input_queues:
            q.put(("PULSE", input_data))
            q.put(("NEXT STEP", None))
        # Wait for any layer to finish
        # ready = [q for q in self.output_queues if not q.empty()]
        # total_output_spikes = sp.csr_matrix((1, self.global_size))
        for q in self.output_queues:
            # idx = self.output_queues.index(ready.pop())
            # layer_id, output_spikes = self.output_queues[idx].get()
            output_spikes = q.get()[1]
            # total_output_spikes = total_output_spikes.maximum(output_spikes)
            # Broadcast this output to all layers for the next step  
            for q in self.input_queues:
                # print("Sending Pulse")
                q.put(("PULSE", output_spikes))
            self.output_layer.receive_pulse(output_spikes)

        # self.output_layer.receive_pulse(input_data)
        self.output_layer.update()
        # input("input")
        return self.output_layer.output()
    
    def feedback(self, label):
        errors = self.output_layer.compute_error(label)
        loss = self.output_layer.compute_loss(label)
        for q in self.input_queues:
            q.put(("FEEDBACK", errors))
        self.output_layer.accumulate_gradient(errors)
        return loss
    
    def update_parameters(self):
        for q in self.input_queues:
            q.put(("UPDATE", None))
        self.output_layer.update_parameters()
    
    def update_outweights(self):
        for i, q in enumerate(self.input_queues):
            start = self.layers_offsets[i]
            end = start + self.num_neurons_list[i]
            out_w = self.output_layer.weights[:, start:end].T  # shape: (layer_size, num_outputs)
            q.put(("OUTWEIGHTS", out_w))
    
    def reset(self):
        for q in self.input_queues:
            q.put(("RESET", None))
        self.output_layer.reset()
    
    def shutdown(self):
    #     for q in self.input_queues:
    #         q.put(("STOP", None))
        for p in self.processes:
            p.terminate()
        for q in self.input_queues:
            q.cancel_join_thread()
        for q in self.output_queues:
            q.cancel_join_thread()

    def get_n_connections(self):
        """Query all LIF layers and the output layer for their number of connections."""
        n_connections = {}
        total = 0
        # Ask each LIF layer process
        for i, q in enumerate(self.input_queues):
            q.put(("N_CONNECTIONS", None))
        for i, q in enumerate(self.output_queues):
            msg_type, n = q.get()
            assert msg_type == "N_CONNECTIONS"
            n_connections[f'layer_{i}'] = n
            total += n
        # Output layer
        n_out = self.output_layer.weights.nnz
        n_connections['output_layer'] = n_out
        total += n_out
        n_connections['total'] = total
        return n_connections
        
