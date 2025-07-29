import numpy as np
import scipy.sparse as sp
import time

class ALIFLayer:
    
    def initialize_beta_distribution(self, beta, num_neurons, beta_hyperparams=None):
        """
        Initialize beta parameter with various distributions
        
        Args:
            beta: str or float specifying distribution type
            num_neurons: number of neurons in the layer
            beta_hyperparams: dict of hyperparameters for each distribution
        
        Returns:
            numpy array of beta values
        """
        
        # Default hyperparameters for each distribution
        default_hyperparams = {
            "randomize": {
                "max_beta": 0.14
            },
            "sparse_adaptive": {
                "lif_fraction": 0.8,  # Fraction of LIF neurons
                "exp_scale": 0.05,    # Scale parameter for exponential distribution
                "max_beta": 0.2       # Maximum beta value
            },
            "bimodal": {
                "adaptive_fraction": 0.3,  # Fraction of adaptive neurons
                "adaptive_mean": 0.12,     # Mean beta for adaptive population
                "adaptive_std": 0.02,      # Std dev for adaptive population
                "max_beta": 0.2            # Maximum beta value
            },
            "power_law": {
                "exponent": 0.5,      # Power law exponent (lower = more skewed)
                "max_beta": 0.15      # Maximum beta value
            },
            "log_normal": {
                "mean": -3.0,         # Mean of underlying normal distribution
                "sigma": 0.8,         # Standard deviation of underlying normal
                "max_beta": 0.2       # Maximum beta value
            },
            "structured": {
                "transition_point": 0.8,  # Where sigmoid transition occurs (fraction of neurons)
                "steepness": 0.01,        # Steepness of sigmoid transition
                "max_beta": 0.15          # Maximum beta value
            },
            "clusters": {
                "n_clusters": 4,                        # Number of clusters
                "cluster_betas": [0.0, 0.05, 0.1, 0.15],  # Beta values for each cluster
                "noise_std": 0.01,                      # Standard deviation of noise around cluster centers
                "max_beta": 0.2                         # Maximum beta value
            },
            "pareto": {
                "shape": 1.5,         # Pareto shape parameter (lower = more skewed)
                "scale": 0.02,        # Scale factor
                "max_beta": 0.2       # Maximum beta value
            }
        }
        
        if isinstance(beta, float):
            return np.array([beta] * num_neurons)
        
        # Merge provided hyperparameters with defaults
        if beta_hyperparams is None:
            beta_hyperparams = {}
        params = default_hyperparams.get(beta, {}).copy()
        params.update(beta_hyperparams)
        
        if isinstance(beta, str):
            np.random.seed()  # Ensure different random seeds across processes
            
            if beta == "randomize":
                # Original uniform distribution
                return params["max_beta"] * np.random.rand(num_neurons)
            
            elif beta == "sparse_adaptive":
                # Heavy bias toward LIF neurons (beta=0) with few adaptive neurons
                betas = np.zeros(num_neurons)
                n_adaptive = int((1 - params["lif_fraction"]) * num_neurons)
                adaptive_indices = np.random.choice(num_neurons, n_adaptive, replace=False)
                # Exponential distribution for adaptive neurons
                betas[adaptive_indices] = np.random.exponential(params["exp_scale"], n_adaptive)
                return np.clip(betas, 0, params["max_beta"])
            
            elif beta == "bimodal":
                # Two populations: LIF (beta≈0) and strongly adaptive (beta≈0.1-0.15)
                betas = np.zeros(num_neurons)
                n_adaptive = int(params["adaptive_fraction"] * num_neurons)
                adaptive_indices = np.random.choice(num_neurons, n_adaptive, replace=False)
                # Adaptive neurons have beta around specified mean
                betas[adaptive_indices] = np.random.normal(
                    params["adaptive_mean"], 
                    params["adaptive_std"], 
                    n_adaptive
                )
                return np.clip(betas, 0, params["max_beta"])
            
            elif beta == "power_law":
                # Power law distribution - many low values, few high values
                betas = np.random.power(params["exponent"], num_neurons) * params["max_beta"]
                return betas
            
            elif beta == "log_normal":
                # Log-normal distribution - naturally skewed toward low values
                betas = np.random.lognormal(mean=params["mean"], sigma=params["sigma"], size=num_neurons)
                return np.clip(betas, 0, params["max_beta"])
            
            elif beta == "structured":
                # Structured assignment based on neuron index
                indices = np.arange(num_neurons)
                # Sigmoid-like transition
                sigmoid_vals = 1 / (1 + np.exp(-params["steepness"] * (indices - params["transition_point"] * num_neurons)))
                betas = sigmoid_vals * params["max_beta"]
                return betas
            
            elif beta == "clusters":
                # Create distinct clusters of neurons with different adaptation levels
                n_clusters = params["n_clusters"]
                cluster_betas = params["cluster_betas"]
                
                # Ensure we have enough cluster beta values
                if len(cluster_betas) < n_clusters:
                    # Extend with linearly spaced values
                    missing = n_clusters - len(cluster_betas)
                    max_existing = max(cluster_betas)
                    additional = np.linspace(max_existing + 0.02, params["max_beta"], missing)
                    cluster_betas = cluster_betas + additional.tolist()
                
                cluster_size = num_neurons // n_clusters
                betas = np.zeros(num_neurons)
                
                for i in range(n_clusters):
                    start_idx = i * cluster_size
                    end_idx = (i + 1) * cluster_size if i < n_clusters - 1 else num_neurons
                    # Add some noise around cluster centers
                    cluster_beta = cluster_betas[i]
                    noise = np.random.normal(0, params["noise_std"], end_idx - start_idx)
                    betas[start_idx:end_idx] = cluster_beta + noise
                
                return np.clip(betas, 0, params["max_beta"])
            
            elif beta == "pareto":
                # Pareto distribution - classic "80-20" rule
                betas = (np.random.pareto(params["shape"], num_neurons) + 1) * params["scale"]
                return np.clip(betas, 0, params["max_beta"])
            
            else:
                raise ValueError(f"Unknown beta distribution: {beta}")
        
        else:
            raise ValueError("Invalid beta parameter type")

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

        if self.self_predict_layer is not None:
            self.self_predict_layer.reset()
        
        self.last_time = time.time()
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
        self_predict: bool = False,
        # ALIF-specific parameters
        tau_adaptation=200e-3,  # Adaptation time constant
        beta: float | str = 0.07,  # Adaptation coupling strength
        beta_params: dict | None = None, 
        # Adam optimizer parameters
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        assert just_input_size + num_neurons <= num_inputs

        self.dt = dt

        self.just_input_size = just_input_size
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.recurrent_start = recurrent_start or (num_inputs - num_neurons)

        self.tau = tau
        self.tau_out = tau_out
        self.tau_adaptation = tau_adaptation
        
        self.alpha = np.exp(-dt / self.tau)
        self.kappa = np.exp(-dt / self.tau_out)
        self.rho = np.exp(-dt / self.tau_adaptation)

        self.firing_threshold = firing_threshold
        self.learning_rate = learning_rate
        self.pseudo_derivative_slope = pseudo_derivative_slope

        self.betas = self.initialize_beta_distribution(beta, num_neurons, beta_params)
        self.betas = self.betas.reshape(1, -1)
        #self.beta = beta

        self.input_connection_density = input_connection_density
        self.hidden_connection_density = hidden_connection_density
        self.local_connection_density = local_connection_density
        
        # Adam optimizer parameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Initialize weights
        weight_mask = self.calculate_weight_mask()
        weight_values = np.random.randn(weight_mask.nnz) / np.sqrt(num_inputs)
        #weight_values = get_init_fluct_weights(num_neurons, firing_threshold, tau, weight_mask.nnz)
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

        self.self_predict_layer = None
        if self_predict:
            self.self_predict_layer = OutputLayer(
                num_hidden=num_inputs, 
                num_outputs=num_neurons//1,
                input_offset=just_input_size,
                dt=dt,
                tau_out=tau_out,
                connection_density=local_connection_density,
                activation_function="sigmoid"
            )
            self.self_predict_loss_weights = abs(sp.random(num_neurons, num_neurons//1, density=1.0, format='csr'))

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

        self.effective_thresholds = (
            self.firing_threshold 
            + np.ndarray.flatten(self.betas * self.adaptive_thresholds)
        )
        
        # Update membrane potentials
        self.membrane_potentials = np.asarray(
            self.membrane_potentials * self.alpha
            + (self.weights * self.input_spikes.T).T
            - self.effective_thresholds * self.sending_pulses
        ).reshape(-1)

        # Determine which neurons are firing (using effective threshold)
        self.sending_pulses = np.multiply(
            self.heavy_side_step_function(
                self.membrane_potentials - self.effective_thresholds
            ),
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
            .multiply(( self.rho - (self.betas * pseudo_derivatives.reshape(-1)).T ))
            + self.low_pass_active_connections.multiply(pseudo_derivatives)
        )

        # Update low-pass eligibility traces
        self.low_pass_eligibility_traces = (
            self.low_pass_eligibility_traces * self.kappa 
            + (
              self.low_pass_active_connections 
              - self.adaptative_eligibility_vector.multiply(self.betas.reshape(-1, 1))
            )
            .multiply(pseudo_derivatives) 
        )
        
        if self.self_predict_layer is not None:
            self.self_predict_layer.receive_pulse(
                self.input_spikes
                # [:, self.recurrent_start:self.recurrent_start + self.num_neurons]
            )
            if self.time_step % 1 == 0:
                error = (
                    self.self_predict_layer.compute_error(
                        self.sending_pulses[:self.num_neurons//1]
                    ) / self.num_neurons
                )
                predict_output = self.self_predict_layer.output()
                entropy_grad = -np.log(predict_output + 1e-9) - 1
                lambda_entropy = 0.1  # Tune this value as needed
                error += lambda_entropy * entropy_grad
                self.self_predict_layer.accumulate_gradient(error)

        # Reset accumulated inputs
        self.input_spikes = sp.csr_matrix((1, self.num_inputs))

        # Update refractory periods
        self.refractory_periods = np.where(
            self.sending_pulses, 
            3,
            np.maximum(0, self.refractory_periods - 1)  # Decrement for others
        )

        self.time_step += 1

    def receive_error(self, errors, self_predict=False):
        """Receive error signals and compute learning signals"""
        # Compute learning signal for each neuron
        if not self_predict:
            self.learning_signals = self.loss_weights @ errors
        else:
            self.learning_signals = self.self_predict_loss_weights @ errors

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
        if self.self_predict_layer is not None:
            self.self_predict_layer.update_parameters()
            start = self.recurrent_start
            end = start + self.num_neurons
            out_w = self.self_predict_layer.weights[:, start:end].T
            self.self_predict_loss_weights[out_w.indices] = out_w[out_w.indices]
            
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
        
        l2_lambda = 1e-3
        l2_penalty = l2_lambda * self.weights

        # Apply L2 penalty during weight updates
        self.weights = self.weights - masked_update - l2_penalty
        
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
        return self.firing_threshold + self.adaptive_thresholds.multiply(self.betas)
    
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
            'betas': self.betas,
            'mean_adaptive_threshold': np.mean(self.adaptive_thresholds),
            'max_adaptive_threshold': np.max(self.adaptive_thresholds),
            'mean_effective_threshold': np.mean(self.get_effective_thresholds()),
            'active_neurons': np.sum(self.adaptive_thresholds > 0.01)
        }

    # Usage examples and statistics
    def analyze_beta_distribution(self):
        """Analyze the properties of a beta distribution"""
        n_lif = np.sum(self.betas < 0.01)  # Nearly LIF neurons
        n_adaptive = np.sum(self.betas >= 0.01)  # Adaptive neurons
        
        print("\nDistribution Analysis:")
        print(f"  Total neurons: {self.num_neurons}")
        print(f"  LIF neurons (β < 0.01): {n_lif} ({(n_lif/self.num_neurons)*100:.1f}%)")
        print(f"  Adaptive neurons (β ≥ 0.01): {n_adaptive} ({(n_adaptive/self.num_neurons)*100:.1f}%)")
        print(f"  Mean β: {np.mean(self.betas):.4f}")
        print(f"  Std β: {np.std(self.betas):.4f}")
        print(f"  Max β: {np.max(self.betas):.4f}")
        print(f"  Min β: {np.min(self.betas):.4f}")


class OutputLayer:
    def reset(self):
        self.input_spikes = sp.csr_matrix((1, self.num_hidden))
        self.membrane_potentials = np.zeros(self.num_outputs)
        self.last_spikes = np.zeros(self.num_hidden)
        self.last_time = time.time()

    def __init__(
        self, 
        num_hidden: int, 
        num_outputs: int,
        input_offset: int = 0,
        learning_rate: float = 0.01,
        connection_density: float = 0.05,  # Fraction of connections present
        tau_out=20e-2,
        dt=1e-3,
        activation_function: str = 'softmax',  # 'linear', 'softmax'
        unary_weights: bool = False
    ):
        self.unary_weights = unary_weights
        self.activation_function = activation_function
        assert self.activation_function in ["softmax", "linear", "sigmoid"]

        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate

        self.tau_out = tau_out
        self.kappa = np.exp(-dt / self.tau_out)
        print(self.kappa)
        # Sparse initialization with input_offset: zero out columns < input_offset
        weight_mask = sp.random(num_outputs, num_hidden, density=connection_density, format='csr')
        if input_offset > 0:
            # Zero out all columns < input_offset
            mask = weight_mask.toarray()
            mask[:, :input_offset] = 0
            weight_mask = sp.csr_matrix(mask)

        if self.unary_weights:
            weight_values = np.ones(weight_mask.nnz)/weight_mask.nnz
        else:
            weight_values = np.random.randn(weight_mask.nnz) / np.sqrt(num_hidden)

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

    def next_time_step(self):
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
        if self.activation_function == "softmax":
            exps = np.exp(self.membrane_potentials - np.max(self.membrane_potentials))
            return exps / np.sum(exps)
        elif self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-self.membrane_potentials))
        else:
            return self.membrane_potentials

    def compute_loss(self, target: int | np.ndarray):
        output = self.output()
        if self.activation_function == "softmax":
            return -np.log(output[target] + 1e-9)
        else:
            return 0.5 * np.sum((output - target) ** 2)

    def compute_error(self, target: int | np.ndarray):
        output = self.output()
        if self.activation_function == "softmax":
            output[target] -= 1
            return output
        else:
            return output - target

    def accumulate_gradient(self, error_signal: np.ndarray):
        """
        Only accumulate gradients for non-zero weights and active spikes.
        """
        if self.unary_weights:
            return
        # Outer product: error_signal (num_outputs) x last_spikes (num_hidden)
        grad_dense = np.outer(error_signal, self.last_spikes)

        # Mask to only keep gradients for existing weights
        grad_sparse = sp.csr_matrix(grad_dense)  # full sparse gradient

        # Only accumulate updates on the current sparse structure of weights
        self.accumulated_gradients += grad_sparse.multiply(self.weights != 0)
        self.accumulated_bias_gradients += error_signal
        self.batch_size += 1

    def update_parameters(self):
        if self.unary_weights or self.batch_size == 0:
            return

        lr_scaled = self.learning_rate / self.batch_size

        # Update only the non-zero weights
        self.weights = self.weights - self.accumulated_gradients.multiply(lr_scaled)

        self.bias -= lr_scaled * self.accumulated_bias_gradients

        # Reset
        self.accumulated_gradients = sp.csr_matrix(self.weights.shape)
        self.accumulated_bias_gradients[:] = 0
        self.batch_size = 0
