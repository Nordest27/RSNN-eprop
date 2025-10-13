from re import S
import numpy as np
import scipy.sparse as sp
import time
from line_profiler import profile


class AdamOptimizer:
    def __init__(
        self,
        shape: tuple,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.step = 0
        self.m_weights = sp.csr_array(shape)
        self.v_weights = sp.csr_array(shape)
    
    def get_update(self, gradients):
        # Increment Adam step counter
        self.step += 1

        # Update biased first moment estimate (momentum)
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * gradients

        # Update biased second moment estimate (RMSprop)
        self.v_weights = self.beta2 * self.v_weights + (
            1 - self.beta2
        ) * gradients.multiply(gradients)

        # Bias correction
        m_hat = self.m_weights / (1 - self.beta1**self.step)
        v_hat = self.v_weights / (1 - self.beta2**self.step)

        s_v_hat_sqrt = np.sqrt(v_hat)
        s_v_hat_sqrt.data += self.epsilon

        # Compute Adam update
        s_v_hat_sqrt.data = 1.0/s_v_hat_sqrt.data

        return m_hat.multiply(s_v_hat_sqrt)


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
            "randomize": {"max_beta": 0.14},
            "sparse_adaptive": {
                "lif_fraction": 0.8,  # Fraction of LIF neurons
                "exp_scale": 0.05,  # Scale parameter for exponential distribution
                "max_beta": 0.2,  # Maximum beta value
            },
            "bimodal": {
                "adaptive_fraction": 0.3,  # Fraction of adaptive neurons
                "adaptive_mean": 0.12,  # Mean beta for adaptive population
                "adaptive_std": 0.02,  # Std dev for adaptive population
                "max_beta": 0.2,  # Maximum beta value
            },
            "power_law": {
                "exponent": 0.5,  # Power law exponent (lower = more skewed)
                "max_beta": 0.15,  # Maximum beta value
            },
            "log_normal": {
                "mean": -3.0,  # Mean of underlying normal distribution
                "sigma": 0.8,  # Standard deviation of underlying normal
                "max_beta": 0.2,  # Maximum beta value
            },
            "structured": {
                "transition_point": 0.8,  # Where sigmoid transition occurs (fraction of neurons)
                "steepness": 0.01,  # Steepness of sigmoid transition
                "max_beta": 0.15,  # Maximum beta value
            },
            "clusters": {
                "n_clusters": 4,  # Number of clusters
                "cluster_betas": [0.0, 0.05, 0.1, 0.15],  # Beta values for each cluster
                "noise_std": 0.01,  # Standard deviation of noise around cluster centers
                "max_beta": 0.2,  # Maximum beta value
            },
            "pareto": {
                "shape": 1.5,  # Pareto shape parameter (lower = more skewed)
                "scale": 0.02,  # Scale factor
                "max_beta": 0.2,  # Maximum beta value
            },
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
                adaptive_indices = np.random.choice(
                    num_neurons, n_adaptive, replace=False
                )
                # Exponential distribution for adaptive neurons
                betas[adaptive_indices] = np.random.exponential(
                    params["exp_scale"], n_adaptive
                )
                return np.clip(betas, 0, params["max_beta"])

            elif beta == "bimodal":
                # Two populations: LIF (beta≈0) and strongly adaptive (beta≈0.1-0.15)
                betas = np.zeros(num_neurons)
                n_adaptive = int(params["adaptive_fraction"] * num_neurons)
                adaptive_indices = np.random.choice(
                    num_neurons, n_adaptive, replace=False
                )
                # Adaptive neurons have beta around specified mean
                betas[adaptive_indices] = np.random.normal(
                    params["adaptive_mean"], params["adaptive_std"], n_adaptive
                )
                return np.clip(betas, 0, params["max_beta"])

            elif beta == "power_law":
                # Power law distribution - many low values, few high values
                betas = (
                    np.random.power(params["exponent"], num_neurons)
                    * params["max_beta"]
                )
                return betas

            elif beta == "log_normal":
                # Log-normal distribution - naturally skewed toward low values
                betas = np.random.lognormal(
                    mean=params["mean"], sigma=params["sigma"], size=num_neurons
                )
                return np.clip(betas, 0, params["max_beta"])

            elif beta == "structured":
                # Structured assignment based on neuron index
                indices = np.arange(num_neurons)
                # Sigmoid-like transition
                sigmoid_vals = 1 / (
                    1
                    + np.exp(
                        -params["steepness"]
                        * (indices - params["transition_point"] * num_neurons)
                    )
                )
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
                    additional = np.linspace(
                        max_existing + 0.02, params["max_beta"], missing
                    )
                    cluster_betas = cluster_betas + additional.tolist()

                cluster_size = num_neurons // n_clusters
                betas = np.zeros(num_neurons)

                for i in range(n_clusters):
                    start_idx = i * cluster_size
                    end_idx = (
                        (i + 1) * cluster_size if i < n_clusters - 1 else num_neurons
                    )
                    # Add some noise around cluster centers
                    cluster_beta = cluster_betas[i]
                    noise = np.random.normal(
                        0, params["noise_std"], end_idx - start_idx
                    )
                    betas[start_idx:end_idx] = cluster_beta + noise

                return np.clip(betas, 0, params["max_beta"])

            elif beta == "pareto":
                # Pareto distribution - classic "80-20" rule
                betas = (np.random.pareto(params["shape"], num_neurons) + 1) * params[
                    "scale"
                ]
                return np.clip(betas, 0, params["max_beta"])

            else:
                raise ValueError(f"Unknown beta distribution: {beta}")

        else:
            raise ValueError("Invalid beta parameter type")

    def reset(self):
        self.membrane_potentials = np.zeros(self.num_neurons)
        self.sending_pulses = np.zeros(self.num_neurons, dtype=bool)
        self.refractory_periods = np.zeros(self.num_neurons, dtype=int)
        self.input_spikes = np.zeros(self.num_inputs, dtype=bool)

        self.firing_rate = 0
        # if self.firing_rate > self.target_firing_rate:
            # self.weights *= .25
        
        # self.weights = self.initial_weights #+ self.weights*0.01

        # ALIF-specific: adaptive threshold components
        self.adaptive_thresholds = np.zeros(self.num_neurons)

        # Learning-related variables
        self.low_pass_input = self.input_spikes * 0
        self.low_pass_active_connections = sp.csr_array(
            (self.num_neurons, self.num_inputs)
        )
        self.adaptative_eligibility_vector = sp.csr_array(
            (self.num_neurons, self.num_inputs)
        )
        self.previous_low_pass_eligibility_traces = sp.csr_array(
            (self.num_neurons, self.num_inputs)
        )
        self.low_pass_eligibility_traces = sp.csr_array(
            (self.num_neurons, self.num_inputs)
        )
        # self.low_pass_rl_etrace = sp.csr_array(
        #     (self.num_neurons, self.num_inputs)
        # )
        self.input_spikes_record = []
        self.refractory_periods_record = []
        self.membrane_potentials_record = []
        self.effective_thresholds_record = []
        self.firing_rates_record = []
        self.base_errors = []
        self.time_steps = []
        self.rl_signals_record = {}
        # self.temp_deltas = []
        # self.rl_learning_signals = []
        # self.low_pass_rl_etraces_record = []

        # self.reward = 0
        # self.reward_trace = 0
        
        # self.previous_value_estimation = 0
        # self.value_output.reset()
        # self.policy_output.reset()
        # self.policy_grads = None

        """
        if self.self_predict_layer is not None:
            self.self_predict_layer.reset()
        """
        
        self.last_time = time.time()
        self.time_step = 1
        self.previous_rl_time_step = 1
    
    def full_reset(self):
        self.weights = sp.csr_array(
            (self.initial_weight_values, self.weight_mask.indices, self.weight_mask.indptr),
            shape=(self.num_neurons, self.num_inputs),
        )
        # self.learned_losses_weights = sp.csr_array(
        #     (self.learned_losses_initial_weight_values, self.learned_losses_weight_mask.indices, self.learned_losses_weight_mask.indptr),
        #     shape=(self.num_neurons, self.num_inputs),
        # )

        self.base_weights_update = self.weights * 0
        # self.learned_losses_weights_update = self.weights * 0
        self.base_adam = AdamOptimizer(self.weights.shape, self.adam_beta1, self.adam_beta2, self.adam_epsilon)
        # self.learned_losses_adam = AdamOptimizer(self.weights.shape, self.adam_beta1, self.adam_beta2, self.adam_epsilon)

        self.error_counter = 0
        self.firing_rate_error = 0

        self.reset()

    def calculate_weight_mask(
        self,
        input_connection_density: float | None = None
    ):
        """Same as LIF layer - creates sparse connectivity pattern"""
        if input_connection_density is None:
            input_connection_density = self.input_connection_density
        external_size_1 = self.recurrent_start - self.just_input_size
        local_size = self.num_neurons
        external_size_2 = self.num_inputs - (self.recurrent_start + self.num_neurons)

        # Input mask
        input_mask = (
            np.random.rand(self.num_neurons, self.just_input_size)
            < input_connection_density
        )
        input_mask = sp.csr_array(input_mask)

        # External 1 (before recurrent block)
        external_mask_1 = (
            np.random.rand(self.num_neurons, external_size_1)
            < self.hidden_connection_density
        )
        external_mask_1 = sp.csr_array(external_mask_1)

        # Local recurrent block
        local_mask = (
            np.random.rand(self.num_neurons, local_size) < self.local_connection_density
        )
        for i in range(0, self.num_neurons):
            local_mask[i][i] = False
        local_mask = sp.csr_array(local_mask)

        # External 2 (after recurrent block)
        external_mask_2 = (
            np.random.rand(self.num_neurons, external_size_2)
            < self.hidden_connection_density
        )
        external_mask_2 = sp.csr_array(external_mask_2)

        # Stack them in the correct order
        return sp.hstack(
            [input_mask, external_mask_1, local_mask, external_mask_2], format="csr"
        )

    def __init__(
        self,
        just_input_size: int,
        num_inputs: int,
        num_neurons: int,
        output_sizes: dict[str, int],
        recurrent_start: int | None = None,
        firing_threshold: float = 1.0,
        target_firing_rate: float = 13,
        learning_rate: float = 0.001,
        learned_loss_learning_rate: float = 1e-3,
        pseudo_derivative_slope: float = 0.3,
        input_connection_density: float = 0.1,
        hidden_connection_density: float = 0.1,
        local_connection_density: float = 0.1,
        tau=200e-3,
        tau_out=20e-3,
        dt=1e-3,
        self_predict: bool = False,
        # ALIF-specific parameters
        tau_adaptation=2000e-3,  # Adaptation time constant
        beta: float | str = 0.07,  # Adaptation coupling strength
        beta_params: dict | None = None,
        # Adam optimizer parameters
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
    ):
        assert just_input_size + num_neurons <= num_inputs

        self.dt = dt

        self.just_input_size = just_input_size
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.output_sizes = output_sizes

        self.recurrent_start = recurrent_start or (num_inputs - num_neurons)

        self.tau = tau
        self.tau_out = tau_out
        self.tau_adaptation = tau_adaptation

        self.alpha = np.exp(-dt / self.tau)
        self.kappa = np.exp(-dt / self.tau_out)
        self.rho = np.exp(-dt / self.tau_adaptation)

        self.firing_threshold = firing_threshold
        self.target_firing_rate = target_firing_rate / 1000
        self.learning_rate = learning_rate
        self.learned_loss_learning_rate = learned_loss_learning_rate
        self.dampening_factor = pseudo_derivative_slope

        self.betas = self.initialize_beta_distribution(beta, num_neurons, beta_params)
        self.betas_col = self.betas.reshape(-1, 1)
        # self.beta = beta

        self.input_connection_density = input_connection_density
        self.hidden_connection_density = hidden_connection_density
        self.local_connection_density = local_connection_density

        # Initialize weights
        self.weight_mask = self.calculate_weight_mask()
        self.initial_weight_values = np.random.randn(self.weight_mask.nnz) / np.sqrt(num_inputs)
        self.initial_weights = sp.csr_array(
            (self.initial_weight_values, self.weight_mask.indices, self.weight_mask.indptr),
            shape=(self.num_neurons, self.num_inputs),
        )
        # self.learned_losses_weight_mask = self.calculate_weight_mask()#input_connection_density=0.0)
        # self.learned_losses_initial_weight_values = np.random.randn(self.learned_losses_weight_mask.nnz) / np.sqrt(num_inputs)

        self.loss_weights_dict = {
            output_name: sp.random(num_neurons, output_size, density=1.0, format="csr")
            for output_name, output_size in self.output_sizes.items()
        }

        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

        self.full_reset()

    @profile
    def receive_pulse(self, spike_vector):
        """Receive input spikes"""
        self.input_spikes[spike_vector.indices] = True

    def receive_error(self, error):
        # Compute learning signal for each neuron
        self.error_counter += 1
        self.base_errors[-1] = error

    @profile
    def next_time_step(self):
        """Update all neurons for one time step - ALIF dynamics"""
        # Compute effective threshold (base + adaptive component)
        # A_j^t = v_th + βa_t^j
        # self.receive_loss_signal(self.learned_loss_learning_rate * self.learned_losses_weights @ self.input_spikes)
        
        self.base_errors.append(None)
        self.effective_thresholds = self.firing_threshold + np.multiply(self.betas, self.adaptive_thresholds)
        
        self.membrane_potentials = (
            self.membrane_potentials * self.alpha
            + self.weights @ self.input_spikes
            - self.effective_thresholds * self.sending_pulses
        )
        # Determine which neurons are firing (using effective threshold)
        self.sending_pulses = np.where(
            self.refractory_periods == 0,
            self.membrane_potentials - self.effective_thresholds > 0,
            0,
        )
        self.firing_rate = (
            self.firing_rate * (self.time_step - 1) / self.time_step
            + sum(self.sending_pulses) / self.time_step
        )

        # Update adaptive thresholds: decay + spike-triggered increase
        self.adaptive_thresholds = (
            self.adaptive_thresholds * self.rho + self.sending_pulses
        )
        
        self.input_spikes_record.append(self.input_spikes.copy())
        self.refractory_periods_record.append(self.refractory_periods.copy())
        self.membrane_potentials_record.append(self.membrane_potentials.copy())
        self.effective_thresholds_record.append(self.effective_thresholds.copy())
        self.firing_rates_record.append(self.firing_rate)
        self.time_steps.append(self.time_step)

        # Reset accumulated inputs
        self.input_spikes[:] = False

        # Update refractory periods
        self.refractory_periods = np.where(
            self.sending_pulses,
            3,
            np.maximum(0, self.refractory_periods - 1),  # Decrement for others
        )
        self.time_step += 1
    

    def no_grad_predict(self, input_sequence):
        no_grad_membrane_potentials = self.membrane_potentials * 0
        no_grad_refractory_periods = self.refractory_periods * 0
        no_grad_sending_pulses = self.sending_pulses * 0
        no_grad_adaptive_thresholds = self.adaptive_thresholds * 0

        output = []

        for ts_v in input_sequence:
            no_grad_input_spikes = sp.hstack([
                ts_v,
                no_grad_sending_pulses.reshape(1, -1)
            ]).todense().reshape(-1)

            no_grad_effective_thresholds = (
                self.firing_threshold + 
                np.multiply(self.betas, no_grad_adaptive_thresholds)
            )
            no_grad_membrane_potentials = (
                no_grad_membrane_potentials * self.alpha
                + self.weights @ no_grad_input_spikes
                - no_grad_effective_thresholds * no_grad_sending_pulses
            )

            no_grad_sending_pulses = np.where(
                no_grad_refractory_periods == 0,
                no_grad_membrane_potentials - no_grad_effective_thresholds > 0,
                0,
            )
            no_grad_adaptive_thresholds = (
                no_grad_adaptive_thresholds * self.rho + no_grad_sending_pulses
            )
            no_grad_refractory_periods = np.where(
                no_grad_sending_pulses,
                3,
                np.maximum(0, no_grad_refractory_periods - 1),  # Decrement for others
            )

            output.append(no_grad_sending_pulses.copy())
        
        return output


    def h_pseudo_derivative(
        self,
        refractory_periods, 
        membrane_potentials, 
        effective_thresholds
    ) -> np.ndarray:
        """Compute pseudo-derivative for ALIF neurons as per e-prop paper"""
        # ψ_j^t = (1/v_th) * γ_pd * max(0, 1 - |v_j - A_j^t|/v_th)
        # Set derivative to 0 for neurons in refractory period
        active_neurons = refractory_periods <= 0

        mp_scaled = (
            membrane_potentials[active_neurons]
            - effective_thresholds[active_neurons]
        ) / self.firing_threshold
        derivatives = np.zeros(self.num_neurons)
        derivatives[active_neurons] = (
            (self.dampening_factor / self.firing_threshold)
            * np.maximum(0.0, 1. - np.abs(mp_scaled))
        )

        return derivatives
    
    def compute_rl_learning_signal(
        self, signal_time_step, 
        low_pass_eligibility_traces,
        previous_low_pass_eligibility_traces,
        policy_loss_weights_name="policy",
        value_loss_weights_name="value",

    ):
        if signal_time_step in self.rl_signals_record:
            (
                policy_gradient, this_state_td_error, 
                previous_policy_grads, previous_state_td_error
            ) = self.rl_signals_record[signal_time_step]

            rl_learning_signal = - this_state_td_error * (
                (self.loss_weights_dict[policy_loss_weights_name] @ policy_gradient)
                - self.loss_weights_dict[value_loss_weights_name].todense().reshape(-1)
            )
            rl_previous_learning_signal = - previous_state_td_error * (
                (self.loss_weights_dict[policy_loss_weights_name] @ previous_policy_grads)
                - self.loss_weights_dict[value_loss_weights_name].todense().reshape(-1)
            )
            # self.learned_losses_weights_update += -self.learned_loss_learning_rate * (
            #     (np.sum(rl_learning_signal.reshape(-1, 1) 
            #     * self.low_pass_eligibility_traces
            #     * prev_low_pass_e_trace, axis=1) *
            #     self.learned_losses_active_connections.T)
            # ).T

            self.base_weights_update += (
                rl_learning_signal.reshape(-1, 1) 
                * low_pass_eligibility_traces
            ) + (
                rl_previous_learning_signal.reshape(-1, 1)
                * previous_low_pass_eligibility_traces
            )
    
    @profile
    def compute_elegibility_traces(self, base_loss_weights_name: str):
        # Update eligibility vectors
        pseudo_derivatives = []
        voltage_errors = []
        for (rf, mp, et) in zip(
            self.refractory_periods_record, 
            self.membrane_potentials_record, 
            self.effective_thresholds_record
        ):
            pseudo_derivatives.append(
                self.h_pseudo_derivative(
                    rf, mp, et
            ).reshape(-1, 1))
            
            # voltage regularization
            voltage_errors.append(
                np.maximum(mp - et, 0) #a+ np.maximum(-mp - et, 0)
            )
        for (
                t, ip, pd, err, fr, ve
            ) in zip(
            self.time_steps,
            self.input_spikes_record,
            pseudo_derivatives,
            self.base_errors,
            self.firing_rates_record,
            voltage_errors
        ):
            self.low_pass_active_connections = (
                self.low_pass_active_connections * self.alpha
                + (self.weights * ip) != 0
            )
            # self.learned_losses_active_connections = (self.learned_losses_weights * ip) != 0

            self.adaptative_eligibility_vector = (
                self.adaptative_eligibility_vector * (
                    self.rho - self.betas_col * pd
                )
                + self.low_pass_active_connections * pd
            )

            elegibility_vector = (
                self.low_pass_active_connections
                - self.adaptative_eligibility_vector * self.betas_col
            )
            
            self.previous_low_pass_eligibility_traces = self.low_pass_eligibility_traces.copy()
            self.low_pass_eligibility_traces = (
                self.low_pass_eligibility_traces * self.kappa + elegibility_vector * pd
            )

            self.firing_rate_error = (
                fr / self.num_neurons - self.target_firing_rate
            )

            learning_signal = self.firing_rate_error
            if err is not None:
                learning_signal = self.loss_weights_dict[base_loss_weights_name] @ err

            # self.learned_losses_weights_update += -self.learned_loss_learning_rate * (
            #     (np.sum(learning_signal.reshape(-1, 1) 
            #     * self.low_pass_eligibility_traces
            #     * prev_low_pass_e_trace, axis=1) *
            #     self.learned_losses_active_connections.T)
            # ).T
            
            self.base_weights_update += (
                # self.firing_rate_error * self.low_pass_eligibility_traces
                learning_signal.reshape(-1, 1) * self.low_pass_eligibility_traces
                # Voltage Regularization
                + 0.1 * ve.reshape(-1, 1) * elegibility_vector
            )
            self.compute_rl_learning_signal(
                t, self.low_pass_eligibility_traces,
                self.previous_low_pass_eligibility_traces
            )

            # print(sum(abs(ve)))
                
        self.input_spikes_record = []
        self.refractory_periods_record = []
        self.membrane_potentials_record = [] 
        self.effective_thresholds_record = []
        self.firing_rates_record = []
        self.base_errors = []
        self.rl_learning_signals = []
        self.learned_loss_signals_record = []
        self.time_steps = []
        self.rl_signals_record = {}

    @profile
    def update_parameters(self, base_loss_weights_name="base"):
        """Apply accumulated weight updates using Adam optimizer"""
        self.compute_elegibility_traces(base_loss_weights_name)

        update = self.base_adam.get_update(self.base_weights_update)
        masked_update = update.multiply(self.weights != 0)
        self.weights = self.weights - self.learning_rate * masked_update

        # update = self.learned_losses_adam.get_update(self.learned_losses_weights_update)
        # masked_update = update.multiply(self.learned_losses_weights != 0)
        # self.learned_losses_weights = self.learned_losses_weights - self.learning_rate * masked_update

        self.base_weights_update *= 0
        # self.learned_losses_weights_update *= 0

    @profile
    def receive_rl_gradient(
        self, 
        signal_time_step,
        policy_gradient, this_state_td_error, 
        previous_policy_grads, previous_state_td_error
    ):
        self.rl_signals_record[signal_time_step] = (
            policy_gradient, this_state_td_error, 
            previous_policy_grads, previous_state_td_error
        )

        return this_state_td_error

    # def receive_loss_signal(
    #     self, loss_signal, 
    #     base_loss_weights_name: str = "base"
    # ):
    #     self.compute_elegibility_traces(base_loss_weights_name)
    #     arbitrary_update = (
    #         loss_signal.reshape(-1, 1) * self.low_pass_eligibility_traces
    #     )
        
    #     masked_update = arbitrary_update.multiply(self.weights != 0)
    #     # prev_weights = self.weights
    #     self.weights = self.weights - masked_update
    #     # print(sum(sum(abs(prev_weights - self.weights))))

    def get_output_spikes(self):
        """Return current spike outputs as sparse matrix"""
        if np.any(self.sending_pulses):
            spike_indices = np.where(self.sending_pulses)[0]
            spike_data = np.ones(len(spike_indices))
            return sp.csr_array(
                (spike_data, (np.zeros(len(spike_indices)), spike_indices)),
                shape=(1, self.num_neurons),
            )
        else:
            return sp.csr_array((1, self.num_neurons))

    def get_output_spikes_dense(self):
        """Return current spike outputs as dense array"""
        return self.sending_pulses.astype(float)

    def get_adaptive_thresholds(self):
        """Return current adaptive threshold values"""
        return self.adaptive_thresholds.copy()

    def get_effective_thresholds(self):
        """Return current effective thresholds (base + adaptive)"""
        return self.firing_threshold + self.adaptive_thresholds.multiply(self.betas)

    def get_alif_stats(self):
        """Return ALIF-specific statistics"""
        return {
            "betas": self.betas,
            "mean_adaptive_threshold": np.mean(self.adaptive_thresholds),
            "max_adaptive_threshold": np.max(self.adaptive_thresholds),
            "mean_effective_threshold": np.mean(self.get_effective_thresholds()),
            "active_neurons": np.sum(self.adaptive_thresholds > 0.01),
        }

    # Usage examples and statistics
    def analyze_beta_distribution(self):
        """Analyze the properties of a beta distribution"""
        n_lif = np.sum(self.betas < 0.01)  # Nearly LIF neurons
        n_adaptive = np.sum(self.betas >= 0.01)  # Adaptive neurons

        print("\nDistribution Analysis:")
        print(f"  Total neurons: {self.num_neurons}")
        print(
            f"  LIF neurons (β < 0.01): {n_lif} ({(n_lif / self.num_neurons) * 100:.1f}%)"
        )
        print(
            f"  Adaptive neurons (β ≥ 0.01): {n_adaptive} ({(n_adaptive / self.num_neurons) * 100:.1f}%)"
        )
        print(f"  Mean β: {np.mean(self.betas):.4f}")
        print(f"  Std β: {np.std(self.betas):.4f}")
        print(f"  Max β: {np.max(self.betas):.4f}")
        print(f"  Min β: {np.min(self.betas):.4f}")


class OutputLayer:

    def reset(self):
        self.input_spikes = np.zeros(self.num_hidden)
        self.values = np.zeros(self.num_outputs)
        self.low_pass_input = np.zeros(self.num_hidden)
        self.previous_low_pass_input = np.zeros(self.num_hidden)
        self.last_time = time.time()

    def __init__(
        self,
        num_hidden: int,
        num_outputs: int,
        input_offset: int = 0,
        learning_rate: float = 0.01,
        connection_density: float = 0.05,  # Fraction of connections present
        tau_out=20e-3,
        dt=1e-3,
        activation_function: str = "softmax",  # 'linear', 'softmax'
        rl_gamma: float | None = None,
        is_policy: bool | None = None,
        # Adam optimizer parameters
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        use_bias: bool = True
    ):
        self.rl_gamma = rl_gamma
        self.use_bias = use_bias 
        assert self.rl_gamma is not None or is_policy is None
        self.is_policy = is_policy
        
        self.activation_function = activation_function
        assert self.activation_function in ["softmax", "linear", "sigmoid"]

        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate

        self.tau_out = tau_out
        self.kappa = np.exp(-dt / self.tau_out)
        print(f"KAPPA: {self.kappa}")
        # Sparse initialization with input_offset: zero out columns < input_offset
        weights_mask = sp.random(
            num_outputs, num_hidden, density=connection_density, format="csr"
        )
        if input_offset > 0:
            # Zero out all columns < input_offset
            mask = weights_mask.toarray()
            mask[:, :input_offset] = 0
            weights_mask = sp.csr_array(mask)

        weight_values = np.random.randn(weights_mask.nnz) / np.sqrt(num_hidden)
        self.weights = sp.csr_array(
            (weight_values, weights_mask.indices, weights_mask.indptr),
            shape=(num_outputs, num_hidden),
        )
        self.non_zero_weights = self.weights != 0

        self.adam_optimizer = AdamOptimizer(
            self.weights.shape, adam_beta1, adam_beta2, adam_epsilon
        )
        
        self.bias = np.zeros(num_outputs)

        self.accumulated_weights_update = self.weights * 0
        self.accumulated_bias_update = self.bias * 0

        self.errors_received = 0

        self.reset()

    def receive_pulse(self, sparse_spike_vector: sp.csr_array):
        self.input_spikes[sparse_spike_vector.indices] = True

    @profile
    def next_time_step(self):
        # Convert to dense for single timestep usage
        self.previous_low_pass_input = self.low_pass_input.copy()
        self.low_pass_input = (
            self.kappa * self.low_pass_input
            + self.input_spikes
        )
        # Sparse matrix-vector multiplication
        self.values = (
            self.kappa * self.values
            + self.weights @ self.input_spikes
        )
        self.input_spikes *= 0

    def no_grad_predict(self, input_sequence, target=None):
        no_grad_values = self.values * 0

        for ts_v in input_sequence:
            no_grad_values = (
                self.kappa * no_grad_values
                + self.weights @ ts_v
            )

        return (
            self.output(no_grad_values),
            self.compute_error(target, no_grad_values) if target is not None else None,
            self.compute_loss(target, no_grad_values) if target is not None else None
        )

    @profile
    def output(self, values=None):
        assert values is None
        values = values if values is not None else self.values
        biased_values = values + self.bias * (self.use_bias)
        if self.activation_function == "softmax":
            exps = np.exp(biased_values - np.max(biased_values))
            return exps / np.sum(exps)
        elif self.activation_function == "sigmoid":
            return 1 / (1 + np.exp(-biased_values))
        else:
            return biased_values

    def compute_loss(self, target: int | np.ndarray, values=None):
        output = self.output(values)
        if self.activation_function == "softmax":
            return -np.log(output[target] + 1e-9)
        else:
            return 0.5 * np.sum((output - target) ** 2)

    def compute_error(self, target: int | np.ndarray, values=None):
        # print(values)
        output = self.output(values)
        if self.activation_function == "softmax":
            output[target] -= 1
            return output
        else:
            return output - target
    
    @profile
    def receive_error(self, error_signal: np.ndarray, use_previous_state: bool = False):
        self.errors_received += 1
        # Outer product: error_signal (num_outputs) x low_pass_input (num_hidden)
        if not use_previous_state:
            grad_dense = np.outer(error_signal, self.low_pass_input)
        else:
            grad_dense = np.outer(error_signal, self.previous_low_pass_input)

        # Only accumulate updates on the current sparse structure of weights
        self.accumulated_weights_update += self.non_zero_weights * grad_dense
        self.accumulated_bias_update += error_signal

    @profile
    def update_parameters(self):
        if self.errors_received == 0:
            return
        
        weights_update = self.non_zero_weights * self.adam_optimizer.get_update(
            self.accumulated_weights_update
        )
        self.weights -= (
            self.learning_rate * weights_update
        )

        self.bias = self.bias - self.learning_rate * (
            self.accumulated_bias_update
        )
        
        # Reset
        self.accumulated_weights_update *= 0
        self.accumulated_bias_update *= 0
        self.errors_received = 0


class ActorCriticOutputLayer:
    
    def reset(self):
        self.reward_trace = 0
        self.previous_value_estimate = 0

        self.previous_policy_grads = np.zeros(self.policy_num_outputs)
        self.policy_grads = np.zeros(self.policy_num_outputs)
        
        self.policy_output.reset()
        self.value_output.reset()

    def __init__(
        self,
        num_hidden: int,
        num_outputs: int,
        connection_density: float,
        learning_rate: float = 1e-3,
        tau_out: float = 20e-3,
        dt: float =1e-3,
        policy_activation_function: str = "softmax",
        gamma=0.99,
        # Adam optimizer parameters
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
    ):
        assert policy_activation_function in ["softmax", "linear"]
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.gamma = gamma
        self.cv = 1

        self.policy_num_outputs = num_outputs
        use_bias = True
        if policy_activation_function == "linear":
            self.policy_num_outputs = 2*num_outputs
            use_bias = True

        self.policy_output: OutputLayer = OutputLayer(
            num_hidden,
            self.policy_num_outputs,
            connection_density=connection_density,
            activation_function=policy_activation_function,
            rl_gamma=self.gamma,
            tau_out=tau_out,
            learning_rate=learning_rate,
            use_bias=use_bias
        )

        self.value_output: OutputLayer = OutputLayer(
            num_hidden,
            1,
            activation_function="linear",
            connection_density=connection_density,
            rl_gamma=self.gamma,
            tau_out=tau_out,
            learning_rate=10*learning_rate
        )
        self.reset()
    
    def receive_pulse(self, sparse_spike_vector: sp.csr_array):
        self.policy_output.receive_pulse(sparse_spike_vector)
        self.value_output.receive_pulse(sparse_spike_vector)
    
    def next_time_step(self):
        self.policy_output.next_time_step()
        self.value_output.next_time_step()

    def _compute_log_normal_dist_policy_grad(self, action_taken, means, log_stds):
        variances = np.exp(log_stds*2)

        means_diff = action_taken - means
        logp_gradient_means = - means_diff / variances
        logp_gradient_logstds = 1 - (means_diff ** 2 / variances)

        self.previous_policy_grads = self.policy_grads.copy()
        self.policy_grads = np.concatenate([logp_gradient_means, 10*logp_gradient_logstds])
        
    def _compute_log_softmax_dist_policy_grad(self, p: np.ndarray, action_taken: np.ndarray):
        self.previous_policy_grads = self.policy_grads.copy()
        self.policy_grads = p - action_taken

    def action(self):
        output = self.policy_output.output()

        if self.policy_output.activation_function == "linear":
            means = output[:self.num_outputs]
            log_stds = output[self.num_outputs:]
            stds = np.exp(log_stds)

            action = np.random.normal(means, stds)
            # print("Action: ", action)
            self._compute_log_normal_dist_policy_grad(action, means, log_stds)

        elif self.policy_output.activation_function == "softmax":
            action_index = np.random.choice(len(output), p=output)
            action = np.zeros_like(output)
            action[action_index] = 1.0
            self._compute_log_softmax_dist_policy_grad(output, action)

        # print(output)

        return action

    def td_error_update(self, reward: float):
        value_estimate = self.value_output.output()[0]

        current_state_td_error = (
            self.previous_value_estimate
            - self.gamma * value_estimate - reward
        )
        # Book formula for td error
        previous_state_td_error = (
            self.gamma * value_estimate + reward 
            - self.previous_value_estimate
        )

        # print(f"PV: {self.previous_value_estimate:.3f}, V: {value_estimate:.3f}, R: {reward:.3f}, dt: {td_error:.3f}")
        self.previous_value_estimate = value_estimate

        self.policy_output.receive_error(
            -previous_state_td_error * self.previous_policy_grads,
            use_previous_state=True
        )
        self.value_output.receive_error(
            -previous_state_td_error,
            use_previous_state=True
        )

        self.policy_output.receive_error(
            -current_state_td_error * self.policy_grads
        )
        self.value_output.receive_error(
            -current_state_td_error
        )


        return (
            self.policy_grads, current_state_td_error, 
            self.previous_policy_grads, previous_state_td_error
        )

    def receive_reward(self, reward: float):
        self.reward_trace = self.reward_trace * self.gamma + reward
        value_estimate = self.value_output.output()[0]

        advantage = np.mean(self.reward_trace - value_estimate)

        # print("Learning signal: ", rl_learning_signal)
        # print(f"V: {value_estimate}, R: {reward}, ADV: {advantage}")

        self.policy_output.receive_error(
            advantage * self.policy_grads
        )
        self.value_output.receive_error(
            -advantage
        )

        return self.policy_grads, advantage
    
    def update_parameters(self):
        self.policy_output.update_parameters()
        self.value_output.update_parameters()
