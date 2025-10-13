import numpy as np
import scipy.sparse as sp
import multiprocessing as mp
import time

from srnn import ALIFLayer, ActorCriticOutputLayer, OutputLayer

class LocalBroadcastSrnn:
    def __init__(
        self, 
        num_neurons_list, 
        input_size, output_size, 
        input_connectivity=0.1,
        hidden_connectivity=0.01,
        local_connectivity=0.1,
        output_connectivity=0.03,
        how_many_outputs=1,
        layer_configs: list | None = None,
        output_activation_function: str = "softmax",
        tau_out: float = 30e-3,
        tau_adaptation: float = 2000e-3,
        tau: float = 200e-3,
        unary_weights: bool = False,
        self_predict: bool = False,
        target_firing_rate: int = 13,
        firing_threshold: float = 0.5,
        default_learning_rate: float = 1e-3,
        rl_gamma: float | None = None
    ):
        if layer_configs is None:
            layer_configs = [{} for _ in num_neurons_list]
        assert len(layer_configs) == len(num_neurons_list)
        assert all("num_neurons" not in l for l in layer_configs)

        self.rl_gamma = rl_gamma

        self.time_step = 0

        self.num_layers = len(num_neurons_list)
        self.num_neurons_list = num_neurons_list
        self.input_size = input_size
        self.output_size = output_size

        self.input_queues = [mp.Queue() for _ in range(self.num_layers)]
        self.status_queues = [mp.Queue() for _ in range(self.num_layers)]
        self.output_queue = mp.Queue()

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

        self.lif_output_size = (
            2*output_size 
                if output_activation_function == "linear" and rl_gamma is not None
                else output_size
        )

        # Default LIFLayer parameters
        default_lif_kwargs = dict(
            just_input_size=input_size,
            num_inputs=self.global_size,
            firing_threshold=firing_threshold,
            learning_rate=default_learning_rate,
            learned_loss_learning_rate=1e-3,
            input_connection_density=input_connectivity,
            hidden_connection_density=hidden_connectivity,
            local_connection_density=local_connectivity,
            self_predict=self_predict,
            output_sizes={
                "base": self.output_size,
                "policy": self.lif_output_size,
                "value": 1
            },
            tau=tau,
            tau_adaptation=tau_adaptation,
            tau_out=tau_out,
            target_firing_rate=target_firing_rate,
            beta="sparse_adaptive",
            beta_params={
                "lif_fraction": 0.8,  # Fraction of LIF neurons
                "exp_scale": 0.2,     # Scale parameter for exponential distribution
                "max_beta": 2.0       # Maximum beta value
            }
        )
        for i in range(self.num_layers):
            p = mp.Process(
                target=self.lif_worker,
                kwargs=dict(
                    n_layers=self.num_layers,
                    layer_id=i, 
                    config=default_lif_kwargs | layer_configs[i] | {"num_neurons": num_neurons_list[i]}, 
                    input_queue=self.input_queues[i], 
                    layer_offset=self.layers_offsets[i], 
                    global_size=self.global_size,
                    left_input_queue=self.input_queues[i-1] if i > 0 else None,
                    right_input_queue=self.input_queues[i+1] if i < self.num_layers-1 else self.output_queue,
                    output_queue=self.output_queue,
                    status_queue=self.status_queues[i]
                )
            )
            p.start()
            self.processes.append(p)

        num_hidden = sum(num_neurons_list)
        self.output_layers = []
        for i in range(how_many_outputs):
            if self.rl_gamma:
                self.output_layers.append(ActorCriticOutputLayer(
                    num_hidden=input_size + num_hidden,
                    num_outputs=output_size,
                    learning_rate=default_learning_rate,
                    connection_density=output_connectivity,
                    policy_activation_function=output_activation_function,
                    # input_offset=self.layers_offsets[-1]
                    # unary_weights=unary_weights,
                    tau_out=tau_out,
                    gamma=self.rl_gamma
                ))
            else:
                self.output_layers.append(OutputLayer(
                    num_hidden=input_size + num_hidden,
                    num_outputs=output_size,
                    learning_rate=default_learning_rate,
                    connection_density=output_connectivity,
                    activation_function=output_activation_function,
                    # input_offset=self.layers_offsets[-1]
                    # unary_weights=unary_weights,
                    tau_out=tau_out,
                    rl_gamma=self.rl_gamma
                ))


        # print("OW:", self.output_layer.weights.nnz)

    @staticmethod
    def lif_worker(
            n_layers,
            layer_id, 
            config, 
            input_queue, 
            layer_offset, 
            global_size,
            left_input_queue,
            right_input_queue,
            output_queue,
            status_queue
        ): 
        np.random.seed((time.time_ns())%2**32)
        # beta = abs(np.random.random())
        # print(f"Random beta: {beta}")
        layer = ALIFLayer(**config, recurrent_start=layer_offset)
        
        print("Layer", layer.weights.nnz)
        last_time = time.time_ns()
        how_many = 0
        how_many_pulses = 0

        right_pulses = sp.csr_matrix((1, global_size))
        left_pulses = sp.csr_matrix((1, global_size))
        iteration = 0
        while True:
            instruction, data = input_queue.get()

            # print(layer_id, instruction)
            match instruction:

                case "STOP":
                    break

                case "PULSE-R":
                    layer.receive_pulse(data)
                    right_pulses = right_pulses.maximum(data)
                    how_many_pulses += 1

                case "PULSE-L":
                    layer.receive_pulse(data)
                    left_pulses = left_pulses.maximum(data)
                    how_many_pulses += 1
                
                case "NEXT_TIME_STEP":
                    if layer_id != 0:
                        left_input_queue.put(("NEXT_TIME_STEP", None))
                    layer.next_time_step()
                    local_spikes = layer.get_output_spikes()
                    output_spikes = sp.hstack([
                        sp.csr_matrix((1, layer_offset)),
                        local_spikes,
                        sp.csr_matrix((1, global_size - layer_offset - layer.num_neurons))
                    ])
                    layer.receive_pulse(output_spikes)

                    left_pulses = left_pulses.maximum(output_spikes)
                    right_pulses = right_pulses.maximum(output_spikes)

                    right_input_queue.put(("PULSE-L", left_pulses))
                    if layer_id != 0:
                        left_input_queue.put(("PULSE-R", right_pulses))
                        # left_input_queue.put(("NEXT_TIME_STEP", None))
                    status_queue.put(("DONE", None))
                    right_pulses = sp.csr_matrix((1, global_size))
                    left_pulses = sp.csr_matrix((1, global_size))

                case "RECEIVE ERROR":
                    layer.receive_error(data)

                case "RECEIVE RL GRADIENT":
                    layer.receive_rl_gradient(*data)

                case "UPDATE":
                    # print(f"FR: {1000*layer.firing_rate/layer.num_neurons:.3f} Hz")
                    layer.update_parameters()

                case "OUTWEIGHTS":
                    layer.loss_weights[data.indices] = data[data.indices]

                case "RESET":
                    how_many_pulses = 0
                    layer.reset()
                    # print(how_many_pulses)
                    # print(input_queue.qsize())

                # case "N_CONNECTIONS":
                #     # Return the number of nonzero weights
                #     output_queue.put(("N_CONNECTIONS", layer.weights.nnz))

    def input(self, input_data):
        # input_data: csr_matrix of shape (1, input_size + total_neurons)
        self.input_queues[-1].put(("PULSE-R", input_data))
        self.input_queues[-1].put(("NEXT_TIME_STEP", None))
        # for q in self.input_queues[::-1]:
        #     q.put(("NEXT_TIME_STEP", None))
        for q in self.status_queues:
            instruction, data = q.get()
            assert instruction == "DONE"

        instruction, data = self.output_queue.get()

        for output_layer in self.output_layers:
            output_layer.receive_pulse(data)
            output_layer.next_time_step()

        self.time_step += 1

    def output(self):
        return [output_layer.output() for output_layer in self.output_layers]
    
    def action(self):
        return [output_layer.action() for output_layer in self.output_layers]

    def td_error_update(self, reward: float):
        total_td_error = 0
        for output_layer in self.output_layers:
            policy_gradient, this_state_td_error, previous_policy_grads, previous_state_td_error = output_layer.td_error_update(reward)
            total_td_error += this_state_td_error
            for i, q in enumerate(self.input_queues[::-1]):
                if self.time_step >= (i-1)*2:
                    #print(f"Sending RL gradient to layer {i}, time step {self.time_step}")
                    q.put(("RECEIVE RL GRADIENT", 
                        (self.time_step-i, policy_gradient, this_state_td_error, 
                    previous_policy_grads, previous_state_td_error)))
        return total_td_error/len(self.output_layers)
        
    def receive_reward(self, reward: float):
        total_advantage = 0
        for output_layer in self.output_layers:
            policy_gradient, advantage = output_layer.receive_reward(reward)
            total_advantage += advantage
            for i, q in enumerate(self.input_queues[::-1]):
                if self.time_step >= (i-1)*2:
                    q.put(("RECEIVE RL GRADIENT", 
                        (self.time_step-i, policy_gradient, advantage, policy_gradient*0, advantage*0))
                    )
        return advantage
    
    def receive_feedback(self, target):
        errors = self.output_layer[0].compute_error(target)
        loss = self.output_layer.compute_loss(target)
        for i, q in enumerate(self.input_queues[::-1]):
            if self.time_step >= (i-1)*2:
                q.put(("RECEIVE ERROR", errors))
        self.output_layer.receive_error(errors)
        return loss
    
    def update_parameters(self):
        for q in self.input_queues:
            q.put(("UPDATE", None))
        for o in self.output_layers:
            o.update_parameters()
    
    # def update_outweights(self):
    #     for i, q in enumerate(self.input_queues):
    #         start = self.layers_offsets[i]
    #         end = start + self.num_neurons_list[i]
    #         out_w = self.output_layer.weights[:, start:end].T  # shape: (layer_size, num_outputs)
    #         q.put(("OUTWEIGHTS", out_w))
    
    def reset(self):
        self.time_step = 0
        for q in self.input_queues:
            q.put(("RESET", None))
        for o in self.output_layers:
            o.reset()

    def shutdown(self):
    #     for q in self.input_queues:
    #         q.put(("STOP", None))
        for p in self.processes:
            p.terminate()
        for q in self.input_queues:
            q.cancel_join_thread()
        self.output_queue.cancel_join_thread()
