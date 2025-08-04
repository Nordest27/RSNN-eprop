import numpy as np
import scipy.sparse as sp
import multiprocessing as mp
import time
from srnn import ALIFLayer, OutputLayer 


class SimpleBroadcastSrnn:
    def __init__(
        self, 
        num_neurons_list, 
        input_size, output_size, 
        input_connectivity=0.1,
        hidden_connectivity=0.01,
        local_connectivity=0.1,
        output_connectivity=0.03,
        layer_configs: list | None = None,
        output_activation_function: str = "softmax",
        tau_out: float = 30e-3,
        unary_weights: bool = False,
        self_predict: bool = False
    ):
        if layer_configs is None:
            layer_configs = [{} for _ in num_neurons_list]
        assert len(layer_configs) == len(num_neurons_list)
        assert all("num_neurons" not in l for l in layer_configs)

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

        self.previous_output_spikes = sp.csr_matrix((1, self.global_size))
        
        # Default LIFLayer parameters
        default_lif_kwargs = dict(
            just_input_size=input_size,
            num_inputs=self.global_size,
            firing_threshold=1.0,
            learning_rate=0.0001,
            input_connection_density=input_connectivity,
            hidden_connection_density=hidden_connectivity,
            local_connection_density=local_connectivity,
            self_predict=self_predict,
            output_size=output_size,
            tau_out=tau_out,
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
                    layer_id=i, 
                    config=default_lif_kwargs | layer_configs[i] | {"num_neurons": num_neurons_list[i]}, 
                    input_queue=self.input_queues[i], 
                    output_queue=self.output_queues[i],
                    layer_offset=self.layers_offsets[i], 
                    global_size=self.global_size,
                )
            )
            p.start()
            self.processes.append(p)

        num_hidden = sum(num_neurons_list)
        self.output_layer = OutputLayer(
            num_hidden=input_size + num_hidden,
            num_outputs=output_size,
            learning_rate=0.001,
            connection_density=output_connectivity,
            activation_function=output_activation_function,
            # input_offset=self.layers_offsets[-1]
            unary_weights=unary_weights,
            tau_out=tau_out
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
        np.random.seed((time.time_ns())%2**32)
        # beta = abs(np.random.random())
        # print(f"Random beta: {beta}")
        layer = ALIFLayer(**config, recurrent_start=layer_offset)
        while True:
            instruction, data = input_queue.get()
            match instruction:
                case "STOP":
                    break
                case "PULSE":
                    # print("PULSE")
                    layer.receive_pulse(data)  # expects csr_matrix
                case "NEXT_TIME_STEP":
                    # Get local spikes
                    layer.next_time_step()
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
        input_data = input_data.maximum(self.previous_output_spikes)
        for q in self.input_queues:
            q.put(("PULSE", input_data))
            q.put(("NEXT_TIME_STEP", input_data))

        # Wait for any layer to finish
        # ready = [q for q in self.output_queues if not q.empty()]
        self.previous_output_spikes = sp.csr_matrix((1, self.global_size))
        for q in self.output_queues:
            # idx = self.output_queues.index(ready.pop())
            # layer_id, output_spikes = self.output_queues[idx].get()
            output_spikes = q.get()[1]
            self.previous_output_spikes = self.previous_output_spikes.maximum(output_spikes)
            # Broadcast this output to all layers for the next step  
            # for q in self.input_queues:
            #     # print("Sending Pulse")
            #     q.put(("PULSE", output_spikes))
            self.output_layer.receive_pulse(output_spikes)

        self.output_layer.next_time_step()
        # input("input")

    def output(self):
        return self.output_layer.output()
    
    def feedback(self, value: int | float):
        errors = self.output_layer.compute_error(value)
        loss = self.output_layer.compute_loss(value)
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
        self.previous_output_spikes = sp.csr_matrix((1, self.global_size))
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
