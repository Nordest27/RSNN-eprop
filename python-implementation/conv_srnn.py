import numpy as np
import scipy.sparse as sp
from srnn import ALIFLayer, AdamOptimizer
from multiprocessing import Process, Queue


def extract_patches(image: np.ndarray, patch_size: tuple, stride: tuple) -> list[sp.csr_array]:
    """
    Args:
        image: 2D input array (e.g., 28x28)
        patch_size: (height, width)
        stride: (stride_y, stride_x)
    Returns:
        List of sparse CSR matrices (flattened patches)
    """
    H, W = image.shape
    ph, pw = patch_size
    sh, sw = stride
    patches = []

    for i in range(0, H - ph + 1, sh):
        for j in range(0, W - pw + 1, sw):
            patch = image[i:i+ph, j:j+pw].flatten()
            sparse_patch = sp.csr_matrix(patch)
            patches.append(sparse_patch)
    return patches


class ConvDistributedALIFLayer:
    def __init__(
        self, 
        num_patches, 
        patch_input_size, 
        shared_weight_config, 
        global_input_size,
        # Adam optimizer parameters
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
    ):
        """
        Args:
            num_patches (int): Number of patches / positions (e.g., sliding windows)
            patch_input_size (int): Input size for each ALIF layer (flattened patch)
            shared_weight_config (dict): ALIFLayer kwargs (excluding num_inputs)
            global_input_size (int): Total input size (used for spike vector shape)
        """
        self.num_patches = num_patches
        self.patch_input_size = patch_input_size
        self.global_input_size = global_input_size
        self.adam_optimizer = AdamOptimizer(
            self.weights.shape, adam_beta1, adam_beta2, adam_epsilon
        )

        self.layers = [
            ALIFLayer(
                just_input_size=patch_input_size,
                num_inputs=global_input_size,
                num_neurons=shared_weight_config["num_neurons"],
                output_sizes=shared_weight_config["output_sizes"],
                recurrent_start=global_input_size - shared_weight_config["num_neurons"],
                **{k: v for k, v in shared_weight_config.items() if k not in ["num_neurons", "output_sizes"]}
            )
            for _ in range(num_patches)
        ]

        first_layer = self.layers[0]
        for layer in self.layers[1:]:
            layer.weight_mask = first_layer.weight_mask
            layer.initial_weight_values = first_layer.initial_weight_values
            layer.initial_weights = first_layer.initial_weights

    def forward(self, input_patches):
        """
        input_patches: list of sparse CSR vectors (one per patch)
        """
        self.outputs = []
        for i in range(self.num_patches):
            self.layers[i].receive_pulse(input_patches[i])
            self.layers[i].next_time_step()
            self.outputs.append(self.layers[i].get_output_spikes())

        return self.outputs

    def receive_error(self, errors):
        for i in range(self.num_patches):
            self.layers[i].receive_error(errors[i])

    def aggregate_and_update(self, method="mean"):
        """
        Aggregate gradients/updates and apply to all layers.
        """

        # Step 1: Collect base weight updates
        updates = [layer.base_weights_update for layer in self.layers]

        # Step 2: Aggregate
        if method == "mean":
            agg_update = sum(updates) * (1 / self.num_patches)
        elif method == "sum":
            agg_update = sum(updates)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        # Step 3: Apply aggregated update to all layers
        for layer in self.layers:
            layer.base_weights_update = agg_update.copy()
            layer.update_parameters()

    def reset(self):
        for layer in self.layers:
            layer.reset()
