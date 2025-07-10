import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle 
import scipy.sparse as sp
import time
from srnn import ALIFLayer, SoftmaxOutputLayer

class SRNNVisualizer:
    def __init__(self, lif_layer, input_data, output_layer=None, target_data=None, figsize=(18, 12)):
        """
        Visualize SRNN network dynamics
        
        Args:
            lif_layer: Your LIFLayer instance
            input_data: Input spike data [time_steps, num_inputs]
            output_layer: Optional SoftmaxOutputLayer for output visualization
            target_data: Optional target data for comparison
            figsize: Figure size tuple
        """
        self.lif_layer = lif_layer
        self.output_layer = output_layer
        self.input_data = input_data
        self.target_data = target_data
        self.time_steps = input_data.shape[0]
        self.current_step = 0
        self.n_inps = len(input_data[0])

        # Storage for animation data
        self.spike_history = []
        self.input_history = []
        self.output_history = []  # For softmax outputs
        self.connections = []
        self.output_connections = []  # For output layer connections
        
        # Create figure and subplots
        self.fig = plt.figure(figsize=figsize)
        self.fig.patch.set_facecolor('black')
        
        # Setup subplots
        self.setup_subplots()
        
        # Initialize visualization elements
        self.setup_visualization_elements()
        
    def setup_subplots(self):
        """Create and arrange subplots"""
        if self.output_layer is not None:
            # With output layer: 4 subplots
            gs = self.fig.add_gridspec(3, 3, height_ratios=[1, 1, 2], width_ratios=[3, 3, 1])
            
            # Input spikes display
            self.ax_input = self.fig.add_subplot(gs[0, :2])
            self.ax_input.set_title('Input Spikes', color='white', fontsize=14, fontweight='bold')
            self.ax_input.set_facecolor('black')

            # Hidden layer spikes
            self.ax_hidden = self.fig.add_subplot(gs[1, :2])
            self.ax_hidden.set_title('Hidden Layer Spikes', color='white', fontsize=14, fontweight='bold')
            self.ax_hidden.set_facecolor('black')

            # Network visualization (main)
            self.ax_network = self.fig.add_subplot(gs[2, :2])
            self.ax_network.set_title('Network Activity', color='white', fontsize=16, fontweight='bold')
            self.ax_network.set_facecolor('black')
            self.ax_network.set_aspect('equal')
            
            # Output layer visualization
            self.ax_output = self.fig.add_subplot(gs[:, 2])
            self.ax_output.set_title('Output Layer\n(Softmax)', color='white', fontsize=14, fontweight='bold')
            self.ax_output.set_facecolor('black')
            
        else:
            # Without output layer: original 3 subplots
            gs = self.fig.add_gridspec(3, 1, height_ratios=[1, 1, 2])
            
            # Input spikes display
            self.ax_input = self.fig.add_subplot(gs[0, :])
            self.ax_input.set_title('Input Spikes', color='white', fontsize=14, fontweight='bold')
            self.ax_input.set_facecolor('black')

            # Hidden layer spikes
            self.ax_hidden = self.fig.add_subplot(gs[1, :])
            self.ax_hidden.set_title('Hidden Layer Spikes', color='white', fontsize=14, fontweight='bold')
            self.ax_hidden.set_facecolor('black')

            # Network visualization (main)
            self.ax_network = self.fig.add_subplot(gs[2, :])
            self.ax_network.set_title('Network Activity', color='white', fontsize=16, fontweight='bold')
            self.ax_network.set_facecolor('black')
            self.ax_network.set_aspect('equal')
            
        # Style all axes
        axes_to_style = [self.ax_input, self.ax_hidden, self.ax_network]
        if self.output_layer is not None:
            axes_to_style.append(self.ax_output)
            
        for ax in axes_to_style:
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
                
    def setup_visualization_elements(self):
        """Initialize visualization elements"""
        # Network layout - arrange neurons in a grid
        self.neuron_positions = self.create_neuron_layout()
        
        # Initialize neuron circles
        self.neuron_circles = []
        for i, (x, y) in enumerate(self.neuron_positions):
            circle = Circle((x, y), 0.3, facecolor='blue', edgecolor='cyan', linewidth=2)
            self.ax_network.add_patch(circle)
            self.neuron_circles.append(circle)
            
        # Draw connections (sample a subset for visibility)
        self.draw_connections()
        
        # Initialize plots
        self.input_plot = self.ax_input.imshow(
            np.zeros((self.n_inps, 50)), 
            cmap='hot', aspect='auto', animated=True
        )    
        
        self.hidden_plot = self.ax_hidden.imshow(
            np.zeros((self.lif_layer.num_neurons, 50)), 
            cmap='hot', aspect='auto', animated=True
        )
        
        # Initialize output layer visualization if present
        if self.output_layer is not None:
            self.setup_output_visualization()
        
        # Add time step text
        self.time_text = self.fig.text(0.01, 0.99, '', ha='left', va='top', 
                                      fontsize=16, color='white', fontweight='bold')
        
    def setup_output_visualization(self):
        """Setup output layer specific visualization elements"""
        num_outputs = self.output_layer.num_outputs
        
        # Create output neuron positions (vertical arrangement)
        self.output_positions = []
        for i in range(num_outputs):
            x = max(self.neuron_positions[:, 0]) + 3  # Position to the right
            y = (num_outputs - i) * 1.5  # Vertical spacing
            self.output_positions.append((x, y))
        
        # Add output neurons to network plot
        self.output_circles = []
        for i, (x, y) in enumerate(self.output_positions):
            circle = Circle((x, y), 0.4, facecolor='green', edgecolor='lime', linewidth=2)
            self.ax_network.add_patch(circle)
            self.output_circles.append(circle)
            
        # Draw connections from hidden layer to output layer
        self.draw_output_connections()
        
        # Setup output heatmap
        self.output_heatmap = self.ax_output.imshow(
            np.zeros((num_outputs, 50)), 
            cmap='viridis', aspect='auto', animated=True,
            vmin=0, vmax=1
        )
        
        # Add class labels
        class_labels = [f'Class {i}' for i in range(num_outputs)]
        self.ax_output.set_yticks(range(num_outputs))
        self.ax_output.set_yticklabels(class_labels)
        self.ax_output.set_xlabel('Time Steps', color='white')
        
        # Add colorbar
        cbar = plt.colorbar(self.output_heatmap, ax=self.ax_output)
        cbar.set_label('Softmax Probability', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.yaxis.set_ticklabels(cbar.ax.yaxis.get_ticklabels(), color='white')
        
    def draw_output_connections(self):
        """Draw connections from hidden layer to output layer"""
        if self.output_layer is None:
            return
            
        # Sample connections to avoid clutter
        rows, cols = self.output_layer.weights.nonzero()
        n_connections = min(len(rows), 50)  # Limit to 50 connections for visibility
        
        if n_connections > 0:
            indices = np.random.choice(len(rows), n_connections, replace=False)
            output_indices = rows[indices]
            hidden_indices = cols[indices]
            
            for i, (out_idx, hidden_idx) in enumerate(zip(output_indices, hidden_indices)):
                # Position of hidden neuron
                x1, y1 = self.neuron_positions[self.n_inps + hidden_idx]
                # Position of output neuron
                x2, y2 = self.output_positions[out_idx]
                
                weight = self.output_layer.weights[out_idx, hidden_idx]
                line = self.ax_network.plot(
                    [x1, x2], [y1, y2], 
                    color='orange', alpha=0.3, 
                    linewidth=min(3.0, abs(weight) * 5)
                )[0]
                self.output_connections.append((hidden_idx, out_idx, line))
        
    def create_neuron_layout(self):
        """Create a grid layout for neurons"""
        n_neurons = self.lif_layer.num_neurons
        inputs_grid_size = int(np.ceil(np.sqrt(self.n_inps)))
        neurons_grid_size = int(np.ceil(np.sqrt(n_neurons)))
        
        positions = []
        # Input neurons
        for i in range(self.n_inps):
            row = inputs_grid_size - (1 + i // inputs_grid_size)
            col = i % inputs_grid_size
            x = col * 1.0
            y = row * 1.0
            positions.append((x, y))
        
        # Hidden neurons
        for i in range(n_neurons):
            row = inputs_grid_size - (1 + i // neurons_grid_size)
            col = inputs_grid_size + 2 + i % neurons_grid_size  # Gap between input and hidden
            x = col * 1.0
            y = row * 1.0
            positions.append((x, y))
            
        return np.array(positions)
        
    def draw_connections(self):
        """Draw a sample of connections between neurons"""
        # Sample connections to avoid clutter
        rows, cols = self.lif_layer.weights.nonzero()
        n_connections = min(len(rows), 500)  # Limit connections for visibility
        
        if n_connections > 0:
            indices = np.random.choice(len(rows), n_connections, replace=False)
            outs = rows[indices]
            outs_shift = rows[indices] + self.n_inps
            inps = cols[indices]
            
            # Draw connections
            for i, (out_s, inp) in enumerate(zip(outs_shift, inps)):
                x1, y1 = self.neuron_positions[inp]
                x2, y2 = self.neuron_positions[out_s]
                line = self.ax_network.plot(
                    [x1, x2], [y1, y2],
                    alpha=0.1, color='gray',
                    linewidth=min(2.0, 5*abs(self.lif_layer.weights[outs[i], inp]))
                )[0]
                self.connections.append((inp, line))
                        
    def update_visualization(self, frame):
        """Update visualization for animation"""
        if self.current_step >= self.time_steps:
            return []
        
        # Get current input
        current_input = self.input_data[self.current_step]
        
        # Update connection colors based on activity
        for i, line in self.connections:
            if i >= self.n_inps:
                line.set_color('red' if self.lif_layer.sending_pulses[i-self.n_inps] else "gray")
                line.set_alpha(0.5 if self.lif_layer.sending_pulses[i-self.n_inps] else 0.1)
            else:
                line.set_color('red' if current_input[i] else "gray")
                line.set_alpha(0.5 if current_input[i] else 0.1)
            
        # Update output connections if present
        if self.output_layer is not None:
            for hidden_idx, out_idx, line in self.output_connections:
                is_active = (hidden_idx < len(self.lif_layer.sending_pulses) and 
                           self.lif_layer.sending_pulses[hidden_idx])
                line.set_alpha(0.7 if is_active else 0.3)
                line.set_color('yellow' if is_active else 'orange')

        # Process through the ACTUAL LIF layer
        if hasattr(current_input, 'toarray'):
            input_sparse = current_input
        else:
            # Convert to sparse format expected by receive_pulse
            if len(current_input.shape) == 1:
                input_sparse = sp.csr_matrix(current_input.reshape(1, -1))
            else:
                input_sparse = sp.csr_matrix(current_input)
                
        output = self.lif_layer.get_output_spikes()
        input_sparse = sp.hstack([
            input_sparse, 
            output.reshape(1, -1)
        ])
        
        # Run the actual LIF layer computation
        self.lif_layer.receive_pulse(input_sparse)
        # self.lif_layer.next_time_step()
        
        # Process output layer if present
        if self.output_layer is not None:
            hidden_output = self.lif_layer.get_output_spikes()
            if hidden_output.nnz > 0:
                self.output_layer.receive_pulse(hidden_output)
            # self.output_layer.update()
            
            # Store output probabilities
            output_probs = self.output_layer.output()
            self.output_history.append(output_probs.copy())
        
        # Gather REAL data from the LIF layer
        self.spike_history.append(self.lif_layer.sending_pulses.copy())
        self.input_history.append(current_input.copy())
        
        # Update neuron colors based on ACTUAL membrane potential and spikes from LIF layer
        max_potential = max(self.lif_layer.firing_threshold * 1.5, 
                           np.max(self.lif_layer.membrane_potentials) if len(self.lif_layer.membrane_potentials) > 0 else 1.0)
        
        for i, circle in enumerate(self.neuron_circles):
            i_neuron = i - self.n_inps
            # Use ACTUAL spiking state from LIF layer
            if i_neuron < 0:
                # Input neurons
                if current_input[i]:
                    circle.set_facecolor('#FFFFFF')
                    circle.set_edgecolor('gray')
                    circle.set_linewidth(3)
                    circle.set_radius(0.4)
                else:
                    circle.set_facecolor('#040404')
                    circle.set_edgecolor('gray')
                    circle.set_linewidth(1)
                    circle.set_radius(0.3)
            elif i_neuron < len(self.lif_layer.sending_pulses) and self.lif_layer.sending_pulses[i_neuron]:
                # Bright color for spiking neurons
                circle.set_facecolor('#FFFF00')
                circle.set_edgecolor('#FF0000')
                circle.set_linewidth(3)
                circle.set_radius(0.4)
            else:
                # Color based on ACTUAL membrane potential from LIF layer
                potential = self.lif_layer.membrane_potentials[i_neuron] if i_neuron < len(self.lif_layer.membrane_potentials) else 0
                intensity = min(1.0, potential / max_potential)
                color_intensity = max(0.1, intensity)
                circle.set_facecolor((0, 0, color_intensity))
                circle.set_edgecolor('cyan')
                circle.set_linewidth(1)
                circle.set_radius(0.3)
        
        # Update output neuron colors if present
        if self.output_layer is not None and self.output_history:
            current_output = self.output_history[-1]
            max_prob = np.max(current_output)
            predicted_class = np.argmax(current_output)
            
            for i, circle in enumerate(self.output_circles):
                prob = current_output[i]
                
                if i == predicted_class:
                    # Highlight predicted class
                    circle.set_facecolor('#00FF00')
                    circle.set_edgecolor('#FFFFFF')
                    circle.set_linewidth(3)
                    circle.set_radius(0.5)
                else:
                    # Color based on probability
                    intensity = prob / max_prob if max_prob > 0 else 0
                    circle.set_facecolor((0, intensity, 0))
                    circle.set_edgecolor('lime')
                    circle.set_linewidth(1)
                    circle.set_radius(0.4)
                
        # Update plots
        window_size = 50
        
        # Input spikes
        if len(self.input_history) > window_size:
            input_window = np.array(self.input_history[-window_size:]).T
        else:
            input_window = np.zeros((self.n_inps, window_size))
            if self.input_history:
                input_data = np.array(self.input_history).T
                input_window[:, -len(self.input_history):] = input_data
                
        self.input_plot.set_array(input_window)
        self.input_plot.set_clim(0, 1)
        
        # Hidden layer spikes
        if len(self.spike_history) > window_size:
            spike_window = np.array(self.spike_history[-window_size:]).T
        else:
            spike_window = np.zeros((self.lif_layer.num_neurons, window_size))
            if self.spike_history:
                spike_data = np.array(self.spike_history).T
                spike_window[:, -len(self.spike_history):] = spike_data
                
        self.hidden_plot.set_array(spike_window)
        self.hidden_plot.set_clim(0, 1)
        
        # Output layer heatmap
        if self.output_layer is not None and self.output_history:
            if len(self.output_history) > window_size:
                output_window = np.array(self.output_history[-window_size:]).T
            else:
                output_window = np.zeros((self.output_layer.num_outputs, window_size))
                if self.output_history:
                    output_data = np.array(self.output_history).T
                    output_window[:, -len(self.output_history):] = output_data
                    
            self.output_heatmap.set_array(output_window)
            self.output_heatmap.set_clim(0, 1)
        
        # Update time text
        self.time_text.set_text(f'Time Step: {self.current_step}')
        
        self.current_step += 1
        
        return_list = [self.input_plot, self.hidden_plot] + self.neuron_circles
        if self.output_layer is not None:
            return_list.extend([self.output_heatmap] + self.output_circles)
        
        return return_list
        
    def animate(self, interval=100, save_path=None):
        """Run the animation using the actual LIF layer"""
        self.lif_layer.reset()
        if self.output_layer is not None:
            self.output_layer.reset()
            
        self.current_step = 0
        self.spike_history = []
        self.input_history = []
        self.output_history = []
        
        # Set axis limits
        self.ax_network.set_xlim(-0.5, max(self.neuron_positions[:, 0]) + 0.5)
        self.ax_network.set_ylim(-0.5, max(self.neuron_positions[:, 1]) + 0.5)
        
        # Extend limits if output layer is present
        if self.output_layer is not None and self.output_positions:
            self.ax_network.set_xlim(-0.5, max(self.output_positions, key=lambda x: x[0])[0] + 0.5)
            self.ax_network.set_ylim(-0.5, max(max(self.neuron_positions[:, 1]), 
                                              max(self.output_positions, key=lambda x: x[1])[1]) + 0.5)
        
        # Create animation
        anim = animation.FuncAnimation(
            self.fig, self.update_visualization, frames=self.time_steps,
            interval=interval, blit=False, repeat=True
        )
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
            print(f"Animation saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
        
        return anim

# Example usage function
def create_demo_visualization():
    """Create a demo visualization with sample data"""
    n_inputs = 3
    n_hidden = 10
    n_outputs = 3
    
    # Create a small LIF layer for demonstration
    lif_layer = ALIFLayer(
        just_input_size=n_inputs,
        num_inputs=n_inputs,
        num_neurons=n_hidden,
        output_size=n_outputs,
        batch_size=1
    )
    
    # Create output layer
    output_layer = SoftmaxOutputLayer(
        num_hidden=n_hidden,
        num_outputs=n_outputs,
        learning_rate=0.001,
        connection_density=0.5
    )
    
    # Generate sample input data (sparse spike train)
    time_steps = 100
    input_data = np.random.random((time_steps, n_inputs)) < 0.3  # 30% spike probability
    
    # Create visualizer
    visualizer = SRNNVisualizer(lif_layer, input_data, output_layer)
    
    return visualizer

# Uncomment to run demo:
if __name__ == "__main__":
    viz = create_demo_visualization()
    anim = viz.animate(interval=200)  # 200ms between frames