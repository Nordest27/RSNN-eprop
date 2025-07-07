import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
import scipy.sparse as sp
from matplotlib.colors import LinearSegmentedColormap
import time
from srnn import LIFLayer

class SRNNVisualizer:
    def __init__(self, lif_layer, input_data, target_data=None, figsize=(15, 10)):
        """
        Visualize SRNN network dynamics
        
        Args:
            lif_layer: Your LIFLayer instance
            input_data: Input spike data [time_steps, num_inputs]
            target_data: Optional target data for comparison
            figsize: Figure size tuple
        """
        self.lif_layer = lif_layer
        self.input_data = input_data
        self.target_data = target_data
        self.time_steps = input_data.shape[0]
        self.current_step = 0
        self.n_inps = len(input_data[0])

        # Storage for animation data
        self.spike_history = []
        self.input_history = []
        self.connections = []
        
        # Create figure and subplots
        self.fig = plt.figure(figsize=figsize)
        self.fig.patch.set_facecolor('black')
        
        # Setup subplots
        self.setup_subplots()
        
        # Initialize visualization elements
        self.setup_visualization_elements()
        
    def setup_subplots(self):
        """Create and arrange subplots"""
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1, 1, 2])
        
        # Input spikes display
        self.ax_input = self.fig.add_subplot(gs[0, :])
        self.ax_input.set_title('Input Spikes', color='white', fontsize=14, fontweight='bold')
        self.ax_input.set_facecolor('black')

        # Output spikes
        self.ax_output = self.fig.add_subplot(gs[1, :])
        self.ax_output.set_title('Output Spikes', color='white', fontsize=14, fontweight='bold')
        self.ax_output.set_facecolor('black')

        # Network visualization (main)
        self.ax_network = self.fig.add_subplot(gs[2, :])
        self.ax_network.set_title('Network Activity', color='white', fontsize=16, fontweight='bold')
        self.ax_network.set_facecolor('black')
        self.ax_network.set_aspect('equal')
        
        
        # Style all axes
        for ax in [self.ax_input, self.ax_network, self.ax_output]:
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
        
        self.output_plot = self.ax_output.imshow(
            np.zeros((self.lif_layer.num_neurons, 50)), 
            cmap='hot', aspect='auto', animated=True
        )
        
        # Add time step text
        self.time_text = self.fig.text(0.01, 0.99, '', ha='left', va='top', 
                                      fontsize=16, color='white', fontweight='bold')
        
    def create_neuron_layout(self):
        """Create a grid layout for neurons"""
        n_neurons = self.lif_layer.num_neurons
        inputs_grid_size = int(np.ceil(np.sqrt(self.n_inps)))
        neurons_grid_size = int(np.ceil(np.sqrt(n_neurons)))
        
        positions = []
        for i in range(self.n_inps):
            row = i // inputs_grid_size
            col = i % inputs_grid_size
            x = col * 1.0
            y = row * 1.0
            positions.append((x, y))
        for i in range(n_neurons):
            row = i // neurons_grid_size
            col = inputs_grid_size + i % neurons_grid_size
            x = col * 1.0
            y = row * 1.0
            positions.append((x, y))
            
        return np.array(positions)
        
    def draw_connections(self):
        """Draw a sample of connections between neurons"""
        # Sample connections to avoid clutter
        rows, cols = self.lif_layer.weights.nonzero()
        n_connections = len(rows)
        
        if n_connections > 0:
            indices = np.random.choice(len(rows), n_connections, replace=False)
            outs = rows[indices]
            outs_shift = rows[indices] + self.n_inps
            inps = cols[indices]
            
            # Draw connections (this is a simplified version - you'd map input indices to positions)
            for i, (out_s, inp) in enumerate(zip(outs_shift, inps)):
                x1, y1 = self.neuron_positions[inp]
                x2, y2 = self.neuron_positions[out_s]
                self.connections.append((inp, self.ax_network.plot(
                    [x1, x2], [y1, y2],
                    alpha=0, 
                    linewidth=min(2.0, abs(self.lif_layer.weights[outs[i], inp]))
                )[0]))
                        
    def update_visualization(self, frame):
        """Update visualization for animation"""
        if self.current_step >= self.time_steps:
            return []
        
        # Get current input
        current_input = self.input_data[self.current_step]
        
        for i, connection in self.connections:
            if i >= self.n_inps:
                connection._color = 'red' if self.lif_layer.sending_pulses[i-self.n_inps] else "gray"
            else:
                connection._color = 'red' if current_input[i] else "gray"
            
            if connection._color == "gray":
                connection._alpha = 0.05
            else:
                connection._alpha = 0.5
            

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
        self.lif_layer.next_time_step()
        
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
                # Bright color for spiking neurons
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
        
        # Output spikes
        if len(self.spike_history) > window_size:
            spike_window = np.array(self.spike_history[-window_size:]).T
        else:
            spike_window = np.zeros((self.lif_layer.num_neurons, window_size))
            if self.spike_history:
                spike_data = np.array(self.spike_history).T
                spike_window[:, -len(self.spike_history):] = spike_data
                
        self.output_plot.set_array(spike_window)
        self.output_plot.set_clim(0, 1)
        
        # Update time text
        self.time_text.set_text(f'Time Step: {self.current_step}')
        
        self.current_step += 1
        
        return [self.input_plot, self.output_plot] + self.neuron_circles
        
    def animate_with_learning(self, errors_data=None, interval=100, save_path=None):
        """Run animation with learning (if error data provided)"""
        self.lif_layer.reset()
        self.current_step = 0
        self.spike_history = []
        self.input_history = []
        
        # Set axis limits
        self.ax_network.set_xlim(-0.5, max(self.neuron_positions[:, 0]) + 0.5)
        self.ax_network.set_ylim(-0.5, max(self.neuron_positions[:, 1]) + 0.5)
        
        # Store error data for learning
        self.errors_data = errors_data
        
        # Create animation
        anim = animation.FuncAnimation(
            self.fig, self.update_with_learning, frames=self.time_steps,
            interval=interval, blit=False, repeat=True
        )
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
            print(f"Animation saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
        
    def animate(self, interval=100, save_path=None):
        """Run the animation using the actual LIF layer"""
        self.lif_layer.reset()
        self.current_step = 0
        self.membrane_history = []
        self.spike_history = []
        self.input_history = []
        
        # Set axis limits
        self.ax_network.set_xlim(-0.5, max(self.neuron_positions[:, 0]) + 0.5)
        self.ax_network.set_ylim(-0.5, max(self.neuron_positions[:, 1]) + 0.5)
        
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
        
    def update_with_learning(self, frame):
        """Update with learning if error data is provided"""
        if self.current_step >= self.time_steps:
            return []
            
        # Get current input
        current_input = self.input_data[self.current_step]
        
        # Process through the ACTUAL LIF layer
        if hasattr(current_input, 'toarray'):
            input_sparse = current_input
        else:
            if len(current_input.shape) == 1:
                input_sparse = sp.csr_matrix(current_input.reshape(1, -1))
            else:
                input_sparse = sp.csr_matrix(current_input)
            
        # Run the actual LIF layer computation
        self.lif_layer.receive_pulse(input_sparse)
        self.lif_layer.next_time_step()
        
        # Apply learning if error data is provided
        if self.errors_data is not None and self.current_step < len(self.errors_data):
            current_error = self.errors_data[self.current_step]
            self.lif_layer.receive_error(current_error)
            # Update parameters every few steps
            if self.current_step % 10 == 0:
                self.lif_layer.update_parameters()
        
        # Use the regular update visualization
        return self.update_visualization(frame)

# Example usage function
def create_demo_visualization():
    """Create a demo visualization with sample data"""
    n_inputs = 3
    # Create a small LIF layer for demonstration
    layer = LIFLayer(
        num_inputs=n_inputs,
        num_neurons=10,
        batch_size=1,
        firing_threshold=0.6,
        learning_rate=0.001,
        connection_density=1
    )
    
    # Generate sample input data (sparse spike train)
    time_steps = 100
    input_data = np.random.random((time_steps, n_inputs)) < 0.5  # 10% spike probability
    
    # Create visualizer
    visualizer = SRNNVisualizer(layer, input_data)
    
    return visualizer

# Uncomment to run demo:
if __name__ == "__main__":
    viz = create_demo_visualization()
    anim = viz.animate(interval=200)  # 200ms between frames