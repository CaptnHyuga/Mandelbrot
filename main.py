import os
import time
import torch
import tkinter as tk
from tkinter import ttk
from PyQt6 import QtWidgets

# Import user config and modules
from config import width, height, x_min, x_max, y_min, y_max, NUM_ENCODED_DIMS, \
                   NUM_HIDDEN, NUM_OUTPUTS, CHECKPOINT_DIR, VIDEO_DIR, NUM_FREQUENCIES, USE_SIREN
from utils import fourier_encode, save_checkpoint, load_latest_checkpoint, log_message
from function_generator import generate_mandelbrot
from neural_network import SimpleNetwork
from visualization import VispyPlot
from siren_network import SIRENNetwork

class ControlWindow:
    def __init__(self, master):
        self.master = master
        master.title("Neural Network Training Control")
        
        # Progress Bar
        self.progress = ttk.Progressbar(master, length=300, mode='determinate')
        self.progress.pack(pady=10)
        
        # Stats Labels
        self.loss_label = ttk.Label(master, text="Loss: 0.0")
        self.loss_label.pack(pady=5)
        self.gen_label = ttk.Label(master, text="Generation: 0")
        self.gen_label.pack(pady=5)
        
        # Control Buttons - Ensure correct text and command
        self.pause_btn = ttk.Button(master, text="Pause Training", command=self.toggle_pause)
        self.pause_btn.pack(pady=5)
        
        self.viz_btn = ttk.Button(master, text="Hide Visualization", command=self.toggle_visualization)
        self.viz_btn.pack(pady=5)
        
        # Add Stop button
        self.stop_btn = ttk.Button(master, text="Stop Training", command=self.stop_training)
        self.stop_btn.pack(pady=5)
        
        # Bind keyboard shortcuts to correct functions
        master.bind('<Control-p>', lambda e: self.toggle_pause())
        master.bind('<Control-v>', lambda e: self.toggle_visualization())
        
        self.paused = False
        self.visualization_enabled = True

        # Add window close handler
        master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.should_stop = False
    
    def stop_training(self):
        self.should_stop = True
        self.stop_btn.config(state='disabled')
        self.master.quit()
    
    def on_closing(self):
        if not self.should_stop:
            self.stop_training()
        self.master.after(100, self.master.quit)  # Give time for final cleanup
    
    def toggle_pause(self):
        self.paused = not self.paused
        status = "Resume Training" if self.paused else "Pause Training"
        self.pause_btn.config(text=status)
    
    def toggle_visualization(self):
        self.visualization_enabled = not self.visualization_enabled
        status = "Show Visualization" if not self.visualization_enabled else "Hide Visualization"
        self.viz_btn.config(text=status)
        # Inform visualization to hide network output (blue) while keeping target visible.
        # We'll call a method on VispyPlot via a global reference.
        try:
            from config import SHOW_NETWORK_OUTPUT
            new_state = self.visualization_enabled and SHOW_NETWORK_OUTPUT
            global vispy_plot
            vispy_plot.set_network_visibility(new_state)
        except Exception as e:
            print(f"Error toggling visualization: {e}")
    
    def update_stats(self, generation, loss, max_gen):
        self.progress['value'] = (generation / max_gen) * 100
        self.loss_label.config(text=f"Loss: {loss:.6f}")
        self.gen_label.config(text=f"Generation: {generation}")

root = None
control_window = None
update_surface = None

def training_step(generation, network, target_values):
    """
    Main training loop that updates the network while the user
    has not pressed Stop or closed the application.
    """
    global control_window, root  # Now root is available globally
    
    # Initialize loss to avoid UnboundLocalError
    loss = torch.tensor(float('inf'))
    
    try:
        log_message("INFO", f"Starting training at generation: {generation}")
        while not control_window.should_stop:
            if control_window.paused:
                root.update()
                time.sleep(0.1)
                continue
            
            # Training code - use clone().detach() instead of torch.tensor()
            encoded_positions = cached_encoded.clone().detach().to(dtype=torch.float32)
            if encoded_positions.shape[1] != NUM_ENCODED_DIMS:
                raise ValueError(f"Input shape {encoded_positions.shape} incompatible with network")
            
            target_values_tensor = target_values.clone().detach().to(dtype=torch.float32)
            
            output_layer = network.forward(encoded_positions)
            loss = torch.mean((output_layer - target_values_tensor) ** 2)
            current_lr = 0.01 * (0.95 ** (generation / 100))
            
            # Call backward differently based on network type:
            if USE_SIREN:
                network.backward(encoded_positions, target_values_tensor)
            else:
                network.backward(encoded_positions, target_values_tensor, learning_rate=current_lr)

            if generation % 10 == 0:
                update_surface(output_layer.reshape(height, width))
                control_window.update_stats(generation, loss.item(), 5000)

            if generation % 100 == 0:
                save_checkpoint(network, generation, loss.item())

            generation += 1
            root.update()

    except Exception as e:
        log_message("ERROR", f"Error during training at gen {generation}: {e}")
    finally:
        # Save final checkpoint with the last known loss value
        save_checkpoint(network, generation, loss.item())
        log_message("INFO", "Training stopped.")

def main():
    """
    Application entrypoint; sets up the UI, loads data, starts training,
    and runs the event loops until user exit.
    """
    # Create Qt application
    qt_app = QtWidgets.QApplication([])

    global control_window, root
    root = tk.Tk()
    control_window = ControlWindow(root)
    
    # Choose network
    if USE_SIREN:
        network = SIRENNetwork(NUM_ENCODED_DIMS, NUM_HIDDEN, NUM_OUTPUTS)
        log_message("INFO", "Using SIREN-based network.")
    else:
        network = SimpleNetwork(NUM_ENCODED_DIMS, NUM_HIDDEN, NUM_OUTPUTS)
        log_message("INFO", "Using SimpleNetwork.")
    
    # Load checkpoint if available.
    checkpoint, start_generation = load_latest_checkpoint()
    if checkpoint:
        try:
            network.load(os.path.join(CHECKPOINT_DIR, f'checkpoint_{checkpoint["generation"]:06d}.json'))
            print(f"Resuming from generation {start_generation}")
        except Exception as e:
            print(f"Error loading model state: {e}")

    # Compute target fractal (true Mandelbrot) using the generator.
    mandelbrot_values = generate_mandelbrot(width, height, x_min, x_max, y_min, y_max)
    target_values = mandelbrot_values.reshape(-1, 1)
    
    # Additionally, create a numpy version of the target (shape: height x width)
    target_numpy = mandelbrot_values.reshape(height, width).detach().cpu().numpy()
    
    # Convert grid coordinates to torch tensors.
    grid_x = torch.linspace(x_min, x_max, width)
    grid_y = torch.linspace(y_min, y_max, height)
    X, Y = torch.meshgrid(grid_x, grid_y, indexing='xy')
    
    # Create positions tensor and encode with correct dimensions.
    positions = torch.stack((X.flatten(), Y.flatten()), dim=-1)
    encoded_positions = fourier_encode(positions, NUM_FREQUENCIES)
    
    log_message("INFO", f"Encoded positions shape: {encoded_positions.shape}")
    expected_shape = (width * height, NUM_ENCODED_DIMS)
    if encoded_positions.shape != expected_shape:
        log_message("ERROR", f"Shape mismatch details:")
        log_message("ERROR", f"  - Expected shape: {expected_shape}")
        log_message("ERROR", f"  - Actual shape: {encoded_positions.shape}")
        raise ValueError("Encoded positions shape mismatch")
    
    global cached_encoded
    cached_encoded = encoded_positions

    # Convert grid coordinates to numpy arrays for visualization.
    X_np = X.numpy()
    Y_np = Y.numpy()
    
    # Setup Vispy scene. Pass target_numpy to show the real fractal.
    vispy_plot = VispyPlot(X_np, Y_np, target_z=target_numpy)
    global update_surface
    def update_surface(tensor_values):
        vispy_plot.update_surface(tensor_values)
    log_message("INFO", "Application started.")
    
    # Start training.
    training_step(start_generation, network, target_values)
    
    # Run event loops.
    while not control_window.should_stop:
        qt_app.processEvents()
        root.update()

    try:
        vispy_plot.close()
        root.destroy()
    except:
        pass

    log_message("INFO", "Application shutdown.")

if __name__ == '__main__':
    main()
