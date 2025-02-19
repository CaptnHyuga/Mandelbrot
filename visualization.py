import numpy as np
import vispy.scene
from vispy.scene import visuals
import torch
from config import TARGET_ALPHA, SHOW_NETWORK_OUTPUT

class VispyPlot:
    def __init__(self, x_vals, y_vals, target_z=None):
        self.canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = vispy.scene.TurntableCamera(up='z', fov=60)
        
        # Create surface plot for network output. Color is blue if shown.
        init_color = (0.5, 0.5, 1, 1) if SHOW_NETWORK_OUTPUT else (0.5, 0.5, 1, 0)
        self.surface_plot = visuals.SurfacePlot(x=x_vals, y=y_vals, z=np.zeros_like(x_vals),
                                                shading='smooth',
                                                color=init_color)
        self.view.add(self.surface_plot)
        
        # Create target surface plot if provided with configurable transparency.
        if target_z is not None:
            self.target_surface = visuals.SurfacePlot(x=x_vals, y=y_vals, z=target_z,
                                                      shading='smooth',
                                                      color=(1.0, 0.5, 0.5, TARGET_ALPHA))
            self.view.add(self.target_surface)
        else:
            self.target_surface = None

    def update_surface(self, tensor_values):
        # Update network output surface.
        if torch.is_tensor(tensor_values):
            array_values = tensor_values.detach().cpu().numpy()
        else:
            array_values = tensor_values
        self.surface_plot.set_data(z=array_values)
    
    def set_network_visibility(self, flag: bool):
        # Toggle network output (blue) visibility.
        color = (0.5, 0.5, 1, 1) if flag else (0.5, 0.5, 1, 0)
        self.surface_plot.set_data(color=color)
    
    def set_target_visibility(self, flag: bool):
        # Toggle target surface (red) visibility.
        if self.target_surface is not None:
            color = (1.0, 0.5, 0.5, TARGET_ALPHA) if flag else (1.0, 0.5, 0.5, 0)
            self.target_surface.set_data(color=color)

    def update_target(self, target_tensor):
        # Update target surface if available.
        if self.target_surface is not None:
            if torch.is_tensor(target_tensor):
                array_values = target_tensor.detach().cpu().numpy()
            else:
                array_values = target_tensor
            self.target_surface.set_data(z=array_values)

    def close(self):
        self.canvas.close()
