# mandelbrot_generator.py
import torch

def calculate_mandelbrot(c, max_iter=20):
    """
    Calculate the number of iterations before the sequence escapes,
    given a complex input c. Returns an integer iteration count.
    """
    z = torch.zeros_like(c)
    for n in range(max_iter):
        z = z * z + c
        if (z.abs() > 2).any():
            return n
    return max_iter  # Return max_iter if it doesn't escape

def generate_mandelbrot(width, height, x_min, x_max, y_min, y_max):
    """
    Generate a normalized Mandelbrot set of shape (width, height).
    Returns a torch.Tensor of floats in [0,1].
    """
    x = torch.linspace(x_min, x_max, steps=width)
    y = torch.linspace(y_min, y_max, steps=height)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    c = xx + 1j * yy

    mandelbrot = torch.empty_like(xx)
    for i in range(width):
        for j in range(height):
            mandelbrot[i, j] = calculate_mandelbrot(c[i, j].clone().detach(),
                                                    max_iter=20)
    max_value = mandelbrot.max()
    return mandelbrot / max_value
