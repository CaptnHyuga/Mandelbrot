import torch
import json
from config import NUM_HIDDEN, NUM_OUTPUTS
from utils import log_message

# Activation functions
def sigmoid(x):
    return torch.sigmoid(x)

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return torch.tanh(x)

def tanh_derivative(x):
    return 1 - torch.tanh(x) ** 2

def relu(x):
    return torch.relu(x)

def relu_derivative(x):
    return torch.where(x > 0, 1, 0)

def leaky_relu(x, alpha=0.01):
    return torch.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    grad = torch.ones_like(x)
    grad[x < 0] = alpha
    return grad

def safe_stats(arr):
    if not torch.all(torch.isfinite(arr)):
        return "Invalid (NaN or Inf detected)"
    return f"Min: {torch.min(arr):.4f}, Max: {torch.max(arr):.4f}, Mean: {torch.mean(arr):.4f}, Std: {torch.std(arr, unbiased=False):.4f}"

def clip_by_norm(grad, max_norm):
    norm = torch.norm(grad)
    if (norm > max_norm):
        return grad * (max_norm / norm)
    return grad

def normalize_array(x, eps=1e-8):
    mean = torch.mean(x, dim=0, keepdim=True)
    std = torch.std(x, dim=0, keepdim=True, unbiased=False) + eps
    return torch.clamp((x - mean) / std, -3, 3)

class BatchNorm:
    def __init__(self, dim, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.normalized = None
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
        self.momentum = momentum

    def forward(self, x, training=True):
        if not torch.all(torch.isfinite(x)):
            print("Warning: Input to BatchNorm contains NaNs or Infs. Returning zeroed output.")
            self.normalized = torch.zeros_like(x)
            self.var = torch.zeros_like(self.running_var)
            return torch.zeros_like(x)
        
        if training:
            mean = torch.mean(x, dim=0)
            var = torch.var(x, dim=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        self.var = var
        self.normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * self.normalized + self.beta

    def backward(self, grad_output, learning_rate):
        if not torch.all(torch.isfinite(grad_output)):
            print("Warning: grad_output to BatchNorm contains NaNs or Infs. Returning zeroed gradient.")
            return torch.zeros_like(grad_output)
        
        if self.normalized is None:
            print("Warning: BatchNorm.backward called without forward pass. Returning zeroed gradient.")
            return torch.zeros_like(grad_output)
        
        grad_gamma = torch.sum(grad_output * self.normalized, dim=0)
        grad_beta = torch.sum(grad_output, dim=0)
        
        self.gamma -= learning_rate * grad_gamma
        self.beta -= learning_rate * grad_beta
        
        N = grad_output.shape[0]
        grad_input = (1.0 / (N * torch.sqrt(self.var + self.eps))) * (
            N * grad_output * self.gamma -
            torch.sum(grad_output * self.gamma, dim=0) -
            self.normalized * torch.sum(grad_output * self.gamma * self.normalized, dim=0)
        )
        return grad_input

class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.normalized = None
    
    def forward(self, x):
        self.mean = torch.mean(x, dim=-1, keepdim=True)
        self.var = torch.var(x, dim=-1, keepdim=True)
        self.std = torch.sqrt(self.var + self.eps)
        self.normalized = (x - self.mean) / self.std
        return self.gamma * self.normalized + self.beta
    
    def backward(self, grad_output, learning_rate):
        if self.normalized is None:
            raise ValueError("Forward pass must be called before backward pass")
            
        grad_gamma = torch.sum(grad_output * self.normalized, dim=0)
        grad_beta = torch.sum(grad_output, dim=0)
        
        self.gamma -= learning_rate * grad_gamma
        self.beta -= learning_rate * grad_beta
        
        N = grad_output.shape[0]
        grad_input = (1.0 / (N * self.std)) * (
            N * grad_output * self.gamma -
            torch.sum(grad_output * self.gamma, dim=0) -
            self.normalized * torch.sum(grad_output * self.gamma * self.normalized, dim=0)
        )
        return grad_input

class SimpleNetwork:
    """
    A simple feedforward neural network with optional dropout and batch normalization.
    """
    def __init__(self, num_inputs, num_hidden, num_outputs, dropout_rate=0.2):
        # Weight initialization
        def init_weights(shape):
            limit = (2.0 / shape[0])**0.5
            return torch.normal(0, limit, size=shape)
        
        self.weights1 = init_weights((num_inputs, num_hidden))
        self.biases1 = torch.zeros(num_hidden)
        self.weights2 = init_weights((num_hidden, num_outputs))
        self.biases2 = torch.zeros(num_outputs)
        
        self.batch_norm1 = BatchNorm(num_hidden, momentum=0.99)  # Adjusted momentum
        self.batch_norm2 = BatchNorm(num_outputs, momentum=0.99)  # Adjusted momentum
        
        self.dropout_rate = dropout_rate
        self.training = True
        self.l2_lambda = 0.001 # L2 regularization strength

    def forward(self, x, training=None):
        """
        Forward pass through the network. If training is True,
        dropout and batch normalization are active.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Scale input data
        x = x / 5.0  # Adjust scaling factor as needed
        
        is_training = self.training if training is None else training
        
        x = (x - torch.mean(x, dim=0)) / (torch.std(x, dim=0) + 1e-8)
        
        self.layer1_input = torch.matmul(x, self.weights1) + self.biases1
        self.layer1_norm = self.batch_norm1.forward(self.layer1_input, training=is_training)
        self.hidden_layer = leaky_relu(self.layer1_norm)
        
        if is_training:
            self.hidden_dropout = self.dropout(self.hidden_layer)
        else:
            self.hidden_dropout = self.hidden_layer
        
        self.layer2_input = torch.matmul(self.hidden_dropout, self.weights2) + self.biases2
        self.layer2_norm = self.batch_norm2.forward(self.layer2_input, training=is_training)
        self.output_layer = sigmoid(self.layer2_norm)
        
        # Log layer inputs and outputs
        log_message("DEBUG", f"Layer1 Input: {safe_stats(self.layer1_input)}")
        log_message("DEBUG", f"Layer1 Norm: {safe_stats(self.layer1_norm)}")
        log_message("DEBUG", f"Hidden Layer: {safe_stats(self.hidden_layer)}")
        log_message("DEBUG", f"Layer2 Input: {safe_stats(self.layer2_input)}")
        log_message("DEBUG", f"Layer2 Norm: {safe_stats(self.layer2_norm)}")
        log_message("DEBUG", f"Output Layer: {safe_stats(self.output_layer)}")
        
        return self.output_layer
    
    def backward(self, x, y, learning_rate=0.01):
        """
        Backpropagation step with optional gradient clipping and L2 regularization.
        """
        output = self.forward(x)
        
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)
        output_delta = self.batch_norm2.backward(output_delta, learning_rate)
        
        hidden_error = torch.matmul(output_delta, self.weights2.T)
        hidden_delta = hidden_error * leaky_relu_derivative(self.hidden_layer)
        hidden_delta = self.batch_norm1.backward(hidden_delta, learning_rate)
        
        # Log weights, biases, and gradients
        log_message("DEBUG", f"Weights1: {safe_stats(self.weights1)}")
        log_message("DEBUG", f"Biases1: {safe_stats(self.biases1)}")
        log_message("DEBUG", f"Weights2: {safe_stats(self.weights2)}")
        log_message("DEBUG", f"Biases2: {safe_stats(self.biases2)}")
        log_message("DEBUG", f"Output Delta: {safe_stats(output_delta)}")
        log_message("DEBUG", f"Hidden Delta: {safe_stats(hidden_delta)}")
        
        # Scale down weight and bias updates
        weight2_grad = torch.matmul(self.hidden_layer.T, output_delta) + self.l2_lambda * self.weights2
        bias2_grad = torch.sum(output_delta, dim=0)
        weight1_grad = torch.matmul(x.T, hidden_delta) + self.l2_lambda * self.weights1
        bias1_grad = torch.sum(hidden_delta, dim=0)

        # Clip gradients
        weight2_grad = clip_by_norm(weight2_grad, 1.0)
        bias2_grad = clip_by_norm(bias2_grad, 1.0)
        weight1_grad = clip_by_norm(weight1_grad, 1.0)
        bias1_grad = clip_by_norm(bias1_grad, 1.0)
        
        self.weights2 -= learning_rate * 0.1 * weight2_grad
        self.biases2 -= learning_rate * 0.1 * bias2_grad
        self.weights1 -= learning_rate * 0.1 * weight1_grad
        self.biases1 -= learning_rate * 0.1 * bias1_grad

    def dropout(self, x):
        if self.training and self.dropout_rate > 0:
            mask = torch.bernoulli(torch.full(x.shape, 1 - self.dropout_rate))
            return x * mask / (1 - self.dropout_rate)
        return x

    def save_state_dict(self):
        return {
            'weights1': self.weights1.cpu().numpy().tolist(),
            'biases1': self.biases1.cpu().numpy().tolist(),
            'weights2': self.weights2.cpu().numpy().tolist(),
            'biases2': self.biases2.cpu().numpy().tolist(),
            'batch_norm1_gamma': self.batch_norm1.gamma.cpu().numpy().tolist(),
            'batch_norm1_beta': self.batch_norm1.beta.cpu().numpy().tolist(),
            'batch_norm1_running_mean': self.batch_norm1.running_mean.cpu().numpy().tolist(),
            'batch_norm1_running_var': self.batch_norm1.running_var.cpu().numpy().tolist(),
            'batch_norm2_gamma': self.batch_norm2.gamma.cpu().numpy().tolist(),
            'batch_norm2_beta': self.batch_norm2.beta.cpu().numpy().tolist(),
            'batch_norm2_running_mean': self.batch_norm2.running_mean.cpu().numpy().tolist(),
            'batch_norm2_running_var': self.batch_norm2.running_var.cpu().numpy().tolist(),
            'model_config': {
                'num_inputs': self.weights1.shape[0],
                'num_hidden': self.weights1.shape[1],
                'num_outputs': self.weights2.shape[1],
                'dropout_rate': self.dropout_rate
            }
        }

    def load_state_dict(self, state_dict):
        try:
            self.weights1 = torch.tensor(state_dict['weights1'])
            self.biases1 = torch.tensor(state_dict['biases1'])
            self.weights2 = torch.tensor(state_dict['weights2'])
            self.biases2 = torch.tensor(state_dict['biases2'])
            self.batch_norm1.gamma = torch.tensor(state_dict['batch_norm1_gamma'])
            self.batch_norm1.beta = torch.tensor(state_dict['batch_norm1_beta'])
            self.batch_norm1.running_mean = torch.tensor(state_dict['batch_norm1_running_mean'])
            self.batch_norm1.running_var = torch.tensor(state_dict['batch_norm1_running_var'])
            self.batch_norm2.gamma = torch.tensor(state_dict['batch_norm2_gamma'])
            self.batch_norm2.beta = torch.tensor(state_dict['batch_norm2_beta'])
            self.batch_norm2.running_mean = torch.tensor(state_dict['batch_norm2_running_mean'])
            self.batch_norm2.running_var = torch.tensor(state_dict['batch_norm2_running_var'])
            
            # Verify model configuration
            config = state_dict.get('model_config', {})
            if (config.get('num_inputs') != self.weights1.shape[0] or
                config.get('num_hidden') != self.weights1.shape[1] or
                config.get('num_outputs') != self.weights2.shape[1]):
                raise ValueError("Model configuration mismatch")
                
        except Exception as e:
            raise ValueError(f"Error loading state dict: {e}")

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.save_state_dict(), f)

    def load(self, filepath):
        with open(filepath, 'r') as f:
            state_dict = json.load(f)
        self.load_state_dict(state_dict)
