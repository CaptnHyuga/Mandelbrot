import torch
import json
from utils import log_message

class SIRENLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, w0=30.0):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
        self.w0 = w0
        torch.nn.init.uniform_(self.linear.weight, -1.0 / in_dim, 1.0 / in_dim)
    
    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))

class SIRENNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.net = torch.nn.Sequential(
            SIRENLayer(num_inputs, num_hidden),
            SIRENLayer(num_hidden, num_hidden),
            torch.nn.Linear(num_hidden, num_outputs),
            torch.nn.Sigmoid()
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
    
    def forward(self, x):
        return self.net(x)
    
    def backward(self, x, y):
        self.optimizer.zero_grad()
        output = self.forward(x)
        loss = torch.mean((output - y) ** 2)
        loss.backward()
        self.optimizer.step()
    
    def save_state_dict(self):
        # Convert each tensor in state_dict to a list for JSON serialization.
        state = self.state_dict()
        serializable = {k: state[k].cpu().numpy().tolist() for k in state}
        return {"model_state": serializable}
    
    def load_from_dict(self, state_dict):
        # Convert lists back into tensors.
        loaded_state = {}
        for k, v in state_dict["model_state"].items():
            loaded_state[k] = torch.tensor(v)
        # Get current state dict.
        current_state = self.state_dict()
        # Update only keys that exist in the current state.
        for key in current_state.keys():
            if key in loaded_state:
                current_state[key] = loaded_state[key]
        # Load the updated state dict (allowing missing keys).
        self.load_state_dict(current_state, strict=False)
    
    def load(self, filepath):
        with open(filepath, 'r') as f:
            state = json.load(f)
        # Remove extra keys that are not part of the model state.
        state.pop("generation", None)
        state.pop("loss", None)
        if 'model_state' in state:
            self.load_from_dict(state)
        else:
            # Fallback: assume state is the raw state dict.
            current_state = self.state_dict()
            for key in current_state.keys():
                if key in state:
                    current_state[key] = torch.tensor(state[key])
            self.load_state_dict(current_state, strict=False)
