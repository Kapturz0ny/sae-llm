import torch

def get_steering_hook(steering_vector, multiplier):
    """Creates a forward hook to add the steering vector to the residual stream."""
    def hook(module, inputs, outputs):
        hidden_states = outputs[0]
        modified_hidden_states = hidden_states + (multiplier * steering_vector.to(hidden_states.device))
        return (modified_hidden_states,) + outputs[1:]
    return hook

def attach_steering_vector(model, vector_path, layer_idx, multiplier):
    """Loads a steering vector and attaches it to the specified model layer."""
    print(f"Loading steering vector from {vector_path}...")
    steering_vector = torch.load(vector_path, map_location=model.device)
    
    print(f"Attaching hook to layer {layer_idx} with multiplier {multiplier}...")
    target_layer = model.model.layers[layer_idx]
    hook_handle = target_layer.register_forward_hook(get_steering_hook(steering_vector, multiplier))
    
    return hook_handle
