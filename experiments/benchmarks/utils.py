import torch

def get_steering_hook(steering_vector, multiplier):
    """Creates a forward hook to add the steering vector to the residual stream."""
    def hook(module, inputs, outputs):
        is_tuple = isinstance(outputs, tuple)
        hidden_states = outputs[0] if is_tuple else outputs
        
        sv = steering_vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
        modified_hidden_states = hidden_states + (multiplier * sv)
        
        if is_tuple:
            return (modified_hidden_states,) + outputs[1:]
        return modified_hidden_states
        
    return hook

def attach_steering_vector(model, vector_path, layer_idx, multiplier):
    """Loads a steering vector and attaches it to the specified model layer."""
    print(f"Loading steering vector from {vector_path}...")
    steering_vector = torch.load(vector_path, map_location=model.device)
    
    print(f"Attaching hook to layer {layer_idx} with multiplier {multiplier}...")
    target_layer = model.model.layers[layer_idx]
    hook_handle = target_layer.register_forward_hook(get_steering_hook(steering_vector, multiplier))
    
    return hook_handle
