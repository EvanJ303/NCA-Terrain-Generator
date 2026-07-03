import torch

def generate_images(height, width, batch_size, model, num_steps):
    # Create an initial latent state for each image in the batch.
    device = next(model.parameters()).device

    states = 0.02 * torch.randn((batch_size, model.state_channels, height, width), device=device)

    # Run the cellular update rule for the requested number of steps.
    for _ in range(num_steps):
        states = model(states)
    
    # Convert the evolved state to RGB values for the final image.
    rgb = torch.tanh(states[:, :3, :, :])
    return rgb