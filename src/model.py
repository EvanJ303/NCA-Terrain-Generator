import torch
import torch.nn as nn

class NCA(nn.Module):
    # Neural Cellular Automaton model with perception and state update.
    def __init__(self, state_channels, hidden_channels, alpha_channel, update_prob):
        super().__init__()

        self.alpha_channel = alpha_channel
        self.update_prob = update_prob

        # Perception layer extracts local spatial features from state.
        self.perception = nn.Conv2d(
            state_channels,
            hidden_channels, 
            kernel_size=3,
            padding=1,
            bias=False
        )
        
        # Update network computes a state delta from perceived features.
        self.update = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, state_channels, kernel_size=1),
        )
    
    def forward(self, state):
        # Cells with a low alpha value are treated as inactive and cannot
        # contribute to their neighbors' updates.

        alpha = torch.sigmoid(state[:, self.alpha_channel:self.alpha_channel + 1, :, :])
        active_mask = (alpha >= 0.1).float().to(state.device)
        masked_state = state * active_mask

        # Compute local perception and update delta for one step.
        features = self.perception(masked_state)
        delta = self.update(features)

        if self.training:
            # Random mask drops updates with probability 1 - update_prob.
            prob_mask = (
                torch.rand(
                    state.shape[0],
                    1,
                    state.shape[2],
                    state.shape[3]
                ) < self.update_prob
            ).float().to(state.device)

            delta = delta * prob_mask

        # Prevent inactive cells from updating themselves or influencing others.
        delta = delta * active_mask

        # Apply the masked delta to the current state.
        return state + delta