import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# Define a Convolutional Neural Network (CNN) for filtering inputs
class FilterCNN(nn.Module):
    def __init__(self, out_dim=3):
        super(FilterCNN, self).__init__()
        # Define the layers: two convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(3, out_dim, kernel_size=1)  # First convolutional layer
        self.bn1 = nn.LazyBatchNorm2d()  # Batch normalization for the first layer
        self.conv2 = nn.Conv2d(out_dim, 1, kernel_size=1)  # Second convolutional layer
        self.bn2 = nn.LazyBatchNorm2d()  # Batch normalization for the second layer
        # Set the device to GPU if available, else CPU
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)  # Move model to the device

    def forward(self, x, mask, bias_mask=None):
        # Concatenate the input tensor with mask and bias_mask along the channels dimension
        x = torch.cat([x.unsqueeze(dim=1), mask.unsqueeze(dim=1), bias_mask.unsqueeze(dim=1)], dim=1)
        # Pass through the first convolution and batch normalization layers
        x = self.conv1(x)
        x = self.bn1(x)
        # Pass through the second convolution and batch normalization layers
        x = self.conv2(x)
        x = self.bn2(x)
        return x


# Define the Actor-Critic model for reinforcement learning
class A2CActorCritic(nn.Module):
    def __init__(self, filter_out, alpha, chkpt_dir='./s22110xxx/models'):
        super(A2CActorCritic, self).__init__()

        # Xóa checkpoint và model PPO cũ
        self.actor = nn.Conv2d(1, 1, kernel_size=1)  # Thay thế bằng mạng actor đơn giản
        self.filter = FilterCNN(filter_out)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # Extract stock, bias_mask, and mask from input state
        stock, bias_mask, mask = state[:, 0], state[:, 1], state[:, 2]
        # Get the output from the filter CNN
        f = self.filter(stock, mask, bias_mask)
        # Process output through actor to get action probabilities
        x = self.actor(f)
        # Mask out invalid positions (set them to a very low probability)
        x = x.masked_fill(mask == 0, -1e9)  
        x = torch.flatten(x, start_dim=-3)  # Flatten the tensor for softmax
        x = torch.softmax(x, dim=-1)  # Apply softmax for probability distribution
        # Create a Categorical distribution from the probabilities
        dist = Categorical(x)
        # Get the critic's value (state value)
        value = torch.max(f)  # Use max here for simplicity
        return dist, value
    

    def save_checkpoint(self):
        torch.save(self.state_dict(), 'a2c_model_checkpoint.pth')

    def load_checkpoint(self):
        self.load_state_dict(torch.load('a2c_model_checkpoint.pth'))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


# Define the A2C (Advantage Actor-Critic) agent model for reinforcement learning
class A2CAgent(nn.Module):
    def __init__(self, input_dim, action_dim, alpha):
        super(A2CAgent, self).__init__()
        # Define the actor network (policy) with two fully connected layers
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128),  # First layer: input_dim to 128
            nn.ReLU(),  # ReLU activation function
            nn.Linear(128, action_dim),  # Second layer: 128 to action_dim (number of actions)
            nn.Softmax(dim=-1)  # Softmax to get a probability distribution over actions
        )
        # Define the critic network (value function) with two fully connected layers
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 128),  # First layer: input_dim to 128
            nn.ReLU(),  # ReLU activation function
            nn.Linear(128, 1)  # Second layer: 128 to 1 (single value for value function)
        )
        # Adam optimizer for the model parameters
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        # Set the device to GPU if available, else CPU
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)  # Move model to the device

    def forward(self, state):
        # Convert state to tensor and move it to the device
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        # Get action probabilities and value estimates from the actor and critic networks
        policy_dist = self.actor(state)
        value = self.critic(state)
        return policy_dist, value

    def choose_action(self, state):
        # Choose action based on current state
        policy_dist, _ = self.forward(state)
        dist = Categorical(policy_dist)  # Create a categorical distribution from the policy
        action = dist.sample()  # Sample an action from the distribution
        return action.item(), dist.log_prob(action)  # Return the action and its log probability

    def update(self, states, actions, rewards, log_probs, values, gamma):
        # Compute returns (discounted future rewards)
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # Convert values and log_probs to tensors
        values = torch.stack(values).squeeze(-1)
        log_probs = torch.stack(log_probs)

        # Compute advantages (difference between returns and value estimates)
        advantages = returns - values
        # Compute actor loss (policy loss)
        actor_loss = -(log_probs * advantages.detach()).mean()
        # Compute critic loss (value loss)
        critic_loss = advantages.pow(2).mean()

        # Total loss is a combination of actor and critic losses
        loss = actor_loss + 0.5 * critic_loss
        # Zero the gradients, perform backpropagation, and update the parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
