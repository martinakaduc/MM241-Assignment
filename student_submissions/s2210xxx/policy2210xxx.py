from student_submissions.s2210xxx.AC2 import *  # Importing necessary modules for Actor-Critic model and environment interaction
import numpy as np  # Importing numpy for numerical operations
import matplotlib.pyplot as plt  # Importing matplotlib for plotting learning curves
from policy import Policy  # Importing base Policy class
import torch  # Importing PyTorch for deep learning functionality
import torch.nn as nn  # Importing neural network functionalities from PyTorch
import torch.optim as optim  # Importing optimization functions from PyTorch
from torch.distributions import Categorical  # Importing Categorical distribution for action sampling


class Policy2210xxx(Policy):
    def __init__(self, env=None, load_check_pontis=False, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        # Initializing key parameters for the model
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set device (GPU if available)
        self.gamma = 0.99  # Discount factor for reward
        self.n_epochs = 5  # Number of training epochs
        self.batch_size = 10  # Size of each batch for training
        self.learning_rate = 0.0003  # Learning rate for the optimizer

        self.customObs = []  # Initialize empty list for custom observations
        self.obsInfo = dict()  # Dictionary to store observation info
        self.filter_out = 1  # Filtering condition (used in Actor-Critic)
        self.actor_critic = A2CActorCritic(self.filter_out, self.learning_rate)  # Instantiate Actor-Critic model
        self.env = env  # Environment object
        if load_check_pontis or env is None:
            self.actor_critic.load_checkpoint()  # Load saved model checkpoint if necessary
        self.memory = A2CAgent(self.batch_size)  # Memory object for A2C
        self.old_action = None  # To store previous action for inference
        
        if policy_id == 1:
            pass
        elif policy_id == 2:
            pass


    def remember(self, state, action, probs, vals, reward, done):
        # Store current step's memory (state, action, reward, etc.)
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        # Save model weights to a checkpoint
        print('... saving models ...')
        self.actor_critic.save_checkpoint()

    def load_models(self):
        # Load model weights from a checkpoint
        print('... loading models ...')
        self.actor_critic.load_checkpoint()

    def choose_action(self, observation):
        # Given an observation, choose an action based on the actor-critic model
        observation = np.array([observation], dtype=float)
        state = torch.tensor(observation, dtype=torch.float).to(self.actor_critic.device)  # Convert observation to tensor
        dist, value = self.actor_critic(state)  # Get action distribution and value from actor-critic model
        action = dist.sample()  # Sample action from the distribution

        probs = torch.squeeze(dist.log_prob(action)).item()  # Get log probability of the action
        action = torch.squeeze(action).item()  # Convert action to a scalar
        value = torch.squeeze(value).item()  # Convert value to a scalar
        return action, probs, value  # Return chosen action, log probability, and value

    def get_action(self, obs, info):
        # Main method to decide the next action to take
        if len(self.customObs) == 0:
            self.resetCustomObservation(obs)  # Reset custom observation if it's the first step
            self.stepInfo()  # Print step info
        else:
            self.updateCustomObservation(obs, self.old_action)  # Update custom observation using previous action
        with torch.no_grad():  # No gradient computation needed during inference
            state = np.array([self.customObs], dtype=float)  # Convert custom observation to array
            state = torch.tensor(state, dtype=torch.float).to(self.actor_critic.device)  # Convert to tensor
            dist, value = self.actor_critic(state)  # Get action distribution and value
            action = torch.argmax(dist.probs).item()  # Select the action with the highest probability
        convertedAction = self.convertAction(action)  # Convert action to a suitable format for environment
        reward, done = self.getCustomReward(obs, action)  # Get reward and done status for the selected action
        self.old_action = action  # Save the current action for the next step
        if done:
            self.customObs = []  # Reset custom observation if the episode is done
        return convertedAction  # Return the action to be executed

    def stepInfo(self, postInfo=False):
        # Print information about the current step (e.g., products remaining)
        count = quantity = 0
        for p in self.obsInfo["productInfo"]:
            count += 1
            quantity += p[3]
        if postInfo:
            print(f'Remaining products: {quantity}, intactStock: {len(self.obsInfo["intact"])}')
        else:
            print(f'---------------------------------------',
                  f'\nTotal product types: {count}, products demand: {quantity}, ')

    def train(self, total_timestep, timestep_per_game=200):
        # Training loop for the agent
        figure_file = 'rewards.png'  # File to save learning curve plot
        best_score = -10000  # Initialize best score
        score_history = []  # List to store scores across episodes
        learn_iters = 0  # Number of learning iterations
        avg_score = 0  # Average score across episodes
        n_steps = 0  # Total steps taken
        save_freq = total_timestep // 10  # Frequency of saving model
        cur_save_step = save_freq  # Current save step
        while n_steps < total_timestep:
            observation, _ = self.env.reset()  # Reset environment to start a new episode
            self.resetCustomObservation(observation)  # Reset custom observation
            done = False  # Initialize done flag
            score = 0  # Initialize score for this episode
            game_timestep = 0  # Initialize game timestep
            self.stepInfo()  # Print step info
            while not done:
                action, prob, val = self.choose_action(self.customObs)  # Choose action
                convertedAction = self.convertAction(action)  # Convert action to suitable format
                reward, _ = self.getCustomReward(observation, action)  # Get reward for the action
                observation_, _, done, _, info = self.env.step(convertedAction)  # Take action in the environment
                n_steps += 1
                game_timestep += 1
                score += reward  # Update score
                self.remember(self.customObs, action, prob, val, reward, done)  # Store memory
                self.updateCustomObservation(observation_, action)  # Update custom observation
                if n_steps % self.batch_size == 0:
                    self.learn()  # Learn from the batch of experiences
                    learn_iters += 1
                observation = observation_  # Update observation
            score_history.append(score)  # Store score for this episode
            avg_score = np.mean(score_history[-100:])  # Calculate moving average of the score
            normalized_reward = score / game_timestep  # Normalize the reward
            if normalized_reward > best_score:
                best_score = normalized_reward
                self.save_models()  # Save model if best score
            if n_steps > cur_save_step:
                cur_save_step += save_freq
                self.save_models()  # Save model at specified frequency
            self.stepInfo(True)  # Print post-episode info
            print('Game step', game_timestep, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                  'time_steps', n_steps, 'learning_steps', learn_iters)  # Print stats
        x = [i + 1 for i in range(len(score_history))]  # Prepare x-axis values for the plot
        self.save_models()  # Save final model
        self.plot_learning_curve(x, score_history, figure_file)  # Plot the learning curve

    def learn(self):
        # Perform learning using A2C algorithm
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = \
                self.memory.generate_batches()  # Generate batches from memory

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)  # Initialize advantage array

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor_critic.device)  # Convert advantage to tensor

            values = torch.tensor(values).to(self.actor_critic.device)  # Convert values to tensor
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor_critic.device)  # Convert states
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor_critic.device)  # Convert old probabilities
                actions = torch.tensor(action_arr[batch]).to(self.actor_critic.device)  # Convert actions

                dist, critic_value = self.actor_critic(states)  # Get new distribution and critic value
                log_prob = dist.log_prob(actions)  # Compute log probability
                ratio = torch.exp(log_prob - old_probs)  # Compute ratio
                clipped_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)  # Clip the ratio to avoid large updates
                loss1 = -advantage[batch] * ratio  # Policy loss
                loss2 = -advantage[batch] * clipped_ratio  # Clipped policy loss
                actor_loss = torch.max(loss1, loss2).mean()  # Final actor loss

                critic_loss = (critic_value - advantage[batch]).pow(2).mean()  # Critic loss
                total_loss = actor_loss + 0.5 * critic_loss  # Total loss

                self.actor_critic.optimizer.zero_grad()  # Zero gradients
                total_loss.backward()  # Backpropagate loss
                self.actor_critic.optimizer.step()  # Update model parameters
        self.memory.clear_memory()  # Clear memory after learning
