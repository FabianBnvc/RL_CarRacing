# %%
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# %%
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 6, kernel_size=3, stride=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(12 * 6 * 6, 216)
        self.fc2 = nn.Linear(216, num_actions)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# %%
class Agent:
    def __init__(self, action_space, frame_stack_num, memory_size, gamma, epsilon, epsilon_min, epsilon_decay, learning_rate):
        self.action_space = action_space
        self.frame_stack_num = frame_stack_num
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model().to(self.device)
        self.target_model = self.build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.update_target_model()

    def build_model(self):
        return DQN(input_shape=(self.frame_stack_num, 96, 96), num_actions=self.action_space.n)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() > self.epsilon:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                act_values = self.model(state)
            action_index = torch.argmax(act_values[0]).item()
        else:
            action_index = random.randrange(self.action_space.n)
        return action_index

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        train_state = []
        train_target = []
        for state, action_index, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            target = self.model(state.unsqueeze(0)).detach().cpu().numpy()[0]
            if done:
                target[action_index] = reward
            else:
                t = self.target_model(next_state.unsqueeze(0)).detach().cpu().numpy()[0]
                target[action_index] = reward + self.gamma * np.amax(t)
            train_state.append(state)
            train_target.append(target)

        train_state = torch.stack(train_state)
        train_target = torch.FloatTensor(np.array(train_target)).to(self.device)
        self.optimizer.zero_grad()
        predictions = self.model(train_state)
        loss = self.loss_fn(predictions, train_target)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name, map_location=self.device))
        self.update_target_model()

    def save(self, name):
        torch.save(self.target_model.state_dict(), name)


