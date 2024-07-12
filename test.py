import random
import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Float32
from sensor_msgs.msg import LaserScan


class Env:

    def __init__(self, robot, max_episode_lenght=1000, final_state=0.15):
        self.next_state = None
        self.episode_return = 0
        self.episode_step = 0

        self.robot = robot
        self.done = False
        self.episode_step = 0
        self.max_episode_lenght = max_episode_lenght
        self.final_state = final_state
        self.info = {}
        self.info["max_legnth_reached"] = False
        self.info["reached"] = False

        self.observation_space = np.ndarray(
            1,
        )
        self.action_space = np.ndarray(
            1,
        )
        self.state = self.robot.read_state()

    def step(self, action):
        self.robot.do(action)
        self.next_state = self.robot.read_state()
        self.episode_step += 1
        self.reward = 0

        if np.sum(self.next_state) <= self.final_state:
            self.info["reached"] = True

        if self.episode_step == self.max_episode_lenght:
            self.info["max_legnth_reached"] = True

        if self.episode_step == self.max_episode_lenght or self.info["reached"]:
            self.done = True
        else:
            self.reward = -np.sum(self.state)
            self.episode_return += self.reward

        self.state = self.next_state

        return self.next_state, self.reward, self.done, self.info

    def reset(self):
        self.state = self.robot.read_state()
        self.episode_return = 0
        self.episode_step = 0
        self.done = False
        self.info.clear()
        self.info["max_legnth_reached"] = False
        self.info["reached"] = False

        return self.state


class MyNode(Node):

    def __init__(self, subscribing_topic, publishing_topic):
        super().__init__("my_node")
        self.state = np.array([0], dtype=np.float32)
        self.publisher_ = self.create_publisher(Float32, publishing_topic, 10)
        self.subscription = self.create_subscription(
            LaserScan, subscribing_topic, self.read_state, 10
        )
        self.subscription  # prevent unused variable warning

    def read_state(self, msg):
        try:
            # TODO translate msg in state
            self.state[0] = msg.ranges[0]
            print(f"i recieved: {self.state}")
        except:
            print("state not update!")

    def send_message(self, msg):
        try:
            self.publisher_.publish(msg)
            self.get_logger().info('Publishing: "%s"' % msg.data)
        except Exception as e:
            print(e)


class Robot:

    def __init__(self, name, node_ros):
        self.name = name
        self.node_ros = node_ros
        self.state = self.node_ros.state
        print(self)

    def do(self, action):
        msg = Float32()
        msg.data = action * 1.0
        self.node_ros.send_message(msg)

    def read_state(self):
        rclpy.spin_once(self.node_ros)
        self.state = self.node_ros.state
        return self.state

    def __str__(self):
        return f"robot named:'{self.name}'"


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PPO:
    def __init__(
        self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2, epochs=10
    ):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_returns(self, rewards, dones, next_value):
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        return returns

    def train(self, memory):
        states, actions, log_probs, rewards, dones = zip(*memory)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs = torch.FloatTensor(log_probs)
        returns = torch.FloatTensor(
            self.compute_returns(rewards, dones, self.critic(states[-1]))
        ).detach()

        for _ in range(self.epochs):
            values = self.critic(states).squeeze()
            advantages = returns - values

            new_probs = self.actor(states)
            dist = torch.distributions.Categorical(new_probs)
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - log_probs)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                * advantages
            )
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, returns)

            loss = actor_loss + 0.5 * critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


def train_ppo(env, episodes=1000):
    ppo = PPO(
        state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0]
    )
    all_rewards = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        memory = []

        while True:
            action, log_prob = ppo.select_action(state)
            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, log_prob, reward, done))
            state = next_state
            episode_reward += reward

            if done:
                ppo.train(memory)
                all_rewards.append(episode_reward)
                print(f"Episode: {episode}, Reward: {episode_reward}")
                break

    env.close()
    return all_rewards


rclpy.init()

my_node = MyNode(
    subscribing_topic="demo/laser/out", publishing_topic="irobot02/cmd_vel"
)
robot = Robot("bot", my_node)
env = Env(robot)

train_ppo(env)
