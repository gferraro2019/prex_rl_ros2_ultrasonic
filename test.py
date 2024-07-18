import random
import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Float32
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import time


class Env:

    def __init__(self, robot, max_episode_lenght=150, final_state=0.5):
        self.next_state = None
        self.state = None

        self.episode_return = 0
        self.episode_step = 0

        self.robot = robot
        self.robot.action_delay = 0.3
        self.done = False
        self.episode_step = 0
        self.max_episode_lenght = max_episode_lenght
        self.final_state = final_state
        self.info = {}
        self.info["max_legnth_reached"] = False
        self.info["reached"] = False

        self.observation_space = np.ndarray(
            6,
        )
        self.action_space = np.ndarray(
            4,
        )

        self.goal_position = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

        while self.robot.node_ros.odom_vector is None:
            rclpy.spin_once(self.robot.node_ros)
        self.state = self.get_state()
        self.target_distance_from_goal = 0.40

        self.reset()

    def get_state(self):
        rclpy.spin_once(self.robot.node_ros)

        position = self.robot.node_ros.odom_vector[:2]
        distance_to_goal = np.linalg.norm(self.goal_position[:2] - position)
        state = np.concatenate(
            [
                position,  # [x, y]
                self.goal_position[:2],  # [goal_x, goal_y]
                [distance_to_goal],  # [distance_to_goal]
                self.robot.node_ros.distance_from_sensor,
            ]
        )
        return state.copy()

    def step(self, action):

        self.robot.do(action)
        time.sleep(self.robot.action_delay)
        # send also stop to interrupt keep last command
        self.robot.do(2)

        self.next_state = self.get_state()
        self.episode_step += 1
        self.reward = 0

        if self.episode_step == self.max_episode_lenght or self.info["reached"]:
            self.done = True
        else:
            self.reward = self.compute_reward(action)
            self.episode_return += self.reward

        if self.next_state[-2] < self.target_distance_from_goal:
            self.info["reached"] = True
            self.reward += 100

        # if np.sum(self.next_state) <= self.final_state:

        if self.episode_step == self.max_episode_lenght:
            self.info["max_legnth_reached"] = True

        self.state = self.next_state.copy()

        return self.next_state, self.reward, self.done, self.info

    def compute_reward(self, action):
        # Ricompensa per avvicinarsi all'obiettivo
        current_distance = self.next_state[-2]
        distance_reward = self.state[-2] - self.next_state[-2]

        too_close = 0.0
        distance_from_sensor = self.next_state[-1]
        if action == 0 and distance_from_sensor <= 0.3:
            too_close += -1

        # Ricompensa totale
        total_reward = distance_reward * 1000 - current_distance / 100 + too_close
        print(
            f"rw: {total_reward},going_farer: {distance_reward},d_to_target: {current_distance},too_close: {too_close}"
        )
        return total_reward

    def reset(self):

        self.robot.turbo = False
        print("Epispdes has ended", self.info)
        print("10*\n")
        for _ in range(10):
            self.state = self.get_state()
            self.previous_distance = self.state
            action = random.randint(0, 1)

            self.robot.do(action, force=3)
            time.sleep(self.robot.action_delay)

        self.episode_return = 0
        self.episode_step = 0
        self.done = False
        self.info.clear()
        self.info["max_legnth_reached"] = False
        self.info["reached"] = False

        return self.state


from nav_msgs.msg import Odometry


class MyNode(Node):

    def __init__(self, subscribing_topic, publishing_topic):
        super().__init__("my_node")
        self.distance_from_sensor = np.array([0], dtype=np.float32)
        self.odom_vector = None

        self.publisher_ = self.create_publisher(Twist, publishing_topic, 1)
        self.subscription = self.create_subscription(
            LaserScan, subscribing_topic, self.read_state, 1
        )
        self.subscription  # prevent unused variable warning
        self.subscription = self.create_subscription(
            Odometry, "/irobot02/odom", self.listener_odometry, 1
        )
        self.subscription  # prevent unused variable warning

    def listener_odometry(self, msg):
        # Estrai la posizione (x, y, z) e l'orientamento (x, y, z, w) dal messaggio Odometry
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        # Crea un array NumPy a partire dai dati estratti
        self.odom_vector = np.array(
            [
                position.x,
                position.y,
                position.z,
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ]
        )

        # Stampa o utilizza l'array NumPy
        # self.get_logger().info(f"Odom Vector: {self.odom_vector}")

    def read_state(self, msg):
        try:
            # TODO translate msg in state
            self.distance_from_sensor[0] = msg.ranges[0]
            if self.distance_from_sensor[0] == np.inf:
                self.distance_from_sensor[0] = 4
            print(f"i recieved: {self.distance_from_sensor}")
        except:
            print("state not update!")

    def send_message(self, msg):
        try:
            self.publisher_.publish(msg)
            # self.get_logger().info('Publishing: "%s"' % msg)
        except Exception as e:
            print(e)


class Robot:

    def __init__(self, name, node_ros):
        self.name = name
        self.node_ros = node_ros
        self.state = node_ros.odom_vector
        print(self)
        self.combined = False
        self.turbo = False

        self.last_linear = 0.0
        self.last_angular = 0.0

    def do(self, action, force=0.0):
        if self.node_ros.distance_from_sensor <= 0.30 and action == 0:
            action = 2

        print(f"action: {action}")
        msg = Twist()
        if action == 0:  # go forward
            msg.linear.x = 0.5 + force
            self.last_linear = msg.linear.x

            if self.turbo:
                msg.linear.x *= 2.0

            if self.combined:
                msg.angular.z = self.last_angular

        elif action == 1:  # turn left 15 degrees
            msg.angular.z = 0.5 + force
            self.last_angular = msg.angular.z

            if self.combined:
                msg.linear.x = self.last_linear

        elif action == 2:  # still
            msg.angular.z = 0.0
            msg.linear.x = 0.0
            self.last_linear = msg.linear.x
            self.last_angular = msg.angular.z

        elif action == 3:  # turn right 15 degrees
            msg.angular.z = -0.5 - force
            self.last_angular = msg.angular.z

            if self.combined:
                msg.linear.x = self.last_linear

        elif action == 4:  # go backward
            msg.linear.x = -0.5 - force
            self.last_linear = msg.linear.x

            if self.turbo:
                msg.linear.x *= 2.0

            if self.combined:
                msg.angular.z = self.last_angular

        elif action == 5:  # combine commands
            self.combined = not self.combined
            print(f"combined: {self.combined}")

        elif action == 6:  # turbo
            self.turbo = not self.turbo
            print(f"turbo: {self.turbo}")

        self.node_ros.send_message(msg)

    def read_state(self):
        rclpy.spin_once(self.node_ros)
        self.state = self.node_ros.state.copy()
        return self.state

    def __str__(self):
        return f"robot named:'{self.name}'"


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.network(x)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)


class PPOBuffer:
    def __init__(self, state_dim, buffer_size):
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.ptr = 0
        self.max_size = buffer_size

    def store(self, state, action, reward, done, log_prob):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        if self.ptr < self.max_size - 1:
            self.ptr += 1

    def get(self):
        self.ptr = 0
        return self.states, self.actions, self.rewards, self.dones, self.log_probs


import torch.optim as optim
from torch.distributions import Categorical


class PPO:
    def __init__(
        self, state_dim, action_dim, buffer_size, policy_lr=3e-4, value_lr=1e-3
    ):
        self.buffer = PPOBuffer(state_dim, buffer_size)
        self.policy = Actor(state_dim, action_dim)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.value_function = Critic(state_dim)
        self.value_optimizer = optim.Adam(self.value_function.parameters(), lr=value_lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def store_transition(self, state, action, reward, done, log_prob):
        self.buffer.store(state, action, reward, done, log_prob)

    def compute_gae(self, rewards, dones, values, gamma=0.99, tau=0.95):
        gae = 0
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + gamma * tau * (1 - dones[i]) * gae
            returns.insert(0, gae + values[i])
        return returns

    def update(self, gamma=0.99, k_epochs=4, eps_clip=0.2):
        states, actions, rewards, dones, log_probs = self.buffer.get()
        values = self.value_function(torch.FloatTensor(states)).detach().numpy()
        values = np.append(
            values, self.value_function(torch.FloatTensor(states[-1])).item()
        )
        returns = self.compute_gae(rewards, dones, values, gamma)
        returns = torch.FloatTensor(returns)
        advantages = returns - torch.FloatTensor(values[:-1])

        for _ in range(k_epochs):
            for _ in range(states.shape[0] // 64):
                indices = np.random.randint(0, states.shape[0], 64)
                sampled_states = torch.FloatTensor(states[indices])
                sampled_actions = torch.LongTensor(actions[indices])
                sampled_log_probs = torch.FloatTensor(log_probs[indices])
                sampled_advantages = advantages[indices]
                sampled_returns = returns[indices]

                probs = self.policy(sampled_states)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(sampled_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - sampled_log_probs)
                surr1 = ratio * sampled_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * sampled_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                value_loss = nn.MSELoss()(
                    self.value_function(sampled_states).squeeze(), sampled_returns
                )
                self.value_optimizer.zero_grad()
                value_loss.backward()

    def save_model(self, actor_path="actor.pth", critic_path="critic.pth"):
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.value_function.state_dict(), critic_path)
        print(f"Modelli di attore e critico salvati in {actor_path} e {critic_path}")

    def load_model(self, actor_path="actor.pth", critic_path="critic.pth"):
        self.policy.load_state_dict(torch.load(actor_path))
        self.value_function.load_state_dict(torch.load(critic_path))
        print(f"Modelli di attore e critico caricati da {actor_path} e {critic_path}")


def evaluate_ppo(env, ppo, episodes=1000):

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            action, log_prob = ppo.select_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward

            if done:
                print(f"Episode: {episode}, Reward: {episode_reward}")
                break


rclpy.init()

my_node = MyNode(
    subscribing_topic="demo/laser/out", publishing_topic="irobot02/cmd_vel"
)
robot = Robot("bot", my_node)
env = Env(robot)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
buffer_size = 2048  # Dimensione del buffer

ppo = PPO(state_dim, action_dim, buffer_size)
load = False

if load is not True:
    for episode in range(1000):
        state = env.reset()
        for t in range(1000):
            action, log_prob = ppo.select_action(state)
            next_state, reward, done, _ = env.step(action)
            ppo.store_transition(state, action, reward, done, log_prob)
            state = next_state
            if done:
                break
        if ppo.buffer.ptr == ppo.buffer.max_size:
            ppo.update(eps_clip=0.2)
            print(f"Episode {episode} complete")

    ppo.save_model()


else:
    ppo.load_model()
    evaluate_ppo(env, ppo, 100)
