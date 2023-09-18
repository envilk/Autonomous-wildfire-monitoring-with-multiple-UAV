import os
import mesa
import math
import random
import numpy
import statistics
import matplotlib.pyplot as plt
import matplotlib as mpl
import collections
import itertools

import plotly.tools

import agents
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import ujson
from matplotlib import pyplot as pyplt


class ReplayMemory(object):

    def __init__(self, capacity, transition, system_random):
        self.memory = collections.deque([], maxlen=capacity)
        self.transition = transition
        self.system_random = system_random

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        return self.system_random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.squeeze()
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class WildFireModel(mesa.Model):

    def __init__(self, width, height, load, fuel_bottom_limit, fuel_upper_limit, rnn=False):

        plt.ion()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transition = collections.namedtuple('Transition',
                                                 ('state', 'action', 'next_state', 'reward'))

        self.new_direction = None
        self.new_direction_counter = None
        self.datacollector = None
        self.grid = None
        self.unique_agents_id = None
        self.NUM_AGENTS = int(input("Type down number of UAV (Remember, for auto-eval max number required): "))
        self.eval_times_RNN = None
        self.NOISE = False
        self.END_EVAL = False
        self.noise_amount = 0

        self.prev_i_episode = 0
        self.eval_step_counter = 0
        self.eval_ac_reward = 0
        self.fire_spread_speed = 2
        self.i_episode = 0
        self.burning_rate = 1
        self.height = height
        self.width = width
        self.LOAD = load
        self.RNN = rnn
        self.RNN_h = 10
        self.BATCH_SIZE = 90
        self.GAMMA = 0.5
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4
        self.N_ACTIONS = 4
        self.UAV_OBSERVATION_RADIUS = 8
        self.fuel_bottom_limit = fuel_bottom_limit
        self.fuel_upper_limit = fuel_upper_limit

        side = ((self.UAV_OBSERVATION_RADIUS * 2) + 1)  # allows to build a square side
        self.N_OBSERVATIONS = side * side  # calculates the observation square

        self.STEPS_DONE = 0
        self.EPISODE_REWARD = []
        self.EPISODE_REWARD_MEANS = []
        self.system_random = random.SystemRandom()

        self.reset()
        print(self.NUM_AGENTS)

        self.policy_net = DQN(self.N_OBSERVATIONS, self.N_ACTIONS).to(self.device)
        self.target_net = DQN(self.N_OBSERVATIONS, self.N_ACTIONS).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(10000, self.transition, self.system_random)

        print('actions:', self.N_ACTIONS)
        print('observations:', self.N_OBSERVATIONS)

        if self.LOAD:
            self.policy_net.load_state_dict(torch.load('checkpoints/'+str(3)
                                                       +'_drones_checkpoint_policy_net15000.pth'))
            self.policy_net.eval()
            print('model correctly loaded')

    def reset(self, eval=False):
        self.unique_agents_id = 0
        # Inverted width and height order, because of matrix accessing purposes, like in many examples:
        #   https://snyk.io/advisor/python/Mesa/functions/mesa.space.MultiGrid
        self.grid = mesa.space.MultiGrid(self.height, self.width, False)
        self.schedule = mesa.time.SimultaneousActivation(self)
        self.set_fire_agents()
        # set two UAV
        x_center = int(self.height / 2)
        y_center = int(self.width / 2)

        self.new_direction_counter = 0
        self.eval_step_counter = 0
        self.eval_ac_reward = 0
        self.eval_times_RNN = [0 for a in range(0, self.NUM_AGENTS)]

        for a in range(0, self.NUM_AGENTS):
            aux_UAV = agents.UAV(self.unique_agents_id, self, self.UAV_OBSERVATION_RADIUS)
            y_center += a if a % 2 == 0 else -a
            self.grid.place_agent(aux_UAV, (x_center, y_center + 1))
            self.schedule.add(aux_UAV)
            self.unique_agents_id += 1

        self.datacollector = mesa.DataCollector()
        self.new_direction = [0 for a in range(0, self.NUM_AGENTS)]
        if not eval:
            self.file_data = None

    def manhattan_distance(self, x1, y1, x2, y2):
        result = abs(x1 - x2) + abs(y1 - y2)
        if self.diagonal(tuple([x1, y1]), tuple([x2, y2])):
            result -= 1
        return result

    def euclidean_distance(self, x1, y1, x2, y2):
        a = numpy.array((x1, y1))
        b = numpy.array((x2, y2))
        dist = numpy.linalg.norm(a - b)
        return dist

    def diagonal(self, s, s_):
        row_diff = abs(s[0] - s_[0])
        col_diff = abs(s[1] - s_[1])
        diagonal = False
        if row_diff == col_diff:
            diagonal = True
        return diagonal

    def distance_rate(self, s, s_, distance_limit):
        m_d = self.manhattan_distance(s[0], s[1], s_[0], s_[1])
        result = 0
        if m_d <= distance_limit:
            result = m_d ** -2.0
        return result

    def plot_durations(self, show_result=False):
        plt.figure(1)
        if show_result:
            plt.title('Result')
            with open(str(self.NUM_AGENTS) + 'UAV_training_results.txt', 'w') as f:
                print(self.EPISODE_REWARD_MEANS)
                to_write = [str(reward) + '\n' for reward in self.EPISODE_REWARD_MEANS]
                f.writelines(to_write)
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        c = numpy.arange(0, self.NUM_AGENTS + 2)
        norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
        cmap_rewards = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
        cmap_rewards_means = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Reds)
        cmap_rewards.set_array([])
        cmap_rewards_means.set_array([])

        if self.LOAD:
            for agent_idx in range(0, self.NUM_AGENTS):
                to_plot = [reward[agent_idx] for reward in self.EPISODE_REWARD]
                plt.plot(to_plot, c=cmap_rewards.to_rgba(agent_idx + 1))
        else:
            min_num = 100
            # for each agent, extract mean of last 'min_num' elements of the 'self.EPISODE_REWARD' list. Then, append
            # to 'self.EPISODE_REWARD',
            aux_means = [0 for agent_idx in range(0, self.NUM_AGENTS)]
            if len(self.EPISODE_REWARD) >= min_num + 1:  # extract means
                aux_means = [statistics.mean([reward[agent_idx] for reward in self.EPISODE_REWARD[-min_num:]])
                             for agent_idx in range(0, self.NUM_AGENTS)]
            self.EPISODE_REWARD_MEANS.append(aux_means)  # store mean
            for agent_idx in range(0, self.NUM_AGENTS):  # plot it
                to_plot = [means[agent_idx] for means in self.EPISODE_REWARD_MEANS]
                plt.plot(to_plot, c=cmap_rewards_means.to_rgba(agent_idx + 1))
        plt.pause(0.001)  # pause a bit so that plots are updated

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)

        dimension = 2 if self.NUM_AGENTS >= 2 else 1
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
        state_batch = torch.stack(list(batch.state), dim=0)
        action_batch = torch.stack(list(batch.action), dim=0)
        reward_batch = torch.stack(list(batch.reward), dim=0)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        action_batch = action_batch.unsqueeze(dimension) if self.NUM_AGENTS >= 2 else action_batch
        sav = self.policy_net(state_batch)
        state_action_values = sav.gather(dimension, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        with torch.no_grad():
            next_state_values = self.target_net(non_final_next_states).max(dimension)[0]
        # Compute the expected Q values
        if self.NUM_AGENTS >= 2:
            expected_state_action_values = (next_state_values.unsqueeze(1) * self.GAMMA) + reward_batch
        else:
            expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch.squeeze()

        # Compute Huber loss
        criterion = nn.SmoothL1Loss(reduction='sum')
        loss = criterion(state_action_values.squeeze(), expected_state_action_values.squeeze())

        # write same file during "self.prev_i_episodes" episodes
        path = "predictor/" + str(self.NUM_AGENTS) + '_drones_' + str(self.prev_i_episode) + '_output.json'
        self.write_json(path, {"episode": self.i_episode, "loss": float(loss)})

        # Optimize the model
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self):
        start = time.perf_counter()
        num_episodes = 15001 if torch.cuda.is_available() else 50

        for episode in range(num_episodes):
            self.i_episode = episode
            reward_dict = {"episode": episode, "elements": []}
            # Initialize the environment and get it's state
            self.reset()
            state, _, = self.state()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            episode_rewards = 0
            for t in itertools.count():
                # for writing json files
                if self.i_episode % 100 == 0:
                    self.prev_i_episode = self.i_episode

                # Input -> state [self.NUM_OBSERVATIONS] | Output -> action [self.NUM_AGENTS] p.e. I[25] -> O[2]
                actions = self.select_action(state)

                # Output -> state [self.NUM_OBSERVATIONS], reward [self.NUM_AGENTS] p.e. I[25] -> O[2]
                state, _ = self.state()
                state_to_store = state.copy()
                state = torch.tensor(state, dtype=torch.float32, device=self.device)
                self.new_direction = actions
                self.step(-1, -1, -1)
                next_state, reward = self.state()

                if self.NOISE:
                    noise = numpy.random.normal(-self.noise_amount, self.noise_amount, len(reward))
                    reward += noise
                    reward = reward.tolist()

                reward = torch.tensor([reward], device=self.device)
                episode_rewards += reward
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)

                # Store the transition in memory Transition([self.NUM_AGENTS, self.NUM_OBSERVATIONS],
                # [1, self.NUM_AGENTS], [self.NUM_AGENTS, self.NUM_OBSERVATIONS], [self.NUM_AGENTS])
                # p.e. T([2, 25], [1, 2], [2, 25], [2])
                self.memory.push(state, actions, next_state, reward)

                if self.RNN:
                    path = "predictor/" + str(self.NUM_AGENTS) + '_drones_' + str(
                        self.prev_i_episode) + '_rewards_dataset.json'
                    reward_dict['elements'].append({"t": t, "reward": reward.tolist()[0],
                                                    "ac_reward": episode_rewards.tolist()[0],
                                                    "action": actions.tolist(),
                                                    "state": state_to_store[0]})

                # Move to the next state
                state = next_state
                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (
                            1 - self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                # if done:
                if t == self.BATCH_SIZE:
                    # print(episode_rewards)
                    self.EPISODE_REWARD.append(*episode_rewards.tolist())
                    self.plot_durations()
                    if self.RNN:
                        self.write_json(path, reward_dict)
                    break
            if self.i_episode % 250 == 0:
                root = 'checkpoints/'
                if os.path.exists(root):
                    torch.save(self.policy_net.state_dict(),
                               root + str(self.NUM_AGENTS) + '_drones_checkpoint_policy_net'
                               + str(self.i_episode) + '.pth')
                else:
                    os.mkdir(root)
            if self.i_episode % 1000 == 0:
                pyplt.savefig(str(self.NUM_AGENTS) + '_' + str(self.i_episode) + '_drones_pyfoo.svg')
                end = time.perf_counter() - start
                print('EPISODE: ', self.i_episode, 'Elapsed time:', end)

        # torch.save(self.policy_net.state_dict(), 'policy_net.pth')
        print('Complete')

        self.plot_durations(show_result=True)
        plt.ioff()
        pyplt.savefig(str(self.NUM_AGENTS) + '_drones_pyfoo.svg')
        # plt.show()

    def load_predictor_auto(self):
        if self.RNN:
            _dir = 'predictor/'
            filenames = os.listdir(_dir)
            self.files = []
            for idx, filename in enumerate(filenames):
                if filename.endswith('_drones_rewards_dataset.json'):
                    with open(_dir + filename, 'r') as f:
                        self.files.append(ujson.load(f))

    def evaluation(self, type='entrenamiento', load=True):
        labels = []
        for dirpath, dirnames, filenames in os.walk("pruebas_" + type):
            print(dirpath)
            if dirpath.endswith("0"):
                labels.append(int(dirpath.split('/')[-1]))

        labels.sort()
        print(labels)
        eval = True
        self.LOAD = load
        if type == 'predictor':
            self.load_predictor_auto()

        max_agents = self.NUM_AGENTS
        for label in labels:
            for UAV_idx in range(0, max_agents):  # "max_agents" must go to the max number of agents on training (self.NUM_AGENTS)
                self.NUM_AGENTS = UAV_idx+1
                if type == 'entrenamiento':  # to not waste time in loading models when predictor activated
                    to_load = 'checkpoints/' + str(UAV_idx + 1) + '_drones_checkpoint_policy_net15000.pth'
                    self.policy_net.load_state_dict(torch.load(to_load))
                    self.policy_net.eval()
                else:
                    print(len(self.files))
                    self.file_data = self.files[UAV_idx]
                num_runs = 100
                self.strings_to_write_eval = []
                for i in range(0, num_runs):
                    self.EPISODE_REWARD = []
                    self.reset(eval)
                    for t in itertools.count():
                        self.step(num_run=i, auto=True, UAV_idx=UAV_idx, type=type, label=label)
                        # if done:
                        if self.END_EVAL:
                            self.END_EVAL = False
                            break

                string_to_write = ''
                # for e in self.strings_to_write_eval:
                #    string_to_write +=  + '\n'
                # f.write(string_to_write)
                with open('pruebas_' + type + '/' + str(label) + '/' + str(UAV_idx + 1) + 'UAV.txt', 'w') as f:
                    f.writelines(self.strings_to_write_eval)

    def write_json(self, path, to_write):
        if os.path.exists(path):
            with open(path, 'r') as f:
                file_data = ujson.load(f)
                file_data["lists"].append(to_write)
        else:
            file_data = {"lists": []}
        with open(path, 'w') as f:
            ujson.dump(file_data, f, indent=4)

    def select_action(self, state):
        sample = self.system_random.random()
        if self.LOAD:
            with torch.no_grad():
                # t.argmax() will return the index of the max action prob (int index).
                # from the four defined (north, east, south, west)
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                dirs = self.policy_net(state)
                returned_list = []

                # needed to be developed like this, because policy.net returns different depending on the dimensions
                if self.NUM_AGENTS < 2:
                    _, idx = dirs.topk(1)
                    # 50% takes first best direction for that cell, other 50% takes second best
                    returned = idx[0]  # if sample > 0.3 else idx[1]
                    returned_list.append(returned.view(1, 1))
                    # print(returned, idx)
                else:
                    for dir in dirs:
                        _, idx = dir.topk(1)
                        # 50% takes first best direction for that cell, other 50% takes second best
                        returned = idx[0]  # if sample > 0.3 else idx[1]
                        returned_list.append(returned.view(1, 1))

                return torch.tensor(returned_list)
        else:
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                            math.exp(-1. * self.STEPS_DONE / self.EPS_DECAY)
            self.STEPS_DONE += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.argmax() will return the index of the max action prob (int index).
                    # from the four defined (north, east, south, west)
                    action_probabilities = self.policy_net(state).tolist()  # .argmax().view(1, 1)
                    if self.NUM_AGENTS >= 2:
                        actions = [numpy.argmax(a_p) for a_p in action_probabilities]
                    else:
                        actions = [numpy.argmax(action_probabilities)]
                    return torch.tensor(actions, device=self.device, dtype=torch.long)
            else:
                return torch.tensor(
                    [self.system_random.choice(range(0, self.N_ACTIONS)) for i in range(0, self.NUM_AGENTS)],
                    device=self.device, dtype=torch.long)

    def set_fire_agents(self):
        x_c = int(self.height / 2)
        y_c = int(self.width / 2)
        x = [x_c]  # , x_c + 1, x_c - 1
        y = [y_c]  # , y_c + 1, y_c - 1
        for i in range(self.height):
            for j in range(self.width):
                if i in x and j in y:
                    self.new_fire_agent(i, j, True)
                else:
                    self.new_fire_agent(i, j, False)

    def new_fire_agent(self, pos_x, pos_y, burning):
        source_fire = agents.Fire(self.unique_agents_id, self, burning, self.fuel_bottom_limit, self.fuel_upper_limit,
                                  self.fire_spread_speed)
        self.unique_agents_id += 1
        self.schedule.add(source_fire)
        self.grid.place_agent(source_fire, tuple([pos_x, pos_y]))

    def normalize(self, to_normalize, upper, multiplier, subtractor):
        return ((to_normalize / upper) * multiplier) - subtractor

    def state(self):
        states = []
        rewards = []
        states_positions = []
        UAV_positions = []
        for agent in self.schedule.agents:
            if type(agent) is agents.UAV:
                surrounding_states, reward, positions = agent.surrounding_states()
                states.append(surrounding_states)
                rewards.append(self.normalize(float(reward), self.N_OBSERVATIONS, 1, 0))
                states_positions.append(positions)
                UAV_positions.append(agent.pos)

        # if there are three agents: intersection [(1-2), (1-3)], [(2-1), (2-3)], [(3-1), (3-2)]
        # Basically the sentence shown before means that an UAV receives a -1 reward when it
        # overlaps one cell in its observation area with another UAV
        # FINNISH COMMENT
        for i in range(0, len(states_positions)):
            reward_discount_counter = 0
            aux_states_positions = states_positions.copy()
            aux_final_states = []
            del aux_states_positions[i]

            for st in aux_states_positions:
                aux_final_states.extend(set(states_positions[i]) & set(st))

            reward_discount_counter += (len(set(aux_final_states)) / self.N_OBSERVATIONS)
            rewards[i] -= reward_discount_counter

        # when UAV reaches edge or corner, takes less surrounding states, so fulfilling the vector till
        # maximum amount of observation is necessary. It is IMPORTANT to mention that this COULD AFFECT to
        # the correct behaviour of drones when these are trying to maintain distance with other, because of
        # less area is taking into account. Either way, in most cases this is expendable.
        for st, _ in enumerate(states):
            counter = len(states[st])
            for i in range(counter, self.N_OBSERVATIONS):
                states[st].append(0)
        return states, rewards

    def set_drone_dirs(self):
        self.new_direction_counter = 0
        for agent in self.schedule.agents:
            if type(agent) is agents.UAV:
                agent.selected_dir = self.new_direction[self.new_direction_counter]
                self.new_direction_counter += 1

    def step(self, UAV_idx=-1, label=-1, type='', num_run=0, auto=False):
        self.datacollector.collect(self)
        if sum(isinstance(i, agents.UAV) for i in self.schedule.agents) > 0:
            if self.LOAD:
                state, _ = self.state()  # s_tl
                if self.RNN:
                    print('hello')
                else:
                    self.new_direction = self.select_action(state)  # a_t

                _, reward = self.state()  # r_t+1
                # print(reward, self.new_direction)
                self.EPISODE_REWARD.append(reward)
                reward = torch.tensor([reward], device=self.device)
                self.eval_ac_reward += reward
                if not auto:
                    self.plot_durations(show_result=True)
                if self.eval_step_counter == self.BATCH_SIZE:
                    self.END_EVAL = True
                    if auto:
                        root = 'pruebas_' + type + '/' + str(label) + '/' + str(UAV_idx + 1) + 'UAV_run_rewards' + '/'
                        if not os.path.isdir(root):
                            os.mkdir(root)
                        title = root + str(num_run) + '_EVAL_drones_pyfoo.svg'
                        self.plot_durations()
                        self.strings_to_write_eval.append(str(*self.eval_ac_reward.tolist()) + '\n')
                    else:
                        title = str(self.NUM_AGENTS) + '_EVAL_drones_pyfoo.svg'
                        print('EVAL AC REWARD: ', *self.eval_ac_reward.tolist())
                    pyplt.savefig(title)

                self.eval_step_counter += 1
            self.set_drone_dirs()
        self.schedule.step()  # self.new_direction is used to execute previous obtained a_t

