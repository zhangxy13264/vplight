from . import RLAgent
from common.registry import Registry
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator, SidewalkPedestrianGenerator, IntersectionMapGenerator
from agent import utils
import gym
import os 

@Registry.register_model('ecolight')
class ECOLightAgent(RLAgent):
    def __init__(self, world, rank):
        super().__init__(world, world.intersection_ids[rank])
        self.world = world
        self.rank = rank
        self.inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[self.inter_id]
        self.num_inters = len(self.world.intersections)
        self.graph = Registry.mapping['world_mapping']['graph_setting'].graph
        self.phase = Registry.mapping['model_mapping']['setting'].param['phase']
        self.one_hot = Registry.mapping['model_mapping']['setting'].param['one_hot']
        self.model_dict = Registry.mapping['model_mapping']['setting'].param
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.action_space = gym.spaces.Discrete(len(self.world.id2intersection[self.inter_id].phases))
        self.num_phase = self.action_space.n
        self.use_con = Registry.mapping['model_mapping']['setting'].param.get('use_con', False)
        self.epsilon = Registry.mapping['model_mapping']['setting'].param['epsilon']

        self.ob_list = Registry.mapping['model_mapping']['setting'].param['ob_list']
        self.ob_list_p = Registry.mapping['model_mapping']['setting'].param['ob_list_p']

        self._reset_generator()
        self.obs_shape = [torch.Size([ob_generator.ob_length]) for ob_generator in self.ob_generators]
        if self.ob_list_p:
            self.obs_shape.append(torch.Size([self.ob_generator_p.ob_length]))
        self.obs_shape.append(torch.Size([self.num_phase])) if self.one_hot else self.obs_shape.append(torch.Size([1])) 

        self.replay_buffer = ReplayBuffer(Registry.mapping['trainer_mapping']['setting'].param['buffer_size'],
                                        Registry.mapping['model_mapping']['setting'].param['batch_size'],
                                        self.obs_shape,
                                        self.device)
        
        self.current_phase = 0
        self.tau = Registry.mapping['model_mapping']['setting'].param['tau']
        self.gamma = Registry.mapping['model_mapping']['setting'].param['gamma']

        self.phase_2_passable_lane = torch.tensor(self.inter_obj.phase_2_passable_lane_idx, device=self.device)
        self.network_local = _Network(self.phase_2_passable_lane).to(self.device)
        self.network_target = _Network(self.phase_2_passable_lane).to(self.device)
        self.network_optim = optim.RMSprop(
            self.network_local.parameters(),
            lr=Registry.mapping['model_mapping']['setting'].param['learning_rate']
        )
        self.network_lr_scheduler = optim.lr_scheduler.StepLR(self.network_optim, 20, 0.1)
        copy_model_params(self.network_local, self.network_target)
        
    def _get_generators(self, generator_name, generator_params):
        # generators = [(idx, LaneVehicleGenerator), ...]
        generators = [
            (
                self.graph['node_id2idx'][inter.id if 'GS_' not in inter.id else inter.id[3:]],
                generator_name(self.world, inter, **generator_params)
            )
            for inter in self.world.intersections
        ]
        return sorted(generators, key=lambda x: x[0])
    
    def _reset_generator(self):
        self.queue = LaneVehicleGenerator(self.world, self.inter_obj, ["lane_waiting_count"], in_only=True, negative=False)
        self.delay = LaneVehicleGenerator(self.world, self.inter_obj, ["lane_delay"], in_only=True, average=None, negative=False)
        self.pedestrian_queue = SidewalkPedestrianGenerator(self.world, self.inter_obj, ["sidewalk_waiting_count"], in_only=True, negative=False)
        self.pedestrian_delay = SidewalkPedestrianGenerator(self.world, self.inter_obj, ["sidewalk_delay"], in_only=True, average=None, negative=False)

        self.ob_generator = LaneVehicleGenerator(self.world,  self.inter_obj, self.ob_list, in_only=True, average=None)
        self.ob_generator_p = SidewalkPedestrianGenerator(self.world,  self.inter_obj, self.ob_list_p, in_only=True, average='road')
        self.phase_generator = IntersectionPhaseGenerator(self.world,  self.inter_obj, ["phase"],
                                                          targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world,  self.inter_obj, ['lane_waiting_count'],
                                                     in_only=True, average=None, negative=True)
        self.reward_generator_p = SidewalkPedestrianGenerator(self.world,  self.inter_obj, ['sidewalk_waiting_count'],
                                                in_only=True, average='road', negative=True)
        self.ob_generators = [LaneVehicleGenerator(self.world,  self.inter_obj, [ob_list], in_only=True, average=None) for ob_list in self.ob_list]

        self.ob_generators = [LaneVehicleGenerator(self.world,  self.inter_obj, [ob_list], in_only=True, average=None) for ob_list in self.ob_list]

    def reset(self):
        self.inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[self.inter_id]
        self._reset_generator()
        self.current_phase = 0
    
    def __repr__(self):
        return self.network_local.__repr__()

    def get_ob(self):
        x_obs = [ob_generator.generate() for ob_generator in self.ob_generators]
        if self.ob_list_p:
            x_obs.append(np.array(self.ob_generator_p.generate()))
        if self.one_hot:
            phase = utils.idx2onehot(np.array([self.current_phase]), self.action_space.n)
        x_obs.append(phase)
        return x_obs
    
    def get_reward(self):
        reward = self.reward_generator.generate()
        reward_p = self.reward_generator_p.generate()
        rewards = np.concatenate([reward, reward_p], axis=0)
        rewards = np.sum(rewards)
        return rewards
    
    def get_phase(self):
        phase = []
        phase.append(self.phase_generator.generate())
        phase = (np.concatenate(phase)).astype(np.int8)
        self.current_phase = phase[0]
        return phase

    
    def get_action(self, ob, phase, test=False):
        if not test:
            if np.random.rand() <= self.epsilon:
                return self.sample()
        
        observation = [torch.tensor(o, dtype=torch.float32, device=self.device) for o in ob]
        self.network_local.eval()
        b_q_value = self.network_local(observation)
        action = torch.argmax(b_q_value).cpu().item()
        self.network_local.train()

        if action == 1:
            self.current_phase = (self.current_phase + 1) % self.num_phase
        return self.current_phase
    
    def sample(self):
        return np.random.randint(0, self.action_space.n)

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
        last_obs = [torch.tensor(ob, dtype=torch.float32, device=self.device) for ob in last_obs]
        obs = [torch.tensor(ob, dtype=torch.float32, device=self.device) for ob in obs]
        rewards = torch.tensor(np.array([rewards]), dtype=torch.float32, device=self.device).unsqueeze(0)

        current_phase = torch.argmax(last_obs[-1]).item()
        binary_action = int(actions != current_phase)
        self.replay_buffer.store_experience(last_obs, binary_action, rewards, obs, done)
        
    def update_target_network(self):
        pass

    def save_model(self, e):
        path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}_{self.rank}.pt')
        torch.save({
            'network_local': self.network_local.state_dict(),
            'network_target': self.network_target.state_dict(),
        }, model_name)

    def load_model(self, e):
        model_name = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                  'model', f'{e}_{self.rank}.pt')
        model_dict = torch.load(model_name)
        self.network_local.load_state_dict(model_dict['network_local'])
        self.network_target.load_state_dict(model_dict['network_target'])

    def train(self):
        obs, act, rew, next_obs, done = self.replay_buffer.sample_experience()
        critic_loss = self._compute_critic_loss(obs, act, rew, next_obs, done)
        self.network_optim.zero_grad()
        critic_loss.backward()
        self.network_optim.step()
        for to_model, from_model in zip(self.network_target.parameters(), self.network_local.parameters()):
            to_model.data.copy_(self.tau * from_model.data + (1.0 - self.tau) * to_model.data)
        return critic_loss.cpu().detach().numpy()
    
    def _compute_critic_loss(self, obs, act, rew, next_obs, done):
        with torch.no_grad():
            q_target_next = self.network_target(next_obs)
            q_target = rew + self.gamma * torch.max(q_target_next, dim=1, keepdim=True)[0] * (~done)
        q_expected = self.network_local(obs).gather(1, act.long())
        critic_loss = F.mse_loss(q_expected, q_target)
        return critic_loss


class _Network(torch.nn.Module):
    def __init__(self, phase_2_passable_lane):
        super(_Network, self).__init__()
        self.phase_2_passable_lane = phase_2_passable_lane.float()  # (num_phase, num_lane)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features=2, out_features=10),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=10, out_features=10),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=10, out_features=2)
        )

    def forward(self, obs):
        # obs[1]: _inter_2_current_phase, current phase, B * num_phase;
        # obs[0]: _inlane_2_num_vehicle, number of cars in each lane, B * num_lane
        b_passable_lane = torch.matmul(obs[1], self.phase_2_passable_lane)  # B * num_lane
        b_num_car_with_green_signal = torch.sum(b_passable_lane * obs[0], dim=1, keepdim=True)  # B * 1
        b_num_car_with_red_signal = torch.sum((1. - b_passable_lane) * obs[0], dim=1, keepdim=True)  # B * 1

        b_feature = torch.cat([b_num_car_with_green_signal, b_num_car_with_red_signal], dim=1)  # B * 2
        q_values = self.net(b_feature)  # B * 2
        return q_values

def copy_model_params(source_model, target_model):
    for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(source_param.clone())

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, obs_size, device):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.obs_size = obs_size 
        self.device = device

        self.memory = {
            'obs': self._get_obs_placeholder(),
            'act': np.empty((self.buffer_size, 1), dtype=np.int64),
            'rew': np.empty((self.buffer_size, 1), dtype=np.float32),
            'next_obs': self._get_obs_placeholder(),
            'done': np.empty((self.buffer_size, 1), dtype=np.bool)
        }
        self._cur_idx = 0
        self.current_size = 0

    def _get_obs_placeholder(self):
        if isinstance(self.obs_size, list):
            return [np.empty((self.buffer_size, *siz), dtype=np.float32) for siz in self.obs_size]
        else:
            return np.empty((self.buffer_size, *self.obs_size), dtype=np.float32)

    def dump(self):
        return {
            "memory": self.memory,
            "_cur_idx": self._cur_idx,
            "current_size": self.current_size
        }

    def load(self, obj):
        self.memory = obj["memory"]
        self._cur_idx = obj["_cur_idx"]
        self.current_size = obj["current_size"]

    def reset(self):
        self._cur_idx = 0
        self.current_size = 0

    def store_experience(self, obs, act, rew, next_obs, done):
        if isinstance(self.obs_size, list):
            for feature_idx, ith_obs in enumerate(obs):
                self.memory['obs'][feature_idx][self._cur_idx] = ith_obs.cpu()
            for feature_idx, ith_next_obs in enumerate(next_obs):
                self.memory['next_obs'][feature_idx][self._cur_idx] = ith_next_obs.cpu()
        else:
            self.memory['obs'][self._cur_idx] = obs.cpu()
            self.memory['next_obs'][self._cur_idx] = next_obs.cpu()

        self.memory['act'][self._cur_idx] = act
        self.memory['rew'][self._cur_idx] = rew.cpu()
        self.memory['done'][self._cur_idx] = done

        self.current_size = min(self.current_size + 1, self.buffer_size)
        self._cur_idx = (self._cur_idx + 1) % self.buffer_size

    def sample_experience(self, batch_size=None, idxs=None):
        batch_size = batch_size or self.batch_size
        if idxs is None:
            idxs = np.random.choice(self.current_size, batch_size, replace=True)

        if isinstance(self.obs_size, list):
            obs, next_obs = [], []
            for obs_feature_idx in range(len(self.obs_size)):
                obs.append(self._to_torch(self.memory['obs'][obs_feature_idx][idxs]))
                next_obs.append(self._to_torch(self.memory['next_obs'][obs_feature_idx][idxs]))
        else:
            obs = self._to_torch(self.memory['obs'][idxs])
            next_obs = self._to_torch(self.memory['next_obs'][idxs])

        act = self._to_torch(self.memory['act'][idxs])
        rew = self._to_torch(self.memory['rew'][idxs])
        done = self._to_torch(self.memory['done'][idxs])
        return obs, act, rew, next_obs, done

    def get_sample_indexes(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        return np.random.choice(self.current_size, batch_size, replace=True)

    def _to_torch(self, np_elem):
        return torch.from_numpy(np_elem).to(self.device)

    def __str__(self):
        return str("current size: {} / {}".format(self.current_size, self.buffer_size))
