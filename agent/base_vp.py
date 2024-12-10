from . import BaseAgent
from common.registry import Registry
from agent import utils
import numpy as np
import os
import random
from collections import deque
import gym

from generator import LaneVehicleGenerator, IntersectionPhaseGenerator, SidewalkPedestrianGenerator, IntersectionMapGenerator

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from agent.model.dqn_st import GatedLayer, FlexibleGatedLayer
from torch.nn.utils import clip_grad_norm_
import math

@Registry.register_model('base_vp')
class BaseVPAgent(BaseAgent):
    def __init__(self, world, rank):
        super().__init__(world)
        self.rank = rank
        self.id = self.world.intersection_ids[self.rank]
        self.inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[self.inter_id]
        self.sub_agents = 1
        self.num_inters = len(self.world.intersections)

        self.model_dict = Registry.mapping['model_mapping']['setting'].param
        self.gamma = Registry.mapping['model_mapping']['setting'].param['gamma']
        self.grad_clip = Registry.mapping['model_mapping']['setting'].param['grad_clip']
        self.epsilon = Registry.mapping['model_mapping']['setting'].param['epsilon']
        self.epsilon_decay = Registry.mapping['model_mapping']['setting'].param['epsilon_decay']
        self.epsilon_min = Registry.mapping['model_mapping']['setting'].param['epsilon_min']
        self.learning_rate = Registry.mapping['model_mapping']['setting'].param['learning_rate']
        self.vehicle_max = Registry.mapping['model_mapping']['setting'].param['vehicle_max']
        self.batch_size = Registry.mapping['model_mapping']['setting'].param['batch_size']
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.phase = Registry.mapping['model_mapping']['setting'].param['phase']
        self.one_hot = Registry.mapping['model_mapping']['setting'].param['one_hot']
        self.action_interval = Registry.mapping['trainer_mapping']['setting'].param['action_interval']
        self.duration = Registry.mapping['model_mapping']['setting'].param.get('duration', False)
        self.map = Registry.mapping['model_mapping']['setting'].param.get('map', False)
        self.use_con = Registry.mapping['model_mapping']['setting'].param.get('use_con', False)
        self.graph = Registry.mapping['world_mapping']['graph_setting'].graph
        self.use_gradnorm = False
        self.duration_time = self.action_interval if self.duration else 1
        self.model_dict['duration_time'] = self.duration_time
        self.action_space = gym.spaces.Discrete(len(self.world.id2intersection[self.inter_id].phases))
        self.model_dict['action_space'] = self.action_space.n

        ######################################################  generator  ###############################################################
        self.ob_list = Registry.mapping['model_mapping']['setting'].param['ob_list']
        self.ob_list_p = Registry.mapping['model_mapping']['setting'].param['ob_list_p']

        self.reward_list = Registry.mapping['model_mapping']['setting'].param['reward_list']
        self.reward_list_p = Registry.mapping['model_mapping']['setting'].param['reward_list_p']

        self.map_list = ['walking_area_map']
        self.model_dict['wmap_num_grids'] = int(self.model_dict['wmap_area_width'] / self.model_dict['wmap_grid_width'])

        self._reset_generator()
        if self.phase:
            if self.one_hot:
                self.phase_length = len(self.world.id2intersection[self.inter_id].phases)
            else:
                self.phase_length = 1
        else:
            self.phase_length = 0
        self.model_dict['phase_length'] = self.phase_length
        self.model_dict['ob_length'] = self.ob_generator.ob_length
        self.model_dict['num_pic'] = self.walking_area_map_generator.ob_length
        self.state_length = sum([gen[1].ob_length for gen in self.v_ob_generators])
        self.model_dict['state_length'] = self.state_length
        self.model_dict['wcon_length'] = self.walking_area_con_generator.ob_length


        ######################################################  model  ###############################################################
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dic_agent_conf = Registry.mapping['model_mapping']['setting']
        self.dic_traffic_env_conf = Registry.mapping['world_mapping']['setting']
        map_name = self.dic_traffic_env_conf.param['network']
        all_valid_acts = self.dic_traffic_env_conf.param['signal_config'][map_name]['valid_acts']
        if all_valid_acts is None:
            self.valid_acts = None
        else:
            if self.inter_id in all_valid_acts.keys():
                self.inter_name = self.inter_id
            else:
                if 'GS_' in self.inter_id:
                    self.inter_name = self.inter_id[3:]
                else:
                    self.inter_name = 'GS_' + self.inter_id
            self.valid_acts = all_valid_acts[self.inter_name]
        
        self.ob_order = None
        if 'lane_order' in self.dic_traffic_env_conf.param['signal_config'][map_name].keys():
            self.ob_order = self.dic_traffic_env_conf.param['signal_config'][map_name]['lane_order'][self.inter_name]

        self.phase_pairs = []
        all_phase_pairs = self.dic_traffic_env_conf.param['signal_config'][map_name]['phase_pairs']
        if self.valid_acts:
            for idx in self.valid_acts:
                self.phase_pairs.append([self.ob_order[x] for x in all_phase_pairs[idx]])
        else:
            self.phase_pairs = all_phase_pairs

        self.comp_mask = self.relation()
        self.num_phases = len(self.phase_pairs)
        self.num_actions = len(self.phase_pairs)

        self.model_dict['phase_pairs'] = self.phase_pairs
        self.model_dict['comp_mask'] = self.comp_mask
        self.model_dict['p_phase_pairs'] = self.dic_traffic_env_conf.param['signal_config'][map_name]['p_phase_pairs']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dict['device'] = self.device
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1e-7)
        self.criterion = nn.MSELoss(reduction='mean')

    def relation(self):
        comp_mask = []
        for i in range(len(self.phase_pairs)):
            zeros = np.zeros(len(self.phase_pairs) - 1, dtype=np.int64)
            cnt = 0
            for j in range(len(self.phase_pairs)):
                if i == j: continue
                pair_a = self.phase_pairs[i]
                pair_b = self.phase_pairs[j]
                if len(list(set(pair_a + pair_b))) == 3: zeros[cnt] = 1
                cnt += 1
            comp_mask.append(zeros)
        comp_mask = torch.from_numpy(np.asarray(comp_mask))
        return comp_mask
        
    def __repr__(self):
        return self.model.__repr__()
        
    def _get_generators(self, generator_name, generator_params):
        # generators = [(idx, LaneVehicleGenerator), ...]
        generators = [(self.graph['node_id2idx'][inter.id if 'GS_' not in inter.id else inter.id[3:]], 
                        generator_name(self.world, inter, **generator_params)) for inter in self.world.intersections]
        return sorted(generators, key=lambda x: x[0])
    
    def _reset_generator(self):
        self.queue = LaneVehicleGenerator(self.world, self.inter_obj, ["lane_waiting_count"], in_only=True, negative=False)
        self.delay = LaneVehicleGenerator(self.world, self.inter_obj, ["lane_delay"], in_only=True, average=None, negative=False)
        self.pedestrian_queue = SidewalkPedestrianGenerator(self.world, self.inter_obj, ["sidewalk_waiting_count"], in_only=True, negative=False)
        self.pedestrian_delay = SidewalkPedestrianGenerator(self.world, self.inter_obj, ["sidewalk_delay"], in_only=True, average=None, negative=False)

        self.ob_generator = LaneVehicleGenerator(self.world,  self.inter_obj, self.ob_list, in_only=True, average=None)
        self.ob_generator_p = SidewalkPedestrianGenerator(self.world,  self.inter_obj, self.ob_list_p, in_only=True, average='road')
        self.phase_generator = IntersectionPhaseGenerator(self.world,  self.inter_obj, ["phase"], targets=["cur_phase"], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world,  self.inter_obj, self.reward_list, in_only=True, average=None, negative=True)
        self.reward_generator_p = SidewalkPedestrianGenerator(self.world,  self.inter_obj, self.reward_list_p, in_only=True, average='road', negative=True)
        self.walking_area_map_generator = IntersectionMapGenerator(self.world,  self.inter_obj,fns=["pedestrian_position", "sidewalk_pedestrians"], targets=['walking_area_map'], area_width = self.model_dict['wmap_area_width'], grid_width = self.model_dict['wmap_grid_width'])
        self.walking_area_con_generator = IntersectionMapGenerator(self.world,  self.inter_obj,fns=["pedestrian_position", "sidewalk_pedestrians"], targets=["walking_area_con"])
        self.walking_area_num_generator = IntersectionMapGenerator(self.world,  self.inter_obj,fns=["pedestrian_position", "sidewalk_pedestrians"], targets=["walking_area_num"])

        self.v_ob_generators = self._get_generators(LaneVehicleGenerator, dict(fns=self.ob_list, in_only=True, average=None))
        self.p_ob_generators = self._get_generators(SidewalkPedestrianGenerator, dict(fns=self.ob_list_p, in_only=True, average='road'))
        self.phase_generators = self._get_generators(IntersectionPhaseGenerator, dict(fns=["phase"], targets=["cur_phase"], negative=False))
        self.v_reward_generators = self._get_generators(LaneVehicleGenerator, dict(fns=self.reward_list, in_only=True, average=None, negative=True))
        self.p_reward_generators = self._get_generators(SidewalkPedestrianGenerator, dict(fns=self.reward_list_p, in_only=True, average='road', negative=True))
        self.walking_area_map_generators = self._get_generators(IntersectionMapGenerator, dict(fns=["pedestrian_position", "sidewalk_pedestrians"], targets=['walking_area_map'], area_width = self.model_dict['wmap_area_width'], grid_width = self.model_dict['wmap_grid_width']))
        self.walking_area_con_generators = self._get_generators(IntersectionMapGenerator, dict(fns=["pedestrian_position", "sidewalk_pedestrians"], targets=["walking_area_con"]))
        self.walking_area_num_generators = self._get_generators(IntersectionMapGenerator, dict(fns=["pedestrian_position", "sidewalk_pedestrians"], targets=["walking_area_num"]))
        
    def reset(self):
        self.inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[self.inter_id]
        self._reset_generator()



    def get_maps(self):
        map_ret = self.walking_area_map_generator.generate()
        walking_area_maps = np.array(map_ret) # (4, 10, 10) num, height, width
        walking_area_maps = np.expand_dims(walking_area_maps, axis=-3) # (4, 1, 10, 10) num, channel, height, width
        walking_area_maps = np.expand_dims(walking_area_maps, axis=0) # (1, 4, 1, 10, 10) batch, num, channel, height, width
        return walking_area_maps
    
    def get_wcon(self):
        return self.walking_area_con_generator.generate()
    
    def get_wnum(self):
        return self.walking_area_num_generator.generate()

    def get_ob(self):
        x_vob = np.array([self.ob_generator.generate()])
        x_wcon = np.array([self.walking_area_con_generator.generate()])
        return [x_vob, x_wcon]
    
    def get_reward(self):
        reward = self.reward_generator.generate()
        reward_p = self.reward_generator_p.generate()
        rewards = np.concatenate([reward, reward_p], axis=0)
        rewards = np.sum(rewards)
        return rewards

    def get_phase(self):
        phase = [self.phase_generator.generate()]
        phase = (np.concatenate(phase)).astype(np.int8)
        phase = np.array([phase])
        return phase
    
    def get_xphase(self, phase):
        return phase
    
    def get_xobss(self, ob):
        return ob[-1][0]
    
    def get_xcon(self, ob):
        return ob[-1][1]
    
    def _build_model(self):
        model = VPLight(**self.model_dict).to(self.device)
        return model
    
    def get_action(self, ob, phase, test=False):
        if not test:
            if np.random.rand() <= self.epsilon:
                return self.sample()

        x_obss = self.get_xobss(ob)
        x_phase = self.get_xphase(phase)
        x_con = self.get_xcon(ob)
        x_obss = torch.tensor(x_obss, dtype=torch.float32, device=self.device) 
        x_phase = torch.tensor(x_phase, dtype=torch.float32, device=self.device) 
        x_con = torch.tensor(x_con, dtype=torch.float32, device=self.device)

        observation = [x_phase, x_obss, x_con]
        actions = self.model(observation, train=False)
        actions = actions.cpu().detach().numpy()
        return np.argmax(actions, axis=1)
    
    def sample(self):
        return np.random.randint(0, self.action_space.n, self.sub_agents)
    
    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
        self.replay_buffer.append((key, (last_obs, last_phase, actions, rewards, obs, cur_phase)))
    
    def batchwise(self, samples):
        x_obss_t = np.concatenate([self.get_xobss(item[1][0]) for item in samples])
        x_phase_t = np.concatenate([self.get_xphase(item[1][1]) for item in samples])
        x_con_t = np.concatenate([self.get_xcon(item[1][0]) for item in samples])
        x_obss_t = torch.tensor(x_obss_t, dtype=torch.float32, device=self.device)
        x_phase_t = torch.tensor(x_phase_t, dtype=torch.float32, device=self.device)
        x_con_t = torch.tensor(x_con_t, dtype=torch.float32, device=self.device)
        state_t = [x_phase_t, x_obss_t, x_con_t]

        x_obss_tp = np.concatenate([self.get_xobss(item[1][4]) for item in samples])
        x_phase_tp = np.concatenate([self.get_xphase(item[1][5]) for item in samples])
        x_con_tp = np.concatenate([self.get_xcon(item[1][4]) for item in samples])
        x_obss_tp = torch.tensor(x_obss_tp, dtype=torch.float32, device=self.device)
        x_phase_tp = torch.tensor(x_phase_tp, dtype=torch.float32, device=self.device)
        x_con_tp = torch.tensor(x_con_tp, dtype=torch.float32, device=self.device)
        state_tp = [x_phase_tp, x_obss_tp, x_con_tp]

        rewards = torch.tensor(np.array([item[1][3] for item in samples]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array([item[1][2] for item in samples]), dtype=torch.long).to(self.device)

        return state_t, state_tp, rewards, actions
    
    def train(self):
        samples = random.sample(self.replay_buffer, self.batch_size)
        b_t, b_tp, rewards, actions = self.batchwise(samples)
        
        out = self.target_model(b_tp, train=False)
        target = rewards + self.gamma * torch.max(out, dim=1)[0]
        target_f = self.model(b_t, train=False)
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        loss = self.criterion(self.model(b_t, train=True), target_f)
        
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.cpu().detach().numpy()
    
    def update_target_network(self):
        weights = self.model.state_dict()
        self.target_model.load_state_dict(weights)

    def load_model(self, e):
        model_name = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                  'model', f'{e}_{self.rank}.pt')
        self.model = self._build_model().to(self.device)
        self.model.load_state_dict(torch.load(model_name))
        self.target_model = self._build_model().to(self.device)
        self.target_model.load_state_dict(torch.load(model_name))
        
    def save_model(self, e):
        self.model.cpu()
        self.target_model.cpu()
        path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}_{self.rank}.pt')
        torch.save(self.target_model.state_dict(), model_name)
        self.model.cuda()
        self.target_model.cuda()

class VPLight(nn.Module):
    def __init__(self, **kwargs):
        super(VPLight, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = FRAP(**kwargs)

    def _forward(self, obs):
        x_phase, x_obss, x_con = obs[0], obs[1], obs[2]
        x = torch.cat((x_phase, x_obss, x_con), dim=-1)
        x = self.feature_extractor(x)
        return x
    
    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)
            
class DQNNet(nn.Module):
    def __init__(self, **kwargs):
        super(DQNNet, self).__init__()
        input_dim = kwargs['ob_length'] + kwargs['wcon_length'] + kwargs['phase_length'] 
        output_dim = kwargs['action_space']  
        self.dense_1 = nn.Linear(input_dim, 20)
        self.dense_2 = nn.Linear(20, 20)
        self.dense_3 = nn.Linear(20, output_dim)
        
    def _forward(self, x):
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = self.dense_3(x)
        return x

    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)

class FRAP(nn.Module):
    def __init__(self, **kwargs):
        super(FRAP, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.oshape = kwargs['action_space']  
        self.phase_pairs = kwargs['phase_pairs']  
        self.comp_mask = kwargs['comp_mask']  
        self.demand_shape = kwargs['demand_shape']      # Allows more than just queue to be used
        self.one_hot = kwargs['one_hot']
        self.d_out = 4      # units in demand input layer
        self.p_out = 4      # size of phase embedding
        self.lane_embed_units = 16
        relation_embed_size = 4
        self.p = nn.Embedding(2, self.p_out)
        self.d = nn.Linear(self.demand_shape, self.d_out)
        self.lane_embedding = nn.Linear(self.p_out + self.d_out, self.lane_embed_units)
        self.lane_conv = nn.Conv2d(2*self.lane_embed_units, 20, kernel_size=(1, 1))
        self.relation_embedding = nn.Embedding(2, relation_embed_size)
        self.relation_conv = nn.Conv2d(relation_embed_size, 20, kernel_size=(1, 1))
        self.hidden_layer = nn.Conv2d(20, 20, kernel_size=(1, 1))
        self.before_merge = nn.Conv2d(20, 1, kernel_size=(1, 1))

    def _forward(self, states):
        num_movements = int((states.size()[1]-1)/self.demand_shape) if not self.one_hot else int((states.size()[1]-len(self.phase_pairs))/self.demand_shape)
        batch_size = states.size()[0]
        acts = states[:, :1].to(torch.int64) if not self.one_hot else states[:, :len(self.phase_pairs)].to(torch.int64)
        states = states[:, 1:] if not self.one_hot else states[:, len(self.phase_pairs):]
        states = states.float()
        extended_acts = []
        if not self.one_hot:
            for i in range(batch_size):
                act_idx = acts[i]
                pair = self.phase_pairs[act_idx]
                zeros = torch.zeros(num_movements, dtype=torch.int64, device=self.device)
                for idx in pair:
                    zeros[idx] = 1
                # zeros[pair[0]] = 1
                # zeros[pair[1]] = 1
                extended_acts.append(zeros)
            extended_acts = torch.stack(extended_acts)
        else:
            extended_acts = acts
        phase_embeds = torch.sigmoid(self.p(extended_acts))
        phase_demands = []
        for i in range(num_movements):
            # phase = phase_embeds[:, idx]  # size 4
            phase = phase_embeds[:, i]  # size 4
            demand = states[:, i:i+self.demand_shape]
            demand = torch.sigmoid(self.d(demand))    # size 4
            phase_demand = torch.cat((phase, demand), -1)
            phase_demand_embed = F.relu(self.lane_embedding(phase_demand))
            phase_demands.append(phase_demand_embed)
        phase_demands = torch.stack(phase_demands, 1)

        pairs = []
        for pair in self.phase_pairs:
            combined_demand = phase_demands[:, pair[0]]
            for idx in pair[1:]:
                combined_demand += phase_demands[:, idx]
            combined_demand = combined_demand / len(pair)
            pairs.append(combined_demand)
            # pairs.append(phase_demands[:, pair[0]] + phase_demands[:, pair[1]])

        rotated_phases = []
        for i in range(len(pairs)):
            for j in range(len(pairs)):
                if i != j: rotated_phases.append(torch.cat((pairs[i], pairs[j]), -1))
        rotated_phases = torch.stack(rotated_phases, 1)
        rotated_phases = torch.reshape(rotated_phases,
                                       (batch_size, self.oshape, self.oshape - 1, 2 * self.lane_embed_units))
        rotated_phases = rotated_phases.permute(0, 3, 1, 2)  # Move channels up
        rotated_phases = F.relu(self.lane_conv(rotated_phases))  # Conv-20x1x1  pair demand representation

        # Phase competition mask
        competition_mask = self.comp_mask.repeat((batch_size, 1, 1)).to(self.device)
        relations = F.relu(self.relation_embedding(competition_mask))
        relations = relations.permute(0, 3, 1, 2)  # Move channels up
        relations = F.relu(self.relation_conv(relations))  # Pair demand representation

        # Phase pair competition
        combine_features = rotated_phases * relations
        combine_features = F.relu(self.hidden_layer(combine_features))  # Phase competition representation
        combine_features = self.before_merge(combine_features)  # Pairwise competition result

        # Phase score
        combine_features = torch.reshape(combine_features, (batch_size, self.oshape, self.oshape - 1))
        q_values = (lambda x: torch.sum(x, dim=2))(combine_features) # (b,8)
        return q_values
        

    def forward(self, states, train=True):
        if train:
            return self._forward(states)
        else:
            with torch.no_grad():
                return self._forward(states)

