import torch
from torch import nn
import torch.nn.functional as F

class FRAP(nn.Module):
    def __init__(self, **kwargs):
        super(FRAP, self).__init__()
        self.device = kwargs['device']
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

    def forward(self, states):
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
                extended_acts.append(zeros)
            extended_acts = torch.stack(extended_acts)
        else:
            extended_acts = acts
        phase_embeds = torch.sigmoid(self.p(extended_acts))
        phase_demands = []
        for i in range(num_movements):
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
        competition_mask = self.comp_mask.repeat((batch_size, 1, 1))
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
