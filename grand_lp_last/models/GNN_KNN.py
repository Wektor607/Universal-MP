import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch import nn
import torch.nn.functional as F
from models.base_classes import BaseGNN
from utils.model_configurations import set_block, set_function
from data_utils.graph_rewiring import KNN, add_edges, edge_sampling, GDCWrapper
from utils.utils import DummyData, get_full_adjacency

# Define the GNN model.
class GNN_KNN(BaseGNN):
  def __init__(self, opt, data, device=torch.device('cpu')):
    super(GNN_KNN, self).__init__(opt, data, device)
    self.f = set_function(opt)
    block = set_block(opt)
    time_tensor = torch.tensor([0, self.T]).to(device)
    self.odeblock = block(self.f, self.regularization_fns, opt, data, device, t=time_tensor).to(device)
    self.data_edge_index = data.edge_index.to(device)
    # self.fa = get_full_adjacency(self.num_nodes).to(device)

  def forward(self, x, pos_encoding):
    # Encode each node based on its feature.

    if self.opt['beltrami']:
      x = F.dropout(x, self.opt['input_dropout'], training=self.training)
      x = self.mx(x)
      if self.opt['dataset'] == 'ogbn-arxiv':
        p = pos_encoding
      else:
        p = F.dropout(pos_encoding, self.opt['input_dropout'], training=self.training)
        p = self.mp(p)
      print(x.shape)
      print(p.shape)
      x = torch.cat([x, p], dim=1)
    else:
      x = F.dropout(x, self.opt['input_dropout'], training=self.training)
      x = self.m1(x)

    if self.opt['use_mlp']:
      x = F.dropout(x, self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)

    # todo investigate if some input non-linearity solves the problem with smooth deformations identified in the ANODE paper
    # if True:
    #   x = F.relu(x)

    if self.opt['batch_norm']:
      x = self.bn_in(x)

    # Solve the initial value problem of the ODE.
    if self.opt['augment']:
      c_aux = torch.zeros(x.shape).to(self.device)
      x = torch.cat([x, c_aux], dim=1)

    self.odeblock.set_x0(x)

    if self.training and self.odeblock.nreg > 0:
      z, self.reg_states = self.odeblock(x)
    else:
      z = self.odeblock(x)

    if self.opt['fa_layer']:
      temp_time = self.opt['time']
      temp_method = self.opt['method']
      temp_step_size = self.opt['step_size']

      self.opt['time'] = 1 # self.opt['fa_layer_time'] #1.0
      self.opt['method'] = 'rk4' # self.opt['fa_layer_method']#'rk4'
      self.opt['step_size'] = 1 # self.opt['fa_layer_step_size']#1.0
      self.odeblock.set_x0(z)
      self.odeblock.odefunc.edge_index = add_edges(self, self.opt)
      if self.opt['edge_sampling_rmv'] != 0:
        edge_sampling(self, z, self.opt)

      z = self.odeblock(z)
      self.odeblock.odefunc.edge_index = self.data_edge_index

      self.opt['time'] = temp_time
      self.opt['method'] = temp_method
      self.opt['step_size'] = temp_step_size


    if self.opt['augment']:
      z = torch.split(z, x.shape[1] // 2, dim=1)[0]

    # if self.opt['batch_norm']:
    #   z = self.bn_in(z)
    
    return z

  def forward_encoder(self, x, pos_encoding):
    # Encode each node based on its feature.

    if self.opt['beltrami']:
      # x = F.dropout(x, self.opt['input_dropout'], training=self.training)
      x = self.mx(x)
      if self.opt['dataset'] == 'ogbn-arxiv':
        p = pos_encoding
      else:
        # p = F.dropout(pos_encoding, self.opt['input_dropout'], training=self.training)
        p = self.mp(pos_encoding)
      x = torch.cat([x, p], dim=1)
    else:
      # x = F.dropout(x, self.opt['input_dropout'], training=self.training)
      x = self.m1(x)

    if self.opt['use_mlp']:
      # x = F.dropout(x, self.opt['dropout'], training=self.training)
      # x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
      # x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)
      x = x + self.m11(F.relu(x))
      x = x + self.m12(F.relu(x))

    # todo investigate if some input non-linearity solves the problem with smooth deformations identified in the ANODE paper
    # if True:
    #   x = F.relu(x)
    # if self.opt['use_labels']:
    #   x = torch.cat([x, y], dim=-1)

    if self.opt['batch_norm']:
      x = self.bn_in(x)

    # Solve the initial value problem of the ODE.
    if self.opt['augment']:
      c_aux = torch.zeros(x.shape).to(self.device)
      x = torch.cat([x, c_aux], dim=1)

    return x

  def forward_ODE(self, x, pos_encoding):
    x = self.forward_encoder(x, pos_encoding)

    self.odeblock.set_x0(x)

    if self.training and self.odeblock.nreg > 0:
      z, self.reg_states = self.odeblock(x)
    else:
      z = self.odeblock(x)

    if self.opt['fa_layer']:
      temp_time = self.opt['time']
      temp_method = self.opt['method']
      temp_step_size = self.opt['step_size']

      self.opt['time'] = 1 # self.opt['fa_layer_time'] #1.0
      self.opt['method'] = 'rk4' # self.opt['fa_layer_method']#'rk4'
      self.opt['step_size'] = 1 # self.opt['fa_layer_step_size']#1.0
      self.odeblock.set_x0(z)
      self.odeblock.odefunc.edge_index = add_edges(self, self.opt)
      if self.opt['edge_sampling_rmv'] != 0:
        edge_sampling(self, z, self.opt)

      z = self.odeblock(z)
      self.odeblock.odefunc.edge_index = self.data_edge_index

      self.opt['time'] = temp_time
      self.opt['method'] = temp_method
      self.opt['step_size'] = temp_step_size


    if self.opt['augment']:
      z = torch.split(z, x.shape[1] // 2, dim=1)[0]

    return z
