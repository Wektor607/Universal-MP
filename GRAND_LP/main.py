
import argparse
import torch
import time
from tqdm import tqdm

from load_lp import *
from metrics import *
from graph_rewiring import apply_KNN
from base_classes import LinkPredictor
from GNN_KNN import GNN_KNN
from GNN_KNN_early import GNNKNNEarly
from GNN import GNN
from GNN_early import GNNEarly
from ogb.linkproppred import Evaluator
from torch.utils.data import DataLoader

# I copied the best parameters for the rest of the data sets from ogbn-arxiv
best_params_dict = {'Cora': {'M_nodes': 64, 'adaptive': False, 'add_source': True, 'adjoint': False, 'adjoint_method': 'adaptive_heun', 'adjoint_step_size': 1, 'alpha': 1.0, 'alpha_dim': 'sc', 'att_samp_pct': 1, 'attention_dim': 128, 'attention_norm_idx': 1, 'attention_rewiring': False, 'attention_type': 'scaled_dot', 'augment': False, 'baseline': False, 'batch_norm': False, 'beltrami': False, 'beta_dim': 'sc', 'block': 'attention', 'cpus': 1, 'data_norm': 'rw', 'dataset': 'Cora', 'decay': 0.00507685443154266, 'directional_penalty': None, 'dropout': 0.046878964627763316, 'dt': 0.001, 'dt_min': 1e-05, 'epoch': 100, 'exact': True, 'fc_out': False, 'feat_hidden_dim': 64, 'function': 'laplacian', 'gdc_avg_degree': 64, 'gdc_k': 64, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_threshold': 0.01, 'gpus': 0.5, 'grace_period': 20, 'heads': 8, 'heat_time': 3.0, 'hidden_dim': 80, 'input_dropout': 0.5, 'jacobian_norm2': None, 'kinetic_energy': None, 'label_rate': 0.5, 'leaky_relu_slope': 0.2, 'lr': 0.022924849756740397, 'max_epochs': 1000, 'max_iters': 100, 'max_nfe': 2000, 'method': 'dopri5', 'metric': 'accuracy', 'mix_features': False, 'name': 'cora_beltrami_splits', 'new_edges': 'random', 'no_alpha_sigmoid': False, 'not_lcc': True, 'num_init': 1, 'num_samples': 1000, 'num_splits': 2, 'ode_blocks': 1, 'optimizer': 'adamax', 'patience': 100, 'pos_enc_hidden_dim': 16, 'pos_enc_orientation': 'row', 'pos_enc_type': 'GDC', 'ppr_alpha': 0.05, 'reduction_factor': 10, 'regularise': False, 'reweight_attention': False, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 10, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'rewiring': None, 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'self_loop_weight': 1, 'sparsify': 'S_hat', 'square_plus': True, 'step_size': 1, 'threshold_type': 'addD_rvR', 'time': 18.294754260552843, 'tol_scale': 821.9773048827274, 'tol_scale_adjoint': 1.0, 'total_deriv': None, 'use_cora_defaults': False, 'use_flux': False, 'use_labels': False, 'use_lcc': True, 'use_mlp': False},
                    'Citeseer': {'M_nodes': 64, 'adaptive': False, 'add_source': True, 'adjoint': False, 'adjoint_method': 'adaptive_heun', 'adjoint_step_size': 1, 'alpha': 1.0, 'alpha_dim': 'sc', 'att_samp_pct': 1, 'attention_dim': 32, 'attention_norm_idx': 1, 'attention_rewiring': False, 'attention_type': 'exp_kernel', 'augment': False, 'baseline': False, 'batch_norm': False, 'beltrami': False, 'beta_dim': 'sc', 'block': 'attention', 'cpus': 1, 'data_norm': 'rw', 'dataset': 'Citeseer', 'decay': 0.1, 'directional_penalty': None, 'dropout': 0.7488085003122172, 'dt': 0.001, 'dt_min': 1e-05, 'epoch': 250, 'exact': True, 'fc_out': False, 'feat_hidden_dim': 64, 'function': 'laplacian', 'gdc_avg_degree': 64, 'gdc_k': 128, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_threshold': 0.01, 'gpus': 1.0, 'grace_period': 20, 'heads': 8, 'heat_time': 3.0, 'hidden_dim': 80, 'input_dropout': 0.6803233752085334, 'jacobian_norm2': None, 'kinetic_energy': None, 'label_rate': 0.5, 'leaky_relu_slope': 0.5825086997804176, 'lr': 0.00863585231323069, 'max_epochs': 1000, 'max_iters': 100, 'max_nfe': 3000, 'method': 'dopri5', 'metric': 'accuracy', 'mix_features': False, 'name': 'Citeseer_beltrami_1_KNN', 'new_edges': 'random', 'no_alpha_sigmoid': False, 'not_lcc': True, 'num_class': 6, 'num_feature': 3703, 'num_init': 2, 'num_nodes': 2120, 'num_samples': 400, 'num_splits': 1, 'ode_blocks': 1, 'optimizer': 'adam', 'patience': 100, 'pos_enc_dim': 'row', 'pos_enc_hidden_dim': 16, 'ppr_alpha': 0.05, 'reduction_factor': 4, 'regularise': False, 'reweight_attention': False, 'rewire_KNN': False, 'rewire_KNN_epoch': 10, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'rewiring': None, 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'self_loop_weight': 1, 'sparsify': 'S_hat', 'square_plus': True, 'step_size': 1, 'threshold_type': 'addD_rvR', 'time': 7.874113442879092, 'tol_scale': 2.9010446330432815, 'tol_scale_adjoint': 1.0, 'total_deriv': None, 'use_cora_defaults': False, 'use_flux': False, 'use_labels': False, 'use_lcc': True, 'use_mlp': False},
                    'Pubmed': {'M_nodes': 64, 'adaptive': False, 'add_source': True, 'adjoint': True, 'adjoint_method': 'adaptive_heun', 'adjoint_step_size': 1, 'alpha': 1.0, 'alpha_dim': 'sc', 'att_samp_pct': 1, 'attention_dim': 16, 'attention_norm_idx': 0, 'attention_rewiring': False, 'attention_type': 'cosine_sim', 'augment': False, 'baseline': False, 'batch_norm': False, 'beltrami': False, 'beta_dim': 'sc', 'block': 'attention', 'cpus': 1, 'data_norm': 'rw', 'dataset': 'Pubmed', 'decay': 0.0018236722171703636, 'directional_penalty': None, 'dropout': 0.07191100715473969, 'dt': 0.001, 'dt_min': 1e-05, 'epoch': 600, 'exact': False, 'fc_out': False, 'feat_hidden_dim': 64, 'function': 'laplacian', 'gdc_avg_degree': 64, 'gdc_k': 64, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_threshold': 0.01, 'gpus': 1.0, 'grace_period': 20, 'heads': 1, 'heat_time': 3.0, 'hidden_dim': 128, 'input_dropout': 0.5, 'jacobian_norm2': None, 'kinetic_energy': None, 'label_rate': 0.5, 'leaky_relu_slope': 0.2, 'lr': 0.014669345840305131, 'max_epochs': 1000, 'max_iters': 100, 'max_nfe': 5000, 'method': 'dopri5', 'metric': 'test_acc', 'mix_features': False, 'name': None, 'new_edges': 'random', 'no_alpha_sigmoid': False, 'not_lcc': True, 'num_init': 1, 'num_samples': 400, 'num_splits': 8, 'ode_blocks': 1, 'optimizer': 'adamax', 'patience': 100, 'pos_enc_dim': 'row', 'pos_enc_hidden_dim': 16, 'ppr_alpha': 0.05, 'reduction_factor': 10, 'regularise': False, 'reweight_attention': False, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 10, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'rewiring': None, 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'self_loop_weight': 1, 'sparsify': 'S_hat', 'square_plus': True, 'step_size': 1, 'threshold_type': 'addD_rvR', 'time': 12.942327880200853, 'tol_scale': 1991.0688305523001, 'tol_scale_adjoint': 16324.368093998313, 'total_deriv': None, 'use_cora_defaults': False, 'use_flux': False, 'use_labels': False, 'use_lcc': True, 'use_mlp': False, 'folder': 'pubmed_linear_att_beltrami_adj2', 'index': 0, 'run_with_KNN': False, 'change_att_sim_type': False, 'reps': 1, 'max_test_steps': 100, 'no_early': False, 'earlystopxT': 5.0, 'pos_enc_csv': False, 'pos_enc_type': 'GDC'},
                    'CoauthorCS': {'M_nodes': 64, 'adaptive': False, 'add_source': False, 'adjoint': True, 'adjoint_method': 'dopri5', 'adjoint_step_size': 1, 'alpha': 1.0, 'alpha_dim': 'sc', 'att_samp_pct': 1, 'attention_dim': 8, 'attention_norm_idx': 1, 'attention_rewiring': False, 'attention_type': 'scaled_dot', 'augment': False, 'baseline': False, 'batch_norm': False, 'beltrami': False, 'beta_dim': 'sc', 'block': 'attention', 'cpus': 1, 'data_norm': 'rw', 'dataset': 'CoauthorCS', 'decay': 0.004738413087298854, 'directional_penalty': None, 'dropout': 0.6857774850321, 'dt': 0.001, 'dt_min': 1e-05, 'edge_sampling': False, 'edge_sampling_T': 'T0', 'edge_sampling_add': 0.05, 'edge_sampling_epoch': 5, 'edge_sampling_online': False, 'edge_sampling_online_reps': 4, 'edge_sampling_rmv': 0.05, 'edge_sampling_space': 'pos_distance', 'edge_sampling_sym': False, 'epoch': 250, 'exact': False, 'fa_layer': False, 'fc_out': False, 'feat_hidden_dim': 128, 'function': 'laplacian', 'gdc_avg_degree': 64, 'gdc_k': 64, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_threshold': 0.0001, 'gpus': 1, 'grace_period': 20, 'heads': 4, 'heat_time': 3.0, 'hidden_dim': 16, 'input_dropout': 0.5275042493231822, 'jacobian_norm2': None, 'kinetic_energy': None, 'label_rate': 0.5, 'leaky_relu_slope': 0.7181389780997276, 'lr': 0.0009342860080741642, 'max_iters': 100, 'max_nfe': 3000, 'method': 'dopri5', 'metric': 'accuracy', 'mix_features': False, 'name': 'CoauthorCS_final_tune_posencGDC', 'new_edges': 'random', 'no_alpha_sigmoid': False, 'not_lcc': True, 'num_init': 1, 'num_samples': 400, 'num_splits': 4, 'ode_blocks': 1, 'optimizer': 'rmsprop', 'pos_dist_quantile': 0.001, 'pos_enc_csv': False, 'pos_enc_hidden_dim': 32, 'pos_enc_orientation': 'row', 'pos_enc_type': 'GDC', 'ppr_alpha': 0.05, 'reduction_factor': 10, 'regularise': False, 'reweight_attention': False, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 5, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'rewiring': None, 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'self_loop_weight': 0, 'sparsify': 'S_hat', 'square_plus': True, 'step_size': 1, 'symmetric_attention': False, 'threshold_type': 'addD_rvR', 'time': 3.126400580172773, 'tol_scale': 9348.983916372074, 'tol_scale_adjoint': 6599.1250595331385, 'total_deriv': None, 'use_cora_defaults': False, 'use_flux': False, 'use_labels': False, 'use_mlp': False},
                    'Computers': {'M_nodes': 64, 'adaptive': False, 'add_source': False, 'adjoint': True, 'adjoint_method': 'dopri5', 'adjoint_step_size': 1, 'alpha': 1.0, 'alpha_dim': 'sc', 'att_samp_pct': 0.572918052062338, 'attention_dim': 64, 'attention_norm_idx': 0, 'attention_rewiring': False, 'attention_type': 'scaled_dot', 'augment': False, 'baseline': False, 'batch_norm': False, 'beltrami': False, 'beta_dim': 'sc', 'block': 'hard_attention', 'cpus': 1, 'data_norm': 'rw', 'dataset': 'Computers', 'decay': 0.007674669913252157, 'directional_penalty': None, 'dropout': 0.08732611854459256, 'dt': 0.001, 'dt_min': 1e-05, 'epoch': 100, 'exact': False, 'fc_out': False, 'feat_hidden_dim': 64, 'function': 'laplacian', 'gdc_avg_degree': 64, 'gdc_k': 64, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_threshold': 0.01, 'gpus': 1.0, 'grace_period': 25, 'heads': 4, 'heat_time': 3.0, 'hidden_dim': 128, 'input_dropout': 0.5973137276937647, 'jacobian_norm2': None, 'kinetic_energy': None, 'label_rate': 0.5, 'leaky_relu_slope': 0.2, 'lr': 0.0035304663972281548, 'max_epochs': 1000, 'max_iters': 100, 'max_nfe': 500, 'method': 'dopri5', 'metric': 'accuracy', 'mix_features': False, 'name': 'computer_beltrami_hard_att1', 'new_edges': 'random', 'no_alpha_sigmoid': False, 'not_lcc': True, 'num_init': 1, 'num_samples': 400, 'num_splits': 2, 'ode_blocks': 1, 'optimizer': 'adam', 'patience': 100, 'pos_enc_hidden_dim': 32, 'pos_enc_orientation': 'row', 'pos_enc_type': 'DW128', 'ppr_alpha': 0.05, 'reduction_factor': 10, 'regularise': False, 'reweight_attention': False, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 10, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'rewiring': None, 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'self_loop_weight': 1.7138583550928912, 'sparsify': 'S_hat', 'square_plus': False, 'step_size': 1, 'threshold_type': 'addD_rvR', 'time': 3.249016177876166, 'tol_scale': 127.46369887079446, 'tol_scale_adjoint': 443.81436775321754, 'total_deriv': None, 'use_cora_defaults': False, 'use_flux': False, 'use_labels': False, 'use_mlp': False},
                    'Photo': {'M_nodes': 64, 'adaptive': False, 'add_source': False, 'adjoint': True, 'adjoint_method': 'rk4', 'adjoint_step_size': 1, 'alpha': 1.0, 'alpha_dim': 'sc', 'att_samp_pct': 0.9282359956104751, 'attention_dim': 64, 'attention_norm_idx': 0, 'attention_rewiring': False, 'attention_type': 'pearson', 'augment': False, 'baseline': False, 'batch_norm': True, 'beltrami': False, 'beta_dim': 'sc', 'block': 'hard_attention', 'cpus': 1, 'data_norm': 'rw', 'dataset': 'Photo', 'decay': 0.004707800883497945, 'directional_penalty': None, 'dropout': 0.46502284638600183, 'dt': 0.001, 'dt_min': 1e-05, 'epoch': 100, 'exact': False, 'fc_out': False, 'feat_hidden_dim': 64, 'function': 'laplacian', 'gdc_avg_degree': 64, 'gdc_k': 64, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_threshold': 0.01, 'gpus': 1.0, 'grace_period': 25, 'heads': 4, 'heat_time': 3.0, 'hidden_dim': 64, 'input_dropout': 0.42903126506740247, 'jacobian_norm2': None, 'kinetic_energy': None, 'label_rate': 0.5, 'leaky_relu_slope': 0.2, 'lr': 0.005560726683883279, 'max_epochs': 1000, 'max_iters': 100, 'max_nfe': 500, 'method': 'dopri5', 'metric': 'accuracy', 'mix_features': False, 'name': 'photo_beltrami_hard_att1', 'new_edges': 'random', 'no_alpha_sigmoid': False, 'not_lcc': True, 'num_init': 1, 'num_samples': 400, 'num_splits': 2, 'ode_blocks': 1, 'optimizer': 'adam', 'patience': 100, 'pos_enc_hidden_dim': 16, 'pos_enc_orientation': 'row', 'pos_enc_type': 'DW128', 'ppr_alpha': 0.05, 'reduction_factor': 10, 'regularise': False, 'reweight_attention': False, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 10, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'rewiring': None, 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'self_loop_weight': 0.05783612585280118, 'sparsify': 'S_hat', 'square_plus': False, 'step_size': 1, 'threshold_type': 'addD_rvR', 'time': 3.5824027975386623, 'tol_scale': 2086.525473167121, 'tol_scale_adjoint': 14777.606112557354, 'total_deriv': None, 'use_cora_defaults': False, 'use_flux': False, 'use_labels': False, 'use_mlp': False},
                    'ogbn-arxiv': {'M_nodes': 64, 'adaptive': False, 'add_source': False, 'adjoint': True, 'adjoint_method': 'rk4', 'adjoint_step_size': 1, 'alpha': 1.0, 'alpha_dim': 'sc', 'att_samp_pct': 0.8105268910037231, 'attention_dim': 32, 'attention_norm_idx': 0, 'attention_rewiring': False, 'attention_type': 'scaled_dot', 'augment': False, 'baseline': False, 'batch_norm': True, 'beltrami': False, 'beta_dim': 'sc', 'block': 'hard_attention', 'cpus': 1, 'data_norm': 'rw', 'dataset': 'ogbn-arxiv', 'decay': 0, 'directional_penalty': None, 'dropout': 0.11594990901233933, 'dt': 0.001, 'dt_min': 1e-05, 'epoch': 100, 'exact': False, 'fc_out': False, 'feat_hidden_dim': 64, 'function': 'laplacian', 'gdc_avg_degree': 64, 'gdc_k': 64, 'gdc_method': 'ppr', 'gdc_sparsification': 'topk', 'gdc_threshold': 0.01, 'gpus': 1.0, 'grace_period': 20, 'heads': 2, 'heat_time': 3.0, 'hidden_dim': 162, 'input_dropout': 0, 'jacobian_norm2': None, 'kinetic_energy': None, 'label_rate': 0.21964773835397075, 'leaky_relu_slope': 0.2, 'lr': 0.005451476553977102, 'max_epochs': 1000, 'max_iters': 100, 'max_nfe': 500, 'method': 'dopri5', 'metric': 'accuracy', 'mix_features': False, 'name': 'arxiv_beltrami_hard_att', 'new_edges': 'random', 'no_alpha_sigmoid': False, 'not_lcc': False, 'num_init': 2, 'num_samples': 200, 'num_splits': 1, 'ode_blocks': 1, 'optimizer': 'rmsprop', 'patience': 100, 'pos_enc_hidden_dim': 98, 'pos_enc_orientation': 'row', 'pos_enc_type': 'DW64', 'ppr_alpha': 0.05, 'reduction_factor': 10, 'regularise': False, 'reweight_attention': False, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 10, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'rewiring': None, 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'self_loop_weight': 1, 'sparsify': 'S_hat', 'square_plus': False, 'step_size': 1, 'threshold_type': 'addD_rvR', 'time': 3.6760155951687636, 'tol_scale': 11353.558848254957, 'tol_scale_adjoint': 1.0, 'total_deriv': None, 'use_cora_defaults': False, 'use_flux': False, 'use_labels': False, 'use_lcc': True, 'use_mlp': False},

                    'ogbl-collab': {'M_nodes': 64, 'adaptive': False, 'add_source': False, 'adjoint': True, 'adjoint_method': 'rk4', 'adjoint_step_size': 1, 'alpha': 1.0, 'alpha_dim': 'sc', 'att_samp_pct': 0.8105268910037231, 'attention_dim': 32, 'attention_norm_idx': 0, 'attention_rewiring': False, 'attention_type': 'scaled_dot', 'augment': False, 'baseline': False, 'batch_norm': True, 'beltrami': False, 'beta_dim': 'sc', 'block': 'hard_attention', 'cpus': 1, 'data_norm': 'rw', 'dataset': 'ogbl-collab', 'decay': 0, 'directional_penalty': None, 'dropout': 0.11594990901233933, 'dt': 0.001, 'dt_min': 1e-05, 'epoch': 100, 'exact': False, 'fc_out': False, 'feat_hidden_dim': 64, 'function': 'laplacian', 'gdc_avg_degree': 64, 'gdc_k': 64, 'gdc_method': 'ppr', 'gdc_sparsification': 'threshold', 'gdc_threshold': 0.01, 'gpus': 1.0, 'grace_period': 20, 'heads': 2, 'heat_time': 3.0, 'hidden_dim': 162, 'input_dropout': 0, 'jacobian_norm2': None, 'kinetic_energy': None, 'label_rate': 0.21964773835397075, 'leaky_relu_slope': 0.2, 'lr': 0.005451476553977102, 'max_epochs': 1000, 'max_iters': 100, 'max_nfe': 500, 'method': 'dopri5', 'metric': 'accuracy', 'mix_features': False, 'name': 'arxiv_beltrami_hard_att', 'new_edges': 'random', 'no_alpha_sigmoid': False, 'not_lcc': False, 'num_init': 2, 'num_samples': 200, 'num_splits': 1, 'ode_blocks': 1, 'optimizer': 'rmsprop', 'patience': 100, 'pos_enc_hidden_dim': 98, 'pos_enc_orientation': 'row', 'pos_enc_type': 'DW64', 'ppr_alpha': 0.05, 'reduction_factor': 10, 'regularise': False, 'reweight_attention': False, 'rewire_KNN': False, 'rewire_KNN_T': 'T0', 'rewire_KNN_epoch': 10, 'rewire_KNN_k': 64, 'rewire_KNN_sym': False, 'rewiring': 'gdc', 'rw_addD': 0.02, 'rw_rmvR': 0.02, 'self_loop_weight': 1, 'sparsify': 'S_hat', 'square_plus': False, 'step_size': 1, 'threshold_type': 'addD_rvR', 'time': 3.6760155951687636, 'tol_scale': 11353.558848254957, 'tol_scale_adjoint': 1.0, 'total_deriv': None, 'use_cora_defaults': False, 'use_flux': False, 'use_labels': False, 'use_lcc': True, 'use_mlp': False}
                    }

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def merge_cmd_args(cmd_opt, opt):
  if cmd_opt['beltrami']:
    opt['beltrami'] = True
  if cmd_opt['function'] is not None:
    opt['function'] = cmd_opt['function']
  if cmd_opt['block'] is not None:
    opt['block'] = cmd_opt['block']
  if cmd_opt['attention_type'] != 'scaled_dot':
    opt['attention_type'] = cmd_opt['attention_type']
  if cmd_opt['self_loop_weight'] is not None:
    opt['self_loop_weight'] = cmd_opt['self_loop_weight']
  if cmd_opt['method'] is not None:
    opt['method'] = cmd_opt['method']
  if cmd_opt['step_size'] != 1:
    opt['step_size'] = cmd_opt['step_size']
  if cmd_opt['time'] != 1:
    opt['time'] = cmd_opt['time']
  if cmd_opt['epoch'] != 100:
    opt['epoch'] = cmd_opt['epoch']
  if not cmd_opt['not_lcc']:
    opt['not_lcc'] = False
  if cmd_opt['num_splits'] != 1:
    opt['num_splits'] = cmd_opt['num_splits']

def get_optimizer(name, parameters, lr, weight_decay=0):
  if name == 'sgd':
    return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'rmsprop':
    return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adagrad':
    return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adam':
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
  elif name == 'adamax':
    return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
  else:
    raise Exception("Unsupported optimizer: {}".format(name))

def print_model_params(model):
  print(model)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)
      print(param.data.shape)
      
@torch.no_grad()
def test_OGB(model, predictor, pos_encoding, data, splits, opt):
  batch_size = opt['batch_size']
  
  if opt['dataset'] == 'ogbn-arxiv':
    name = 'ogbn-arxiv'
  elif opt['dataset'] == 'ogbl-collab':
    name = 'ogbl-collab'

  evaluator = Evaluator(name=name)
  model.eval()
  h = model(data.x, pos_encoding)
  
  pos_train_edge = splits['train']['pos_edge_label_index'].to(data.x.device)
  neg_train_edge = splits['train']['neg_edge_label_index'].to(data.x.device)
  pos_valid_edge = splits['valid']['pos_edge_label_index'].to(data.x.device)
  neg_valid_edge = splits['valid']['neg_edge_label_index'].to(data.x.device)
  pos_test_edge = splits['test']['pos_edge_label_index'].to(data.x.device)
  neg_test_edge = splits['test']['neg_edge_label_index'].to(data.x.device)
  
  pos_train_preds = []
  for perm in DataLoader(range(pos_train_edge.size(1)), batch_size):
      edge = pos_train_edge[:, perm]#.t()
      pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
  pos_train_pred = torch.cat(pos_train_preds, dim=0)

  neg_train_preds = []
  for perm in DataLoader(range(neg_train_edge.size(1)), batch_size):
      edge = neg_train_edge[:, perm]#.t()
      neg_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
  neg_train_pred = torch.cat(neg_train_preds, dim=0)

  pos_valid_preds = []
  for perm in DataLoader(range(pos_valid_edge.size(1)), batch_size):
      edge = pos_valid_edge[:, perm]#.t()
      pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
  pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

  neg_valid_preds = []
  for perm in DataLoader(range(neg_valid_edge.size(1)), batch_size):
      edge = neg_valid_edge[:, perm]#.t()
      neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
  neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

  pos_test_preds = []
  for perm in DataLoader(range(pos_test_edge.size(1)), batch_size):
      edge = pos_test_edge[:, perm]#.t()
      pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
  pos_test_pred = torch.cat(pos_test_preds, dim=0)

  neg_test_preds = []
  for perm in DataLoader(range(neg_test_edge.size(1)), batch_size):
      edge = neg_test_edge[:, perm]#.t()
      neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
  neg_test_pred = torch.cat(neg_test_preds, dim=0)

  results = {}
  for K in [1, 3, 10, 20, 50, 100]:
      evaluator.K = K
      train_hits = evaluator.eval({
          'y_pred_pos': pos_train_pred,
          'y_pred_neg': neg_train_pred,
      })[f'hits@{K}']
      valid_hits = evaluator.eval({
          'y_pred_pos': pos_valid_pred,
          'y_pred_neg': neg_valid_pred,
      })[f'hits@{K}']
      test_hits = evaluator.eval({
          'y_pred_pos': pos_test_pred,
          'y_pred_neg': neg_test_pred,
      })[f'hits@{K}']

      results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

      # # Log the hits@K values
      # writer.add_scalar(f'Accuracy/Train_Hits@{K}', train_hits, epoch)
      # writer.add_scalar(f'Accuracy/Valid_Hits@{K}', valid_hits, epoch)
      # writer.add_scalar(f'Accuracy/Test_Hits@{K}', test_hits, epoch)
  
  print(f"Shape of pos_val_pred: {pos_test_pred.shape}")
  print(f"Shape of neg_val_pred: {neg_test_pred.shape}")

  result_mrr_test = evaluate_mrr(pos_test_pred, neg_test_pred)  
  
  for name in ['MRR', 'mrr_hit1', 'mrr_hit3', 'mrr_hit10', 'mrr_hit20', 'mrr_hit50', 'mrr_hit100']:
      results[name] = (result_mrr_test[name])
      # writer.add_scalar(f'Accuracy/Test_{name}', result_mrr_test[name], epoch)
  
  test_pred = torch.cat([pos_test_pred, neg_test_pred])
  test_true = torch.cat([torch.ones(pos_test_pred.size(0), dtype=int), 
                          torch.zeros(neg_test_pred.size(0), dtype=int)])
  
  result_auc_test = evaluate_auc(test_pred, test_true)
  for name in ['AUC', 'AP']:
      results[name] = (result_auc_test[name])
      # writer.add_scalar(f'Accuracy/Test_{name}',result_auc_test[name], epoch)

  result_acc_test = acc(pos_test_pred, neg_test_pred)
  results['ACC'] = (result_acc_test)
  # writer.add_scalar(f'Accuracy/Test_ACC',result_acc_test, epoch)
  
  return results

def train(model, predictor, pos_encoding, data, splits, optimizer, batch_size):
    predictor.train()
    model.train()
    
    pos_encoding = pos_encoding.to(model.device) if pos_encoding is not None else None
    
    pos_train_edge = splits['train']['pos_edge_label_index'].to(data.x.device)
    neg_train_edge = splits['train']['neg_edge_label_index'].to(data.x.device)
    
    total_loss = 0
    data_loader = DataLoader(range(pos_train_edge.size(1)), batch_size, shuffle=True)
    
    with tqdm(data_loader, desc="Training Progress", unit="batch") as pbar:
        for perm in pbar:
            optimizer.zero_grad()
            h = model(data.x, pos_encoding)
            
            edge = pos_train_edge[:, perm]
            pos_out = predictor(h[edge[0]], h[edge[1]])
            pos_loss = -torch.log(pos_out + 1e-15).mean()
        
            edge = neg_train_edge[:, perm]
            neg_out = predictor(h[edge[0]], h[edge[1]])
            neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        
            loss = pos_loss + neg_loss
            
            if model.odeblock.nreg > 0:
                reg_states = tuple(torch.mean(rs) for rs in model.reg_states)
                regularization_coeffs = model.regularization_coeffs

                reg_loss = sum(
                    reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
                )
                loss = loss + reg_loss

            # Update parameters
            model.fm.update(model.getNFE())
            model.resetNFE()
            loss.backward()
            optimizer.step()
            model.bm.update(model.getNFE())
            model.resetNFE()
            
            total_loss += loss.item()
            # Update progress bar description with current loss
            pbar.set_postfix({"Loss": loss.item()})
    
    return total_loss

def grand_dataset(data, splits, opt):
    edge_index = data.edge_index
    data.num_nodes = data.x.shape[0]
    # data.edge_weight = None
    data.adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    # data.max_x = -1
    # Use training + validation edges for inference on test set.
    if opt['use_valedges_as_input']:
        val_edge_index = splits['valid']['pos_edge_label_index']
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
        data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t
    return data
  
def random_sampling(splits, scale: float):
    print(f"train adj shape: {splits['train'].edge_index.shape[1]}")

    for k, data in splits.items():
        if k!='train':
            print(f"{k}: original length {data.pos_edge_label_index.shape[1]}")
            num_samples = int(data.neg_edge_label_index.shape[1] * scale)
            sampled_indices = np.random.choice(data.neg_edge_label_index.shape[1], num_samples, replace=False)
            data.pos_edge_label_index = data.pos_edge_label_index[:, sampled_indices]
            data.neg_edge_label_index = data.neg_edge_label_index[:, sampled_indices]
            data.neg_edge_label = data.neg_edge_label[sampled_indices]
            data.pos_edge_label = data.pos_edge_label[sampled_indices]
            print(f"{k}: downsampled length {data.pos_edge_label_index.shape[1]}")

    return splits
  
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    ### MPLP PARAMETERS ###
    # dataset setting
    parser.add_argument('--dataset', type=str, default='ogbl-collab')
    # parser.add_argument('--val_ratio', type=float, default=0.1)
    # parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--dataset_dir', type=str, default='./data')
    parser.add_argument('--use_valedges_as_input', type=str2bool, default='False', help='whether to use val edges as input')
    parser.add_argument('--year', type=int, default=-1)

    # MPLP settings
    # parser.add_argument('--signature_dim', type=int, default=1024, help="the node signature dimension `F` in MPLP")
    # parser.add_argument('--minimum_degree_onehot', type=int, default=-1, help='the minimum degree of hubs with onehot encoding to reduce variance')
    # parser.add_argument('--mask_target', type=str2bool, default='True', help='whether to mask the target edges to remove the shortcut')
    # parser.add_argument('--use_degree', type=str, default='none', choices=["none","mlp","AA","RA"], help="rescale vector norm to facilitate weighted count")
    # parser.add_argument('--signature_sampling', type=str, default='torchhd', help='whether to use torchhd to randomize vectors', choices=["torchhd","gaussian","onehot"])
    # parser.add_argument('--fast_inference', type=str2bool, default='False', help='whether to enable a faster inference by caching the node vectors')
    # parser.add_argument('--adj2', type=str2bool, default="False", help='Whether to use 2-hop adj for MPLP+prop_only.')
    
    # model setting
    # parser.add_argument('--predictor', type=str, default='MPLP', choices=["inner","mlp","ENL",
    # "MPLP+exact","MPLP+","MPLP","MPLP+precompute"])
    # parser.add_argument('--encoder', type=str, default='gcn')
    # parser.add_argument('--hidden_channels', type=int, default=256)
    # parser.add_argument('--xdp', type=float, default=0.2)
    # parser.add_argument('--feat_dropout', type=float, default=0.5)
    # parser.add_argument('--label_dropout', type=float, default=0.5)
    # parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_feature', type=str2bool, default='True', help='whether to use node features as input')
    # parser.add_argument('--feature_combine', type=str, default='hadamard', choices=['hadamard','plus_minus'], help='how to represent a link with two nodes features')
    # parser.add_argument('--jk', type=str2bool, default='True', help='whether to use Jumping Knowledge')
    # parser.add_argument('--batchnorm_affine', type=str2bool, default='True', help='whether to use Affine in BatchNorm')
    # parser.add_argument('--use_embedding', type=str2bool, default='False', help='whether to train node embedding')
    # parser.add_argument('--dgcnn', type=str2bool, default='False', help='whether to use DGCNN as the target edge pooling')

    # training setting
    # parser.add_argument('--batch_size', type=int, default=64 * 1024)
    # parser.add_argument('--test_batch_size', type=int, default=100000)
    # parser.add_argument('--epochs', type=int, default=20000)
    # parser.add_argument('--num_neg', type=int, default=1)
    # parser.add_argument('--num_hops', type=int, default=2)
    # parser.add_argument('--lr', type=float, default=0.005)
    # parser.add_argument('--weight_decay', type=float, default=0)
    # parser.add_argument('--log_steps', type=int, default=20)
    # parser.add_argument('--patience', type=int, default=100, help='number of patience steps for early stopping')
    # parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--metric', type=str, default='Hits@50', help='main evaluation metric')

    # misc
    # parser.add_argument('--log_dir', type=str, default='./logs')
    # parser.add_argument('--data_split_only', type=str2bool, default='False')
    parser.add_argument('--print_summary', type=str, default='')

    ### GRAND PARAMETERS ###
    parser.add_argument('--use_cora_defaults', action='store_true',
                    help='Whether to run with best params for cora. Overrides the choice of dataset')
    # data args
    parser.add_argument('--data_norm', type=str, default='rw',
                    help='rw for random walk, gcn for symmetric gcn norm')
    parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
    parser.add_argument('--use_labels', dest='use_labels', action='store_true', help='Also diffuse labels')
    parser.add_argument('--geom_gcn_splits', dest='geom_gcn_splits', action='store_true',
                    help='use the 10 fixed splits from '
                        'https://arxiv.org/abs/2002.05287')
    parser.add_argument('--num_splits', type=int, dest='num_splits', default=1,
                    help='the number of splits to repeat the results on')
    parser.add_argument('--label_rate', type=float, default=0.5,
                    help='% of training labels to use when --use_labels is set.')
    parser.add_argument('--planetoid_split', action='store_true',
                    help='use planetoid splits for Cora/Citeseer/Pubmed')
    # GNN args
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension.')
    parser.add_argument('--fc_out', dest='fc_out', action='store_true',
                    help='Add a fully connected layer to the decoder.')
    parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument("--batch_norm", dest='batch_norm', action='store_true', help='search over reg params')
    parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
    parser.add_argument('--epoch', type=int, default=100, help='Number of training epochs per iteration.')
    parser.add_argument('--alpha', type=float, default=1.0, help='Factor in front matrix A.')
    parser.add_argument('--alpha_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) alpha')
    parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                    help='apply sigmoid before multiplying by alpha')
    parser.add_argument('--beta_dim', type=str, default='sc', help='choose either scalar (sc) or vector (vc) beta')
    parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, hard_attention')
    parser.add_argument('--function', type=str, default='laplacian', help='laplacian, transformer, dorsey, GAT')
    parser.add_argument('--use_mlp', dest='use_mlp', action='store_true',
                    help='Add a fully connected layer to the encoder.')
    parser.add_argument('--add_source', dest='add_source', action='store_true',
                    help='If try get rid of alpha param and the beta*x0 source term')
    parser.add_argument('--cgnn', dest='cgnn', action='store_true', help='Run the baseline CGNN model from ICML20')

    # ODE args
    parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
    parser.add_argument('--augment', action='store_true',
                    help='double the length of the feature vector by appending zeros to stabilist ODE learning')
    parser.add_argument('--method', type=str, help="set the numerical solver: dopri5, euler, rk4, midpoint")
    parser.add_argument('--step_size', type=float, default=1,
                    help='fixed step size when using fixed step solvers e.g. rk4')
    parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
    parser.add_argument("--adjoint_method", type=str, default="adaptive_heun",
                    help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
    parser.add_argument('--adjoint', dest='adjoint', action='store_true',
                    help='use the adjoint ODE method to reduce memory footprint')
    parser.add_argument('--adjoint_step_size', type=float, default=1,
                    help='fixed step size when using fixed step adjoint solvers e.g. rk4')
    parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
    parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                    help="multiplier for adjoint_atol and adjoint_rtol")
    parser.add_argument('--ode_blocks', type=int, default=1, help='number of ode blocks to run')
    parser.add_argument("--max_nfe", type=int, default=1000,
                    help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")
    parser.add_argument("--no_early", action="store_true",
                    help="Whether or not to use early stopping of the ODE integrator when testing.")
    parser.add_argument('--earlystopxT', type=float, default=3, help='multiplier for T used to evaluate best model')
    parser.add_argument("--max_test_steps", type=int, default=100,
                    help="Maximum number steps for the dopri5Early test integrator. "
                        "used if getting OOM errors at test time")

    # Attention args
    parser.add_argument('--leaky_relu_slope', type=float, default=0.2,
                    help='slope of the negative part of the leaky relu used in attention')
    parser.add_argument('--attention_dropout', type=float, default=0., help='dropout of attention weights')
    parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
    parser.add_argument('--attention_norm_idx', type=int, default=0, help='0 = normalise rows, 1 = normalise cols')
    parser.add_argument('--attention_dim', type=int, default=64,
                    help='the size to project x to before calculating att scores')
    parser.add_argument('--mix_features', dest='mix_features', action='store_true',
                    help='apply a feature transformation xW to the ODE')
    parser.add_argument('--reweight_attention', dest='reweight_attention', action='store_true',
                    help="multiply attention scores by edge weights before softmax")
    parser.add_argument('--attention_type', type=str, default="scaled_dot",
                    help="scaled_dot,cosine_sim,pearson, exp_kernel")
    parser.add_argument('--square_plus', action='store_true', help='replace softmax with square plus')

    # regularisation args
    parser.add_argument('--jacobian_norm2', type=float, default=None, help="int_t ||df/dx||_F^2")
    parser.add_argument('--total_deriv', type=float, default=None, help="int_t ||df/dt||^2")

    parser.add_argument('--kinetic_energy', type=float, default=None, help="int_t ||f||_2^2")
    parser.add_argument('--directional_penalty', type=float, default=None, help="int_t ||(df/dx)^T f||^2")

    # rewiring args
    parser.add_argument("--not_lcc", action="store_false", help="don't use the largest connected component")
    parser.add_argument('--rewiring', type=str, default=None, help="two_hop, gdc")
    parser.add_argument('--gdc_method', type=str, default='ppr', help="ppr, heat, coeff")
    # topk is not implemented in the original implementation
    parser.add_argument('--gdc_sparsification', type=str, default='threshold', help="threshold, topk")
    parser.add_argument('--gdc_k', type=int, default=64, help="number of neighbours to sparsify to when using topk")
    parser.add_argument('--gdc_threshold', type=float, default=0.0001,
                    help="obove this edge weight, keep edges when using threshold")
    parser.add_argument('--gdc_avg_degree', type=int, default=64,
                    help="if gdc_threshold is not given can be calculated by specifying avg degree")
    parser.add_argument('--ppr_alpha', type=float, default=0.05, help="teleport probability")
    parser.add_argument('--heat_time', type=float, default=3., help="time to run gdc heat kernal diffusion for")
    parser.add_argument('--att_samp_pct', type=float, default=1,
                    help="float in [0,1). The percentage of edges to retain based on attention scores")
    parser.add_argument('--use_flux', dest='use_flux', action='store_true',
                    help='incorporate the feature grad in attention based edge dropout')
    parser.add_argument("--exact", action="store_true",
                    help="for small datasets can do exact diffusion. If dataset is too big for matrix inversion then you can't")
    parser.add_argument('--M_nodes', type=int, default=64, help="new number of nodes to add")
    parser.add_argument('--new_edges', type=str, default="random", help="random, random_walk, k_hop")
    parser.add_argument('--sparsify', type=str, default="S_hat", help="S_hat, recalc_att")
    parser.add_argument('--threshold_type', type=str, default="topk_adj", help="topk_adj, addD_rvR")
    parser.add_argument('--rw_addD', type=float, default=0.02, help="percentage of new edges to add")
    parser.add_argument('--rw_rmvR', type=float, default=0.02, help="percentage of edges to remove")
    parser.add_argument('--rewire_KNN', action='store_true', help='perform KNN rewiring every few epochs')
    parser.add_argument('--rewire_KNN_T', type=str, default="T0", help="T0, TN")
    parser.add_argument('--rewire_KNN_epoch', type=int, default=5, help="frequency of epochs to rewire")
    parser.add_argument('--rewire_KNN_k', type=int, default=64, help="target degree for KNN rewire")
    parser.add_argument('--rewire_KNN_sym', action='store_true', help='make KNN symmetric')
    parser.add_argument('--KNN_online', action='store_true', help='perform rewiring online')
    parser.add_argument('--KNN_online_reps', type=int, default=4, help="how many online KNN its")
    parser.add_argument('--KNN_space', type=str, default="pos_distance", help="Z,P,QKZ,QKp")
    # beltrami args
    parser.add_argument('--beltrami', action='store_true', help='perform diffusion beltrami style')
    parser.add_argument('--fa_layer', action='store_true', help='add a bottleneck paper style layer with more edges')
    parser.add_argument('--pos_enc_type', type=str, default="DW64",
                    help='positional encoder either GDC, DW64, DW128, DW256')
    parser.add_argument('--pos_enc_orientation', type=str, default="row", help="row, col")
    parser.add_argument('--feat_hidden_dim', type=int, default=64, help="dimension of features in beltrami")
    parser.add_argument('--pos_enc_hidden_dim', type=int, default=32, help="dimension of position in beltrami")
    parser.add_argument('--edge_sampling', action='store_true', help='perform edge sampling rewiring')
    parser.add_argument('--edge_sampling_T', type=str, default="T0", help="T0, TN")
    parser.add_argument('--edge_sampling_epoch', type=int, default=5, help="frequency of epochs to rewire")
    parser.add_argument('--edge_sampling_add', type=float, default=0.64, help="percentage of new edges to add")
    parser.add_argument('--edge_sampling_add_type', type=str, default="importance",
                    help="random, ,anchored, importance, degree")
    parser.add_argument('--edge_sampling_rmv', type=float, default=0.32, help="percentage of edges to remove")
    parser.add_argument('--edge_sampling_sym', action='store_true', help='make KNN symmetric')
    parser.add_argument('--edge_sampling_online', action='store_true', help='perform rewiring online')
    parser.add_argument('--edge_sampling_online_reps', type=int, default=4, help="how many online KNN its")
    parser.add_argument('--edge_sampling_space', type=str, default="attention",
                    help="attention,pos_distance, z_distance, pos_distance_QK, z_distance_QK")
    parser.add_argument('--symmetric_attention', action='store_true',
                    help='maks the attention symmetric for rewring in QK space')

    parser.add_argument('--fa_layer_edge_sampling_rmv', type=float, default=0.8, help="percentage of edges to remove")
    parser.add_argument('--gpu', type=int, default=0, help="GPU to run on (default 0)")
    parser.add_argument('--pos_enc_csv', action='store_true', help="Generate pos encoding as a sparse CSV")

    parser.add_argument('--pos_dist_quantile', type=float, default=0.001, help="percentage of N**2 edges to keep")

    # MY PARAMETERS
    parser.add_argument('--mlp_num_layers', type=int, default=3, help="Number of layers in MLP")
    parser.add_argument('--batch_size', type=int, default=2048)
    
    args = parser.parse_args()
    
    cmd_opt = vars(args)
    try:
        best_opt = best_params_dict[cmd_opt['dataset']]
        opt = {**cmd_opt, **best_opt}
        merge_cmd_args(cmd_opt, opt)
    except KeyError:
        opt = cmd_opt
    
    data, splits = get_dataset(opt['dataset_dir'], opt['dataset'], opt, opt['use_valedges_as_input'], opt['year'])
      
    if args.dataset == "ogbl-citation2":
        opt['metric'] = "MRR"
    if data.x is None:
        opt['use_feature'] = False

    if args.print_summary:
        data_summary(args.dataset, data, header='header' in args.print_summary, latex='latex' in args.print_summary);exit(0)
    else:
        print(args)
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # splits = random_sampling(splits, 0.1)

    # data.edge_index = splits['train']['pos_edge_label_index']
    # data = grand_dataset(data, splits, opt).to(device)
    data = data.to(device)
    if opt['beltrami']:
      pos_encoding = apply_beltrami(data, opt).to(device)
      opt['pos_enc_dim'] = pos_encoding.shape[1]
    else:
      pos_encoding = None
    
    num_features = opt['hidden_dim']
    predictor = LinkPredictor(num_features, opt['hidden_dim'], 1, opt['mlp_num_layers'], opt['dropout']).to(device)
    batch_size = opt['batch_size']    
    if opt['rewire_KNN'] or opt['fa_layer']:
      model = GNN_KNN(opt, data, splits, predictor, batch_size, device).to(device) if opt["no_early"] else GNNKNNEarly(opt, data, splits, predictor, batch_size, device).to(device)
    else:
      model = GNN(opt, data, splits, predictor, batch_size, device).to(device) if opt["no_early"] else GNNEarly(opt, data, splits, predictor, batch_size, device).to(device)

    parameters = [p for p in model.parameters() if p.requires_grad]
    # print_model_params(model)
    optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
    best_time = best_epoch = train_acc = val_acc = test_acc = 0

    this_test = test_OGB #if opt['dataset'] == 'ogbn-arxiv' else test
  
    for epoch in range(1, opt['epoch']):
      start_time = time.time()

      if opt['rewire_KNN'] and epoch % opt['rewire_KNN_epoch'] == 0 and epoch != 0:
        ei = apply_KNN(data, pos_encoding, model, opt)
        model.odeblock.odefunc.edge_index = ei

      loss = train(model, predictor, pos_encoding, data, splits, optimizer, opt['batch_size'])
      print(f'Loss Epoch {epoch}: ', loss)
      # if epoch % 1 == 0:
      results = this_test(model, predictor, pos_encoding, data, splits, opt)

      best_time = opt['time']
      # if tmp_val_acc > val_acc:
      #   best_epoch = epoch
      #   train_acc = tmp_train_acc
      #   val_acc = tmp_val_acc
      #   test_acc = tmp_test_acc
      #   best_time = opt['time']
      # if not opt['no_early'] and model.odeblock.test_integrator.solver.best_val > val_acc:
      #   best_epoch = epoch
      #   val_acc = model.odeblock.test_integrator.solver.best_val
      #   test_acc = model.odeblock.test_integrator.solver.best_test
      #   train_acc = model.odeblock.test_integrator.solver.best_train
      #   best_time = model.odeblock.test_integrator.solver.best_time

      # log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Best time: {:.4f}'
      log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Best time: {:.4f}, Results: {}'

      print(log.format(epoch, time.time() - start_time, loss, model.fm.sum, model.bm.sum, best_time, results))
      
  
    print('best val accuracy {:03f} with test accuracy {:03f} at epoch {:d} and best time {:03f}'.format(val_acc, test_acc,
                                                                                                      best_epoch,                                                                                               best_time))
    print(train_acc, val_acc, test_acc)