import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score


def evaluate_auc(val_pred, val_true):
    valid_auc = roc_auc_score(val_true, val_pred)
    
    results = {}
    
    valid_auc = round(valid_auc, 4)

    results['AUC'] = valid_auc

    valid_ap = average_precision_score(val_true, val_pred)
    
    valid_ap = round(valid_ap, 4)
    
    results['AP'] = valid_ap


    return results

def eval_mrr(y_pred_pos, y_pred_neg):
    '''
        compute mrr
        y_pred_neg is an array with shape (batch size, num_entities_neg).
        y_pred_pos is an array with shape (batch size, )
    '''

    # calculate ranks
    y_pred_pos = y_pred_pos.view(-1, 1)
    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    # print(y_pred_pos.shape)
    # print(y_pred_neg.shape)
    optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
    
    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    
    pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
    
    hits1_list = (ranking_list <= 1).to(torch.float)
    hits3_list = (ranking_list <= 3).to(torch.float)

    hits20_list = (ranking_list <= 20).to(torch.float)
    hits50_list = (ranking_list <= 50).to(torch.float)
    hits10_list = (ranking_list <= 10).to(torch.float)
    hits100_list = (ranking_list <= 100).to(torch.float)
    mrr_list = 1./ranking_list.to(torch.float)

    return { 'hits@1_list': hits1_list,
                'hits@3_list': hits3_list,
                'hits@20_list': hits20_list,
                'hits@50_list': hits50_list,
                'hits@10_list': hits10_list,
                'hits@100_list': hits100_list,
                'mrr_list': mrr_list}



def evaluate_mrr(pos_val_pred, neg_val_pred, opt):
    
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    if opt['dataset'] == 'ogbl-ppa':
        mrr_output =  eval_mrr_batch(pos_val_pred, neg_val_pred, opt['batch_size'])
    else:
        mrr_output =  eval_mrr(pos_val_pred, neg_val_pred)
        
    valid_mrr=mrr_output['mrr_list'].mean().item()
    valid_mrr_hit1 = mrr_output['hits@1_list'].mean().item()
    valid_mrr_hit3 = mrr_output['hits@3_list'].mean().item()
    valid_mrr_hit10 = mrr_output['hits@10_list'].mean().item()

    valid_mrr_hit20 = mrr_output['hits@20_list'].mean().item()
    valid_mrr_hit50 = mrr_output['hits@50_list'].mean().item()
    valid_mrr_hit100 = mrr_output['hits@100_list'].mean().item()

    valid_mrr = round(valid_mrr, 4)
    valid_mrr_hit1 = round(valid_mrr_hit1, 4)
    valid_mrr_hit3 = round(valid_mrr_hit3, 4)
    valid_mrr_hit10 = round(valid_mrr_hit10, 4)

    valid_mrr_hit20 = round(valid_mrr_hit20, 4)
    valid_mrr_hit50 = round(valid_mrr_hit50, 4)
    valid_mrr_hit100 = round(valid_mrr_hit100, 4)
    
    results = {}
    results['mrr_hit1'] = valid_mrr_hit1
    results['mrr_hit3'] = valid_mrr_hit3
    results['mrr_hit10'] = valid_mrr_hit10

    results['MRR'] = valid_mrr

    results['mrr_hit20'] = valid_mrr_hit20
    results['mrr_hit50'] = valid_mrr_hit50
    results['mrr_hit100'] = valid_mrr_hit100

    
    return results

def acc(pos_pred, neg_pred):
    hard_thres = (max(torch.max(pos_pred).item(), torch.max(neg_pred).item()) + min(torch.min(pos_pred).item(), torch.min(neg_pred).item())) / 2

    # Initialize predictions with zeros and set ones where condition is met
    y_pred = torch.zeros_like(pos_pred)
    y_pred[pos_pred >= hard_thres] = 1

    # Do the same for negative predictions
    neg_y_pred = torch.ones_like(neg_pred)
    neg_y_pred[neg_pred <= hard_thres] = 0

    # Concatenate the positive and negative predictions
    y_pred = torch.cat([y_pred, neg_y_pred], dim=0)

    # Initialize ground truth labels
    pos_y = torch.ones_like(pos_pred)
    neg_y = torch.zeros_like(neg_pred)
    y = torch.cat([pos_y, neg_y], dim=0)
    y_logits = torch.cat([pos_pred, neg_pred], dim=0)
    # Calculate accuracy    
    return (y == y_pred).float().mean().item()