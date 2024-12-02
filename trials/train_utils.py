import os
import sys

# Add the parent directory to the system path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)


# PyTorch and related libraries
import torch
import torch.nn.functional as F


def train(encoder, decoder,  optimizer, data, splits, device, batch_size=512):
    encoder.train()
    decoder.train()
    total_loss = 0
    # compare train loop with train_batch and see the difference and reanalyse
    pos_edge_index = splits['train']['pos_edge_label_index'].to(device)
    neg_edge_index = splits['train']['neg_edge_label_index'].to(device)
    pos_edge_label = splits['train']['pos_edge_score'].to(device)
    neg_edge_label = splits['train']['neg_edge_score'].to(device)

    num_batches = (pos_edge_index.size(1) + batch_size - 1) // batch_size
        
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, pos_edge_index.size(1))

        batch_pos_edge_index = pos_edge_index[:, start_idx:end_idx]
        batch_neg_edge_index = neg_edge_index[:, start_idx:end_idx]
        batch_pos_edge_label = pos_edge_label[start_idx:end_idx]
        batch_neg_edge_label = neg_edge_label[start_idx:end_idx]

        optimizer.zero_grad()

        batch_edge_index = torch.cat(
            [batch_pos_edge_index, batch_neg_edge_index], dim=1
        )

        z = encoder(
            data.x, batch_edge_index
        )

        pos_pred = decoder(
            z[batch_pos_edge_index[0]], z[batch_pos_edge_index[1]]
        ).squeeze()
        neg_pred = decoder(
            z[batch_neg_edge_index[0]], z[batch_neg_edge_index[1]]
        ).squeeze()

        pos_loss = F.mse_loss(pos_pred, batch_pos_edge_label)
        neg_loss = F.mse_loss(neg_pred, batch_neg_edge_label)
        batch_loss = pos_loss + neg_loss

        batch_loss.backward()

        total_loss += batch_loss.item()

        optimizer.step()

    return total_loss / num_batches


@torch.no_grad()
def valid(model, data, splits, device, batch_size=512):
    model.eval()

    total_loss = 0

    pos_edge_index = splits['valid']['pos_edge_label_index'].to(device)
    neg_edge_index = splits['valid']['neg_edge_label_index'].to(device)
    pos_edge_label = splits['valid']['pos_edge_score'].to(device)
    neg_edge_label = splits['valid']['neg_edge_score'].to(device)

    num_batches = (pos_edge_index.size(1) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, pos_edge_index.size(1))

        batch_pos_edge_index = pos_edge_index[:, start_idx:end_idx]
        batch_neg_edge_index = neg_edge_index[:, start_idx:end_idx]
        batch_pos_edge_label = pos_edge_label[start_idx:end_idx]
        batch_neg_edge_label = neg_edge_label[start_idx:end_idx]

        batch_edge_index = torch.cat(
            [batch_pos_edge_index, batch_neg_edge_index], dim=1
        )

        z = model.encode(data.x, batch_edge_index)

        pos_pred = model.decode(
            z[batch_pos_edge_index[0]], z[batch_pos_edge_index[1]]
            ).squeeze()
        neg_pred = model.decode(
            z[batch_neg_edge_index[0]], z[batch_neg_edge_index[1]]
            ).squeeze()

        pos_loss = F.mse_loss(pos_pred, batch_pos_edge_label)
        neg_loss = F.mse_loss(neg_pred, batch_neg_edge_label)
        batch_loss = pos_loss + neg_loss

        total_loss += batch_loss.item()

    return total_loss / num_batches


@torch.no_grad()
def test(model, data, splits, device, batch_size=512):
    model.eval()

    # Positive and negative edges for test
    total_loss = 0
    pos_edge_index = splits['test']['pos_edge_label_index'].to(device)
    neg_edge_index = splits['test']['neg_edge_label_index'].to(device)
    pos_edge_label = splits['test']['pos_edge_score'].to(device)
    neg_edge_label = splits['test']['neg_edge_score'].to(device)

    num_batches = (pos_edge_index.size(1) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, pos_edge_index.size(1))

        batch_pos_edge_index = pos_edge_index[:, start_idx:end_idx]
        batch_neg_edge_index = neg_edge_index[:, start_idx:end_idx]
        batch_pos_edge_label = pos_edge_label[start_idx:end_idx]
        batch_neg_edge_label = neg_edge_label[start_idx:end_idx]

        batch_edge_index = torch.cat(
            [batch_pos_edge_index, batch_neg_edge_index], dim=1
        )
        
        z = model.encode(data.x, batch_edge_index)

        pos_pred = model.decode(
            z[batch_pos_edge_index[0]], z[batch_pos_edge_index[1]]
            ).squeeze()
        neg_pred = model.decode(
            z[batch_neg_edge_index[0]], z[batch_neg_edge_index[1]]
            ).squeeze()

        pos_loss = F.mse_loss(pos_pred, batch_pos_edge_label)
        neg_loss = F.mse_loss(neg_pred, batch_neg_edge_label)
        batch_loss = pos_loss + neg_loss

        total_loss += batch_loss.item()

    return total_loss / num_batches

