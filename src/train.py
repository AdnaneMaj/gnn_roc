from tqdm import tqdm
import os

import torch
import torch.nn.functional as F
from torch.optim import Adam

#Configuration
BATCH_SIZE = 1024

def compute_bpr_loss(E, batch, edge_index, k=1):
    """
    Compute the BPR loss for a batch of users.

    Args:
        E (torch.Tensor): Embedding matrix of shape (n_users + n_items, embedding_dim).
        batch (torch.Tensor): Batch of user indices of shape (batch_size,).
        edge_index (torch.Tensor): Edge index in COO format of shape (2, N).
        k (int): Number of negative samples per positive pair.

    Returns:
        torch.Tensor: BPR loss for the batch.
    """
    batch_loss = 0.0

    for u in batch:
        # Find positive items (items interacted with by user u)
        mask = edge_index[0] == u
        pos_items = edge_index[1, mask]

        # Sample one positive item per user
        if len(pos_items) == 0:
            continue  # Skip users with no positive interactions
        pos_item = pos_items[torch.randint(0, len(pos_items), (1,))]

        # Find negative items (items not interacted with by user u)
        non_neighbors_mask = ~mask
        neg_items = edge_index[1, non_neighbors_mask]

        # Sample k negative items
        neg_items = neg_items[torch.randint(0, len(neg_items), (k,))]

        # Compute scores for positive and negative items
        u_embedding = E[u]  # User embedding
        pos_embedding = E[pos_item]  # Positive item embedding
        neg_embeddings = E[neg_items]  # Negative item embeddings

        # Compute predicted scores
        pos_score = torch.sum(u_embedding * pos_embedding, dim=-1)  # Dot product
        neg_scores = torch.sum(u_embedding.unsqueeze(0) * neg_embeddings, dim=-1)  # Dot product

        # Compute BPR loss for this user
        for neg_score in neg_scores:
            bpr_loss = -F.logsigmoid(pos_score - neg_score)
            batch_loss += bpr_loss

    # Normalize by the number of users and negative samples
    batch_loss = batch_loss / (len(batch) * k)

    return batch_loss


def train_step(model, optimizer, edge_index, batch, k=1):
    """
    Perform a training step with BPR loss.

    Args:
        model: The LightGCN model.
        optimizer: The optimizer.
        edge_index (torch.Tensor): Edge index in COO format of shape (2, N).
        batch (torch.Tensor): Batch of user indices of shape (batch_size,).
        k (int): Number of negative samples per positive pair.

    Returns:
        torch.Tensor: BPR loss for the batch.
    """
    # Forward pass: compute embeddings
    E0, E = model.forward(x=None, edge_index=edge_index)

    # Compute BPR loss
    batch_loss = compute_bpr_loss(E=E, batch=batch, edge_index=edge_index, k=k)

    # Backpropagation and optimization
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    return batch_loss

def train(model, epochs, batches, edge_index, split='train', k=5, device=None, checkpoint_dir='checkpoints', checkpoint_freq=1):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if not device else torch.device(device)

    # Move the model and the data to device
    model = model.to(device)
    edge_index = edge_index[split].to(device)

    optimizer = Adam(params=model.E0.parameters(), lr=0.0001)

    losses = []

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch_idx in range(epochs):

        print(f"Epoch {epoch_idx + 1}/{epochs}")

        epoch_loss = 0  # To accumulate loss for the epoch

        # Wrap the batch loop with tqdm for a progress bar
        with tqdm(batches, desc=f"Epoch {epoch_idx + 1}", unit="batch") as pbar:
            for batch in pbar:
                
                batch_loss = train_step(model=model, optimizer=optimizer, edge_index=edge_index, batch=batch, k=k)

                # Accumulate epoch loss
                epoch_loss += batch_loss.item()
                losses.append(batch_loss.item())

                # Update progress bar with current batch loss
                pbar.set_postfix(batch_loss=f"{batch_loss.item():.9f}")

        print(f"Epoch {epoch_idx + 1} Loss: {epoch_loss:.9f}")

        # Save checkpoint at the end of each epoch or at the specified frequency
        if (epoch_idx + 1) % checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch_idx + 1}.pt')
            torch.save({
                'epoch': epoch_idx + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    return losses
