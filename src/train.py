from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.optim import Adam

#Configuration
BATCH_SIZE = 1024

# Function to compute batch loss
import torch
import torch.nn.functional as F

def compute_batch_loss(E, batch, edge_index, k=5):
    batch_neighbors = []
    batch_neg_samples = []

    for u in batch:
        # Find neighbors
        mask = edge_index[0] == u
        N_u = edge_index[1, mask]

        # Find non-neighbors and sample negatives
        non_neighbors_mask = ~mask
        non_neighbors = edge_index[1, non_neighbors_mask]
        neg_samples = non_neighbors[torch.randint(0, len(non_neighbors), (k,))]

        batch_neighbors.append(N_u)
        batch_neg_samples.append(neg_samples)

    # Pad neighbors to the same length
    max_neighbors = max(len(n) for n in batch_neighbors)
    padded_neighbors = torch.stack([
        F.pad(n, (0, max_neighbors - len(n)), value=-1)
        for n in batch_neighbors
    ])

    # Valid mask for padded neighbors
    valid_mask = padded_neighbors != -1

    # Compute embeddings
    neighbor_embeds = E[padded_neighbors]  # Assuming E is predefined
    neg_sample_embeds = torch.stack([E[neg] for neg in batch_neg_samples])

    # Compute similarities and log probabilities
    similarities = torch.bmm(neighbor_embeds, neg_sample_embeds.transpose(1, 2))
    log_probs = F.log_softmax(similarities, dim=-1)

    # Compute loss (normalized by the number of valid entries)
    batch_loss = -log_probs[valid_mask].mean()

    return batch_loss


def train_step(model,optimizer,edge_index,batch,k):

        E0, E = model.forward(x=None, edge_index=edge_index)

        # Compute batch loss
        batch_loss = compute_batch_loss(batch=batch, E=E,edge_index=edge_index,k=k)

        # Backpropagation and optimization
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        return batch_loss

def train(model,epochs,batches,edge_index,split='train',k=5,device=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if not device else torch.device(device)

    #Move the model and the data to device
    model = model.to(device)
    edge_index = edge_index[split].to(device)

    optimizer = Adam(params=model.E0.parameters(),lr=0.0001)

    losses = []

    for epoch_idx in range(epochs):

        print(f"Epoch {epoch_idx + 1}/{epochs}")

        epoch_loss = 0  # To accumulate loss for the epoch

        # Wrap the batch loop with tqdm for a progress bar
        with tqdm(batches, desc=f"Epoch {epoch_idx + 1}", unit="batch") as pbar:
            for batch in pbar:
                
                batch_loss = train_step(model=model,optimizer=optimizer,edge_index=edge_index,batch=batch,k=k)

                # Accumulate epoch loss
                epoch_loss += batch_loss.item()
                losses.append(batch_loss.item())

                # Update progress bar with current batch loss
                pbar.set_postfix(batch_loss=f"{batch_loss.item():.9f}")

        print(f"Epoch {epoch_idx + 1} Loss: {epoch_loss:.9f}")

    return losses
