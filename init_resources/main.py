# In w:\Transformers\Iter1\main.py

import os
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sentencepiece as spm
from multiprocessing import freeze_support
import math
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler

from decoder import MyDecoder

torch.cuda.empty_cache()
# -------------------------
# Hyperparameters & Paths
# -------------------------
training_examples_file = './data/text8_tokens_flat.pkl'
seq_length             = 256
d_model                = 256
num_layers             = 8
d_ff                   = 1024
batch_size             = 64
num_epochs             = 100
num_heads              = 8
learning_rate          = 3e-4
weight_decay           = 0.01 

# Early Stopping Configuration
early_stopping_patience = 5 # Stop after 5 epochs of no improvement

@torch.no_grad()
def evaluate(model, data, batch_size, device, vocab_size):
    """Calculates the validation loss for the model."""
    model.eval()
    total_loss = 0
    steps = 0
    pbar = tqdm(range(0, data.size(0), batch_size), desc="  Evaluating", unit="batch", leave=False)
    for i in pbar:
        batch_data = data[i : i + batch_size]
        inputs = batch_data[:, :-1]
        targets = batch_data[:, 1:]

        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        total_loss += loss.item()
        steps += 1
    
    model.train()
    return total_loss / max(1, steps)


def main():
    # -------------------------
    # Device & cuDNN tuning
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print("Using device:", device)

    # -------------------------
    # SentencePiece
    # -------------------------
    sp = spm.SentencePieceProcessor(model_file='./spm_vocab_text8_32k.model')
    vocab_size = sp.get_piece_size()
    print("Vocab size:", vocab_size)
    print("Cross-entropy baseline:", math.log(vocab_size))

    # -------------------------
    # Model, Optimizer, and Data
    # -------------------------
    model = MyDecoder(
        vocab_size=vocab_size,
        max_seq_length=seq_length,
        d_model=d_model,
        num_layers=num_layers,
        d_ff=d_ff,
        num_heads=num_heads,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=torch.cuda.is_available())
    scaler = GradScaler()

    with open(training_examples_file, 'rb') as f:
        raw = pickle.load(f)
    print(f"Loaded {len(raw)} training examples.")

    data_tensor = torch.tensor(raw, dtype=torch.long)
    num_examples = len(data_tensor) // (seq_length + 1)
    data_tensor = data_tensor[:num_examples * (seq_length + 1)]
    data_chunks = data_tensor.view(num_examples, seq_length + 1)
    
    val_size = int(0.05 * data_chunks.size(0))
    train_data = data_chunks[:-val_size].to(device, non_blocking=True)
    val_data = data_chunks[-val_size:].to(device, non_blocking=True)
    
    print(f"Created {train_data.size(0):,} training and {val_data.size(0):,} validation examples.")
    
    # -------------------------
    # Learning Rate Scheduler
    # -------------------------
    steps_per_epoch = math.ceil(train_data.size(0) / batch_size)
    warmup_steps = 2000
    total_steps  = steps_per_epoch * num_epochs

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    print("CUDA available:", torch.cuda.is_available())
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

    # --- Training State ---
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = None
    
    # Create a base filename with hyperparameters
    model_base_name = (
        f"GPT_s{seq_length}_d{d_model}_l{num_layers}_h{num_heads}"
    )

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0.0
        indices = torch.randperm(train_data.size(0))
        pbar = tqdm(range(0, train_data.size(0), batch_size), desc=f"Epoch {epoch}", unit="batch")

        for i in pbar:
            batch_indices = indices[i : i + batch_size]
            batch_data = train_data[batch_indices]
            inputs = batch_data[:, :-1]
            targets = batch_data[:, 1:]

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(inputs)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

        avg_train_loss = total_train_loss / steps_per_epoch
        avg_val_loss = evaluate(model, val_data, batch_size, device, vocab_size)
        
        print(f"Epoch {epoch} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Format the performance metrics to be filename-friendly
            train_loss_str = f"{avg_train_loss:.2f}".replace('.', '_')
            val_loss_str = f"{avg_val_loss:.2f}".replace('.', '_')
            
            # Combine base name (hyperparams) with dynamic metrics (performance)
            new_model_filename = (
                f"./models/{model_base_name}"
                f"_E{epoch}"
                f"_T{train_loss_str}"
                f"_V{val_loss_str}.pth"
            )

            print(f"  -> New best validation loss! Saving model to {new_model_filename}")
            torch.save(model.state_dict(), new_model_filename)

            # Delete the old best model if it exists
            if best_model_path and os.path.exists(best_model_path):
                print(f"  -> Deleting old best model: {best_model_path}")
                os.remove(best_model_path)
            
            # Update the path to track the current best model
            best_model_path = new_model_filename

        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{early_stopping_patience}")

        if patience_counter >= early_stopping_patience:
            print(f"\nStopping early. Validation loss has not improved for {early_stopping_patience} epochs.")
            break

    if best_model_path:
        print(f"\nTraining complete. Best model saved to: {best_model_path}")
    else:
        print("\nTraining complete, but no model was saved.")

if __name__ == "__main__":
    freeze_support()
    main()