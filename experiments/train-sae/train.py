import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math

class TopKSAE(nn.Module):
    """Top-K Sparse Autoencoder with Tied Weights."""
    def __init__(self, d_model, n_features):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features
        
        # Encoder weights (Decoder will use W_enc.T)
        self.W_enc = nn.Parameter(torch.empty(d_model, n_features))
        # Kaiming Uniform initialization as per standard practices
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def forward(self, x, k):
        # 1. Pre-activation (subtract decoder bias first)
        # x: [batch, d_model], W_enc: [d_model, n_features]
        z = torch.matmul(x - self.b_dec, self.W_enc) + self.b_enc
        
        # 2. Top-K Activation
        z = F.relu(z)
        topk_values, topk_indices = torch.topk(z, k=k, dim=-1)
        
        f = torch.zeros_like(z)
        f.scatter_(-1, topk_indices, topk_values)
        
        # 3. Decode using tied weights (W_enc.T)
        x_hat = torch.matmul(f, self.W_enc.t()) + self.b_dec
        
        return x_hat, f

class Trainer:
    def __init__(self, config, llm, sae, tokenizer, dataset):
        self.config = config
        self.llm = llm
        self.sae = sae
        self.tokenizer = tokenizer
        self.dataset = dataset
        
        self.device = llm.device
        self.layer_idx = config['layer_index']
        
        self.optimizer = torch.optim.Adam(
            self.sae.parameters(),
            lr=config['lr'],
            eps=config['eps'],
            betas=(config['beta1'], config['beta2'])
        )
        
        # Mixed Precision
        self.use_amp = config['mixed_precision']
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # Logging
        os.makedirs(config['log_dir'], exist_ok=True)
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        
        # Calculate decay steps (50% of first epoch)
        # 1 epoch = (total_tokens / sae_batch_tokens) updates
        # We estimate total tokens by taking num_examples * avg_tokens (approx 150)
        estimated_total_tokens = len(self.dataset['train']) * 150 
        total_updates_per_epoch = estimated_total_tokens // config['sae_batch_tokens']
        self.decay_updates = max(1, int(total_updates_per_epoch * 0.5))
        
        self.global_update_step = 0

    def get_current_k(self):
        """Linearly decay K from k_init to k_final over decay_updates."""
        if self.global_update_step >= self.decay_updates:
            return self.config['k_final']
        
        progress = self.global_update_step / self.decay_updates
        k_diff = self.config['k_init'] - self.config['k_final']
        current_k = self.config['k_init'] - (k_diff * progress)
        return int(round(current_k))

    def train(self):
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        def collate_fn(batch):
            texts = [item['text'] for item in batch]
            return self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=self.config['max_seq_len'], 
                return_tensors="pt"
            )

        train_loader = DataLoader(
            self.dataset['train'], 
            batch_size=self.config['llm_batch_size'], 
            shuffle=True, 
            collate_fn=collate_fn
        )

        for epoch in range(self.config['epochs']):
            print(f"\n--- Epoch {epoch+1}/{self.config['epochs']} ---")
            
            token_buffer = []
            buffer_token_count = 0
            
            pbar = tqdm(train_loader, desc="Processing LLM batches")
            
            for batch in pbar:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                
                # 1. Extract hidden states from LLM
                with torch.no_grad():
                    outputs = self.llm(**inputs, output_hidden_states=True)
                    # hidden_states[0] is embeddings, so index layer_idx is the exact layer output
                    hidden_states = outputs.hidden_states[self.layer_idx]
                
                # 2. Filter out padding tokens using attention mask
                mask = inputs['attention_mask'].bool()
                valid_hidden_states = hidden_states[mask] # Shape: [num_valid_tokens, d_model]
                
                # Convert to FP32 for buffer to prevent overflow during SAE training
                token_buffer.append(valid_hidden_states.float())
                buffer_token_count += valid_hidden_states.size(0)
                
                # 3. Train SAE when buffer reaches target size (~90k tokens)
                if buffer_token_count >= self.config['sae_batch_tokens']:
                    # Concatenate all tokens in buffer
                    all_tokens = torch.cat(token_buffer, dim=0)
                    
                    # We might have slightly more than sae_batch_tokens. 
                    # Take exactly sae_batch_tokens for this update.
                    train_tokens = all_tokens[:self.config['sae_batch_tokens']]
                    
                    # Keep the remainder for the next update
                    remainder = all_tokens[self.config['sae_batch_tokens']:]
                    token_buffer = [remainder] if remainder.size(0) > 0 else []
                    buffer_token_count = remainder.size(0)
                    
                    # --- SAE Optimization Step ---
                    current_k = self.get_current_k()
                    loss = self.train_step_sae(train_tokens, current_k)
                    
                    # Logging
                    self.writer.add_scalar("Loss/Train", loss, self.global_update_step)
                    self.writer.add_scalar("Hyperparams/K", current_k, self.global_update_step)
                    
                    pbar.set_postfix({"Loss": f"{loss:.4f}", "K": current_k})
                    self.global_update_step += 1

            # Save checkpoint after each epoch
            ckpt_path = os.path.join(self.config['output_dir'], f"sae_epoch_{epoch+1}.pt")
            torch.save(self.sae.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        self.writer.close()
        print("Training Complete!")

    
    def train_step_sae(self, tokens, k):
        self.sae.train()
        self.optimizer.zero_grad()
        
        micro_batch_size = 4096 
        num_tokens = tokens.size(0)
        total_loss = 0.0
        
        for i in range(0, num_tokens, micro_batch_size):
            micro_batch = tokens[i:i + micro_batch_size]
            
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                # tokens shape: [4096, 1024]
                reconstructed, features = self.sae(micro_batch, k)
                
                # MSE Loss
                loss = F.mse_loss(reconstructed, micro_batch)
                
                weight = micro_batch.size(0) / num_tokens
                scaled_loss = loss * weight
                
            self.scaler.scale(scaled_loss).backward()
            
            total_loss += scaled_loss.item()
            
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return total_loss