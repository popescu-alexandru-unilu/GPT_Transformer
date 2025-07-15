import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(torch.nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0)) # Add batch dimension for easier broadcasting

    def forward(self, x):
        # x is [B, T, D]
        return x + self.pe[:, :x.size(1)]

class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = torch.nn.Linear(d_model, d_ff)
        self.linear2 = torch.nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        # --- OPTIMIZATION 4: Use GELU ---
        return self.linear2(F.gelu(self.linear1(x)))


# --- MAJOR CHANGES HERE ---

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads):
        super(DecoderLayer, self).__init__()
        # Use Post-LN architecture for better stability
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Post-LN: Operation -> Add -> Norm
        attn_output, _ = self.self_attn(
            x, x, x,
            is_causal=True, # Use the efficient, built-in causal masking
            attn_mask=None
        )
        x = self.norm1(x + attn_output) # Add & Norm

        ff_output = self.ff(x)
        x = self.norm2(x + ff_output) # Add & Norm
        
        return x
    
# In w:\Transformers\Iter1\decoder.py

class MyDecoder(nn.Module):
    def __init__(self, vocab_size, max_seq_length, d_model, num_layers, d_ff, num_heads):
        super(MyDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(max_seq_length, d_model)
        
        # --- FIX 1: Use TransformerEncoderLayer as the building block for GPT ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            batch_first=True, # IMPORTANT
            activation='gelu'
        )
        
        # --- FIX 2: Stack the ENCODER layers ---
        self.transformer_stack = nn.TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=num_layers
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size, bias=False)

        # --- OPTIMIZATION: Weight Tying ---
        # Share the weights between the embedding layer and the final output layer
        self.fc_out.weight = self.embed.weight

        # Your excellent initialization code
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        seq_len = x.size(1)
        
        # --- FIX 3: Generate the causal mask needed to make the encoder act like a decoder ---
        # Note: In PyTorch 2.0+, you can pass `is_causal=True` directly to the stack,
        # but creating the mask manually is more explicit and compatible.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)

        x = self.embed(x)
        x = self.pos_enc(x)
        
        # --- FIX 4: Pass the input as `src` and provide the `causal_mask` ---
        output = self.transformer_stack(
            src=x,
            mask=causal_mask
        )

        return self.fc_out(output)
