import torch
import torch.nn.functional as F
import sentencepiece as spm
import argparse

from decoder import MyDecoder # Import the model definition from your decoder.py

# -------------------------
# Argument Parser
# -------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Inference script for the GPT-style Transformer")
    parser.add_argument('--prompt', type=str, required=True, help="The initial text prompt to start generation.")
    
    # --- FIX: Updated default model path ---
    parser.add_argument('--model_path', type=str, default='./GPT_model_best.pth', help="Path to the trained model weights.")
    
    parser.add_argument('--spm_model_path', type=str, default='./spm_vocab_text8_32k.model', help="Path to the SentencePiece model file.")
    parser.add_argument('--max_new_tokens', type=int, default=100, help="Maximum number of new tokens to generate.")
    parser.add_argument('--temperature', type=float, default=0.8, help="Controls randomness. Higher is more random, lower is more deterministic.")
    parser.add_argument('--top_k', type=int, default=40, help="Sample from the top k most likely tokens. 0 to disable.")
    
    # --- FIX: Model Hyperparameters now match your trained model ---
    parser.add_argument('--seq_length', type=int, default=256, help="Max sequence length of the model.")
    parser.add_argument('--d_model', type=int, default=512, help="Dimension of the model.")
    parser.add_argument('--num_layers', type=int, default=8, help="Number of decoder layers.")
    parser.add_argument('--d_ff', type=int, default=2048, help="Dimension of the feed-forward network.")
    parser.add_argument('--num_heads', type=int, default=8, help="Number of attention heads.")
    
    return parser.parse_args()

# -------------------------
# Generation Function
# -------------------------
@torch.no_grad() # Use decorator for efficiency
def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k, device, max_seq_length):
    """
    Generates text using the model with temperature and top-k sampling.
    """
    model.eval()
    prompt_tokens = tokenizer.encode(prompt, out_type=int)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    print(f"--- Prompt --- \n{prompt}\n")
    print(f"--- Generating {max_new_tokens} tokens ---")

    for _ in range(max_new_tokens):
        # Crop context if it exceeds max_seq_length
        context_ids = input_ids if input_ids.size(1) <= max_seq_length else input_ids[:, -max_seq_length:]

        # Get model logits
        logits = model(context_ids)
        last_token_logits = logits[:, -1, :] 

        # Apply temperature
        if temperature > 0:
            last_token_logits /= temperature

        # Apply Top-K sampling
        if top_k > 0:
            v, _ = torch.topk(last_token_logits, min(top_k, last_token_logits.size(-1)))
            last_token_logits[last_token_logits < v[:, [-1]]] = -float('Inf')

        # Convert logits to probabilities and sample
        probs = F.softmax(last_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, next_token), dim=1)

    # Decode the generated token IDs back to text
    generated_tokens = input_ids[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text

# -------------------------
# Main Execution Block
# -------------------------
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        tokenizer = spm.SentencePieceProcessor(model_file=args.spm_model_path)
        vocab_size = tokenizer.get_piece_size()
        print(f"Tokenizer loaded. Vocab size: {vocab_size}")
    except Exception as e:
        print(f"Error loading SentencePiece model: {e}")
        return

    # --- FIX: Instantiate model with all correct arguments from args ---
    model = MyDecoder(
        vocab_size=vocab_size,
        max_seq_length=args.seq_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        num_heads=args.num_heads
    ).to(device)

    # Load trained model weights
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model weights loaded successfully from {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}.")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        # This error often means the model architecture doesn't match the saved weights.
        return

    # Generate text
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
        max_seq_length=args.seq_length
    )

    print("\n--- Full Generated Text ---")
    print(generated_text)


if __name__ == "__main__":
    main()