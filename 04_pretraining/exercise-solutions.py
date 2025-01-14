import torch
from supplementary import GPTModel, generate_text_simple
import tiktoken




class GPTTextGenerator:
    def __init__(self, config, model_path, device=None):
        self.config = config
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GPTModel(self.config)
        self._load_model_weights(model_path)
        self.model.to(self.device)
        self.tokenizer = tiktoken.get_encoding("gpt2")
    
    def _load_model_weights(self, model_path):
        weights = torch.load(model_path)
        self.model.load_state_dict(weights)

    def text_to_token_ids(self, text):
        encoded = self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Add batch dimension
        return encoded_tensor

    def token_ids_to_text(self, token_ids):
        flat = token_ids.squeeze(0)  # Remove batch dimension
        return self.tokenizer.decode(flat.tolist())

    def generate_text(self, start_context, max_new_tokens=10):
        token_ids = generate_text_simple(
            model=self.model,
            idx=self.text_to_token_ids(start_context).to(self.device),
            max_new_tokens=max_new_tokens,
            context_size=self.config["context_length"]
        )
        return self.token_ids_to_text(token_ids)


if __name__ == "__main__":
    torch.manual_seed(123)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    model_path = "model.pth"
    generator = GPTTextGenerator(config=GPT_CONFIG_124M, model_path=model_path)

    start_context = "Every effort moves you"
    output_text = generator.generate_text(start_context, max_new_tokens=10)

    print("Output text:\n", output_text)
 
