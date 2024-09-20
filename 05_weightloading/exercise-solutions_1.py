import torch
import numpy as np
import tiktoken
from gpt_download import download_and_load_gpt2
from supplementary import GPTModel, generate_text_simple, text_to_token_ids, token_ids_to_text


class GPTHandler:
    """
    Handles the GPT model setup, configuration, and weight loading for text generation.

    Attributes:
        config (dict): Configuration parameters for the GPT model.
        model (GPTModel): The initialized GPT model.
        device (torch.device): The device on which the model will be loaded (CPU or GPU).
        tokenizer (tiktoken.Encoding): Tokenizer used for encoding and decoding text.
    """

    def __init__(self, model_name="gpt2-medium (355M)", models_dir="gpt2", model_size="355M"):
        """
        Initializes the GPTHandler with model configurations and loads the GPT model with pre-trained weights.

        Args:
            model_name (str): The name of the GPT model variant to load.
            models_dir (str): Directory containing GPT model files.
            model_size (str): The size of the GPT model to load.
        """
        # Download and load the pre-trained GPT model parameters
        self.settings, self.params = download_and_load_gpt2(model_size=model_size, models_dir=models_dir)

        # Define the base GPT configuration
        self.GPT_CONFIG_124M = {
            "vocab_size": 50257,
            "context_length": 256,
            "emb_dim": 768,
            "n_heads": 12,
            "n_layers": 12,
            "drop_rate": 0.1,
            "qkv_bias": False
        }

        # Define specific model configurations
        model_configs = {
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }

        # Update the base configuration with the selected model's settings
        self.config = self.GPT_CONFIG_124M.copy()
        self.config.update(model_configs[model_name])
        self.config.update({"context_length": 1024, "qkv_bias": True})

        # Initialize the GPT model
        self.model = GPTModel(self.config)
        self.model.eval()

        # Load the model weights
        self.load_weights_into_gpt(self.params)

        # Set the device (CUDA if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Initialize the tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def assign(self, left, right):
        """
        Assigns the given weights to the model parameters, ensuring that their shapes match.

        Args:
            left (torch.nn.Parameter): The model parameter to update.
            right (numpy.ndarray): The weights to assign to the model parameter.

        Returns:
            torch.nn.Parameter: The updated model parameter with the assigned weights.
        """
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(torch.tensor(right))

    def load_weights_into_gpt(self, params):
        """
        Loads pre-trained weights into the GPT model.

        Args:
            params (dict): The pre-trained model parameters.
        """
        self.model.pos_emb.weight = self.assign(self.model.pos_emb.weight, params['wpe'])
        self.model.tok_emb.weight = self.assign(self.model.tok_emb.weight, params['wte'])
        
        for b in range(len(params["blocks"])):
            q_w, k_w, v_w = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
            self.model.trf_blocks[b].att.W_query.weight = self.assign(
                self.model.trf_blocks[b].att.W_query.weight, q_w.T)
            self.model.trf_blocks[b].att.W_key.weight = self.assign(
                self.model.trf_blocks[b].att.W_key.weight, k_w.T)
            self.model.trf_blocks[b].att.W_value.weight = self.assign(
                self.model.trf_blocks[b].att.W_value.weight, v_w.T)

            q_b, k_b, v_b = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
            self.model.trf_blocks[b].att.W_query.bias = self.assign(
                self.model.trf_blocks[b].att.W_query.bias, q_b)
            self.model.trf_blocks[b].att.W_key.bias = self.assign(
                self.model.trf_blocks[b].att.W_key.bias, k_b)
            self.model.trf_blocks[b].att.W_value.bias = self.assign(
                self.model.trf_blocks[b].att.W_value.bias, v_b)

            self.model.trf_blocks[b].att.out_proj.weight = self.assign(
                self.model.trf_blocks[b].att.out_proj.weight, 
                params["blocks"][b]["attn"]["c_proj"]["w"].T)
            self.model.trf_blocks[b].att.out_proj.bias = self.assign(
                self.model.trf_blocks[b].att.out_proj.bias, 
                params["blocks"][b]["attn"]["c_proj"]["b"])

            self.model.trf_blocks[b].ff.layers[0].weight = self.assign(
                self.model.trf_blocks[b].ff.layers[0].weight, 
                params["blocks"][b]["mlp"]["c_fc"]["w"].T)
            self.model.trf_blocks[b].ff.layers[0].bias = self.assign(
                self.model.trf_blocks[b].ff.layers[0].bias, 
                params["blocks"][b]["mlp"]["c_fc"]["b"])
            self.model.trf_blocks[b].ff.layers[2].weight = self.assign(
                self.model.trf_blocks[b].ff.layers[2].weight, 
                params["blocks"][b]["mlp"]["c_proj"]["w"].T)
            self.model.trf_blocks[b].ff.layers[2].bias = self.assign(
                self.model.trf_blocks[b].ff.layers[2].bias, 
                params["blocks"][b]["mlp"]["c_proj"]["b"])

            self.model.trf_blocks[b].norm1.scale = self.assign(
                self.model.trf_blocks[b].norm1.scale, 
                params["blocks"][b]["ln_1"]["g"])
            self.model.trf_blocks[b].norm1.shift = self.assign(
                self.model.trf_blocks[b].norm1.shift, 
                params["blocks"][b]["ln_1"]["b"])
            self.model.trf_blocks[b].norm2.scale = self.assign(
                self.model.trf_blocks[b].norm2.scale, 
                params["blocks"][b]["ln_2"]["g"])
            self.model.trf_blocks[b].norm2.shift = self.assign(
                self.model.trf_blocks[b].norm2.shift, 
                params["blocks"][b]["ln_2"]["b"])

        self.model.final_norm.scale = self.assign(self.model.final_norm.scale, params["g"])
        self.model.final_norm.shift = self.assign(self.model.final_norm.shift, params["b"])
        self.model.out_head.weight = self.assign(self.model.out_head.weight, params["wte"])

    def generate_text(self, start_context, max_new_tokens=10):
        """
        Generates text using the GPT model starting from the provided context.

        Args:
            start_context (str): The initial context text for the model to generate text from.
            max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
            str: The generated text.
        """
        token_ids = generate_text_simple(
            model=self.model,
            idx=text_to_token_ids(start_context, self.tokenizer).to(self.device),
            max_new_tokens=max_new_tokens,
            context_size=self.config["context_length"]
        )
        return token_ids_to_text(token_ids, self.tokenizer)


if __name__ == "__main__":
    # Initialize the GPT handler
    gpt_handler = GPTHandler()

    # Generate text based on the provided start context
    start_context = "Every effort moves you"
    generated_text = gpt_handler.generate_text(start_context)

    # Print the generated text
    print("Output text:\n", generated_text)
