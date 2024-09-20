from litgpt import LLM

class LLMHandler:
    """
    A class to handle operations related to the LitGPT language model, such as loading the model and generating text.

    Attributes:
        model_name (str): The name of the language model to load.
        model (LLM): The loaded language model.
    """

    def __init__(self, model_name="microsoft/phi-2"):
        """
        Initializes the LLMHandler by loading the specified language model.

        Args:
            model_name (str): The name of the language model to load. Defaults to "microsoft/phi-2".
        """
        self.model_name = model_name
        self.model = LLM.load(self.model_name)

    def generate_text(self, prompt, stream=False, max_new_tokens=200):
        """
        Generates text based on the given prompt.

        Args:
            prompt (str): The input text prompt for the language model.
            stream (bool): Whether to stream the output text. Defaults to False.
            max_new_tokens (int): The maximum number of new tokens to generate. Defaults to 200.

        Returns:
            str or generator: If streaming is disabled, returns the generated text as a string.
                              If streaming is enabled, returns a generator that yields the text tokens.
        """
        if stream:
            return self.model.generate(prompt, stream=True, max_new_tokens=max_new_tokens)
        else:
            return self.model.generate(prompt)

    def print_streamed_output(self, prompt, max_new_tokens=200):
        """
        Generates text based on the given prompt and prints the output in a streaming fashion.

        Args:
            prompt (str): The input text prompt for the language model.
            max_new_tokens (int): The maximum number of new tokens to generate. Defaults to 200.
        """
        result = self.generate_text(prompt, stream=True, max_new_tokens=max_new_tokens)
        for token in result:
            print(token, end="", flush=True)


if __name__ == "__main__":
    # Initialize the LLMHandler with the default model
    llm_handler = LLMHandler()

    # Generate and print a single response
    prompt = "What do Llamas eat?"
    generated_text = llm_handler.generate_text(prompt)
    print(generated_text)

    # Generate and print streamed output
    llm_handler.print_streamed_output(prompt)
