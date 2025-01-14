import json
from litgpt import LLM
from tqdm import tqdm

class DataHandler:
    """
    Handles loading, splitting, and saving of dataset for fine-tuning.
    
    Attributes:
        file_path (str): The path to the dataset file.
        train_data (list): List of training data entries.
        test_data (list): List of testing data entries.
    """

    def __init__(self, file_path):
        """
        Initializes the DataHandler by loading the data from a JSON file and splitting it into training and testing sets.

        Args:
            file_path (str): The path to the dataset file.
        """
        self.file_path = file_path
        self.data = self.load_data()
        self.train_data, self.test_data = self.split_data()

    def load_data(self):
        """
        Loads data from a JSON file.

        Returns:
            list: The list of data entries.
        """
        with open(self.file_path, "r") as file:
            data = json.load(file)
        print("Number of entries:", len(data))
        return data

    def split_data(self, train_ratio=0.85):
        """
        Splits the data into training and testing sets based on the given ratio.

        Args:
            train_ratio (float): The ratio of data to be used for training. Defaults to 0.85 (85%).

        Returns:
            tuple: A tuple containing the training data and testing data as lists.
        """
        train_portion = int(len(self.data) * train_ratio)
        train_data = self.data[:train_portion]
        test_data = self.data[train_portion:]
        return train_data, test_data

    def save_data(self, train_file="train.json", test_file="test.json"):
        """
        Saves the training and testing data to JSON files.

        Args:
            train_file (str): The filename for the training data. Defaults to "train.json".
            test_file (str): The filename for the testing data. Defaults to "test.json".
        """
        with open(train_file, "w") as json_file:
            json.dump(self.train_data, json_file, indent=4)
        
        with open(test_file, "w") as json_file:
            json.dump(self.test_data, json_file, indent=4)


class FineTuningHandler:
    """
    Handles the fine-tuning of the language model using the LLM library.
    """

    def __init__(self, base_model="microsoft/phi-2", finetuned_model_path=None):
        """
        Initializes the FineTuningHandler with the base and finetuned models.

        Args:
            base_model (str): The path to the base model. Defaults to "microsoft/phi-2".
            finetuned_model_path (str): The path to the finetuned model. If provided, loads the finetuned model.
        """
        self.base_model = LLM.load(base_model)
        self.finetuned_model = LLM.load(finetuned_model_path) if finetuned_model_path else None

    def generate_response(self, model, entry):
        """
        Generates a response from the specified model for the given entry.

        Args:
            model (LLM): The language model to generate the response.
            entry (dict): The input data entry.

        Returns:
            str: The generated response.
        """
        return model.generate(entry)

    def format_input(self, entry):
        """
        Formats an input entry into an instruction-compliant format for the model.

        Args:
            entry (dict): The input data entry.

        Returns:
            str: The formatted instruction text.
        """
        instruction_text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}"
        )

        input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

        return instruction_text + input_text

    def run_inference(self, model, data, key_name):
        """
        Runs inference on the dataset using the specified model and updates the dataset with generated responses.

        Args:
            model (LLM): The language model to generate the responses.
            data (list): The dataset to run inference on.
            key_name (str): The key to store the generated responses in the dataset.
        """
        for i in tqdm(range(len(data))):
            response = self.generate_response(model, data[i])
            data[i][key_name] = response


if __name__ == "__main__":
    # Initialize DataHandler and split the dataset
    data_handler = DataHandler("LLM-workshop-2024/06_finetuning/instruction-data.json")
    data_handler.save_data()

    # Fine-tune the base model
    fine_tuner = FineTuningHandler()
    fine_tuner.run_inference(fine_tuner.base_model, data_handler.test_data, key_name="base_model")

    # Load the finetuned model and run inference
    finetuner_with_model = FineTuningHandler(finetuned_model_path="/teamspace/studios/this_studio/out/finetune/lora/final/")
    finetuner_with_model.run_inference(finetuner_with_model.finetuned_model, data_handler.test_data, key_name="finetuned_model")
