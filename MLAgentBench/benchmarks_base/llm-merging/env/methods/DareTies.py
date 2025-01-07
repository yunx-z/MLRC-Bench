import os
import tempfile
import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge
import shutil

from methods.BaseMethod import BaseMethod
from safetensors.torch import save_file
from tqdm import tqdm


class DareTies(BaseMethod):
    def __init__(self, name):
        super().__init__(name)
        # Define the models with their respective density and weight
        self.models_with_params = [
            {
                "model": "meta-llama/Meta-Llama-3-8B-Instruct",
                "parameters": {
                    "density": 0.6,
                    "weight": 0.5
                }
            },
            {
                "model": "MaziyarPanahi/Llama-3-8B-Instruct-v0.8",
                "parameters": {
                    "density": 0.55,
                    "weight": 0.5
                }
            }
        ]

    def get_model_config(self):
        """
        Prepare the merge configuration dictionary.
        """
        config = {
            "models": [
                {
                    "model": model["model"],
                    "parameters": {
                        "density": model["parameters"]["density"],
                        "weight": model["parameters"]["weight"]
                    }
                }
                for model in self.models_with_params
            ],
            "merge_method": "dare_ties",
            "base_model": self.base_model_name,
            "parameters": {
                "normalize": True,
                "int8_mask": True
            },
            "dtype": "bfloat16"
        }
        return config

    def run_mergekit(self, temp_output_path):
        """
        Execute the DARE-TIES merging process using mergekit.
        """
        # Step 1: Prepare the merge configuration
        merge_config_dict = self.get_model_config()

        # Write the config to a temporary YAML file
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.yml') as temp_config_file:
            yaml.dump(merge_config_dict, temp_config_file)
            config_path = temp_config_file.name

        try:
            # Load the merge configuration using mergekit
            with open(config_path, "r", encoding="utf-8") as fp:
                merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))
            
            # Define merge options
            merge_options = MergeOptions(
                lora_merge_cache=temp_output_path,  # Adjust if necessary
                cuda=torch.cuda.is_available(),
                copy_tokenizer=True,
                lazy_unpickle=False,
                low_cpu_memory=False
            )
            
            # Step 2: Execute the merge
            run_merge(
                merge_config,
                out_path=temp_output_path,
                options=merge_options
            )
            print("DARE-TIES Merge Completed!")

        finally:
            # Clean up the temporary config file
            os.remove(config_path)

    def load_merged_model(self, temp_output_path):
        """
        Load the merged model from the temporary output directory using transformers.
        """
        if not os.path.exists(temp_output_path):
            raise FileNotFoundError(f"Merged model directory not found at {temp_output_path}")

        # Load the merged model
        merged_model = AutoModelForCausalLM.from_pretrained(
            temp_output_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        merged_model.to(self.device)
        merged_model.eval()
        print(f"Merged model loaded successfully from {temp_output_path}")

        return merged_model

    def run(self):
        """
        Execute the DARE-TIES merging process using mergekit and load the merged model into memory.
        """
        # Create a temporary directory for mergekit's output
        os.makedirs("./tmp", exist_ok=True)
        temp_dir = tempfile.mkdtemp(dir="./tmp")
        try:
            # Step 1: Perform the merge into the temporary directory
            self.run_mergekit(temp_output_path=temp_dir)

            # Step 2: Load the merged model from the temporary directory
            self.base_model = self.load_merged_model(temp_output_path=temp_dir)
            self._load_tokenizer()

            # Step 3: Extract the merged state dict into memory
            self.merged_model = self.base_model.state_dict()

        finally:
            # Step 4: Delete the temporary directory to clean disk storage
            shutil.rmtree(temp_dir)

        return self.base_model

    def save_model(self, output_dir):
        """
        Save the merged model to the specified directory.
        """
        assert self.merged_model is not None, "Merged model is empty"

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save the merged model's state dict as safetensors
        save_file(self.merged_model, os.path.join(output_dir, "safetensors.pt"))
        
        # Save tokenizer and config files
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        if self.base_model is not None:
            self.base_model.config.to_json_file(os.path.join(output_dir, "config.json"))

        print(f"Merged model saved to {output_dir}")

