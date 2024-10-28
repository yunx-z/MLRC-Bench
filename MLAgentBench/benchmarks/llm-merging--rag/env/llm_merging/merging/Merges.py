import copy
import os

from peft import load_peft_weights, PeftConfig
from safetensors.torch import save_file

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
)

import llm_merging.model.decoder_functions as decoder_functions
import llm_merging.model.encoder_decoder_functions as encoder_decoder_functions
from llm_merging.constants import * 


class Merges(object):

    def __init__(self, name):
        self.name = name

        # Give a list of models to load for the merge. Each element is the list a is a tuple of (model, revision_id). We recommend specifying a revision id to ensure the model was not modified after May 31 
        """
        # this is an example
        self.list_models = [("predibase/magicoder", "58d0eedad92a223cd45e94534450066952c6de25"),
                            ("predibase/conllpp", "b9d370f421cb61389e345763ec629e50a58c2676"),
                            ("predibase/cnn", "dec34c493cc448bb8a8e322adf874198af3399b5"),
                            ("predibase/agnews_explained", "8efbbefadd6a24b8df7a2697a13a321054b4f8f1"),
                            ("predibase/gsm8k", "d54e59aa31095b6670d0b2717e0082e76660a0c1"),
                            # ...
                           ]
        """
        self.list_models = MODEL_LIST

        # Hyperparameters 
        self.base_model_name = BASE_MODEL
        # We recommend specifying a revision id to ensure the model was not modified after May 31 
        self.base_model_revision_id = BASE_MODEL_REVISION_ID 
         
        self.base_model = None
        self.tokenizer = None
        self.input_tokenizer = None
        self.target_tokenizer = None

        self.is_peft = True

        self.max_seq_len = None
        self.max_gen_len = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Architecture must match base model. 
        self.architecture = "decoder"

        self.loaded_models = {}
        self.loaded_configs = {}

        # Merged model parameters
        self.merged_model = {}


    def get_name(self):
        return self.name

    def get_model_config(self):
        raise NotImplementedError

    def _load_base_model(self):
        if self.architecture == "encoder_decoder":
            self.base_model =  AutoModelForSeq2SeqLM.from_pretrained(self.base_model_name, revision=self.base_model_revision_id, token=os.environ["HF_AUTH_TOKEN"]).to(self.device)
        elif self.architecture == "decoder":
            self.base_model =  AutoModelForCausalLM.from_pretrained(self.base_model_name, revision=self.base_model_revision_id, token=os.environ["HF_AUTH_TOKEN"]).to(self.device)
        else:
            raise NotImplementedError(f"Architecture not implemented {self.architecture}")
        

    def _load_tokenizer(self):

        if self.architecture == "encoder_decoder":
            if self.tokenizer is None:

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model_name,
                    revision=self.base_model_revision_id,
                    model_max_length=self.max_seq_len,
                    legacy=False,
                    token=os.environ["HF_AUTH_TOKEN"]
                )

        elif self.architecture == "decoder":
            if self.input_tokenizer is None or self.target_tokenizer is None:
                    
                self.input_tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model_name,
                    revision=self.base_model_revision_id,
                    model_max_length=self.max_seq_len,
                    legacy=False,
                    token=os.environ["HF_AUTH_TOKEN"]
                )
                self.target_tokenizer = copy.deepcopy(self.input_tokenizer)

                # Use eos_token for pad_token if it doesn't exist. This is ok since the
                # pad tokens will be ignored through the mask
                if self.input_tokenizer.pad_token_id is None:
                    self.input_tokenizer.pad_token_id = self.input_tokenizer.eos_token_id
                if self.target_tokenizer.pad_token_id is None:
                    self.target_tokenizer.pad_token_id = self.target_tokenizer.eos_token_id

                # Add BOS and not EOS token 
                self.input_tokenizer.padding_side = "left"

                # Add EOS and not BOS token 
                self.target_tokenizer.padding_side = "right"
                self.target_tokenizer.add_bos_token = False
                self.target_tokenizer.add_eos_token = True
        else:
            raise NotImplementedError(f"Architecture not implemented {self.architecture}")

    def predict_multiple_choice(self, batch):
        assert self.base_model is not None
        if self.architecture == "encoder_decoder":
            assert self.tokenizer is not None
            return encoder_decoder_functions.predict_multiple_choice(self.base_model, self.tokenizer, batch)
        elif self.architecture == "decoder":
            return decoder_functions.predict_multiple_choice(self.base_model, self.input_tokenizer, self.target_tokenizer, batch)
        else:
            raise NotImplementedError(f"Architecture not implemented {self.architecture}")
    
    def generate(self, batch):
        assert self.base_model is not None
        if self.architecture == "encoder_decoder":
            assert self.tokenizer is not None
            return encoder_decoder_functions.generate(self.base_model, self.tokenizer, batch, self.max_gen_len)
        elif self.architecture == "decoder":
            return decoder_functions.generate(self.base_model, self.input_tokenizer, self.target_tokenizer, batch, self.max_gen_len)
        else:
            raise NotImplementedError(f"Architecture not implemented {self.architecture}")

    def _load_huggingface_models_and_configs(self):
        assert len(self.list_models) > 0, f"List of models must include at leat 1 model"

        parameter_names = None
        for model_name, revision_id in self.list_models:
            if self.is_peft:
                peft_model_parameters = load_peft_weights(model_name, revision=revision_id, token=os.environ["HF_AUTH_TOKEN"])
                peft_config = PeftConfig.from_pretrained(model_name)

                if parameter_names is None:
                    parameter_names = set(peft_model_parameters.keys())

                if parameter_names != set(peft_model_parameters.keys()):
                    print(f"WARNING: parameters in {model_name} do not match {self.list_models[0]}")

                self.loaded_models[model_name] = peft_model_parameters 
                self.loaded_configs[model_name] = peft_config

            else:
                if self.architecture == "encoder_decoder":
                    model_parameters = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=revision_id, token=os.environ["HF_AUTH_TOKEN"]).to(self.device).state_dict()
                elif self.architecture == "decoder":
                    model_parameters = AutoModelForCausalLM.from_pretrained(model_name, revision=revision_id, token=os.environ["HF_AUTH_TOKEN"]).to(self.device).state_dict()
                else:
                    raise NotImplementedError(f"Architecture not implemented {self.architecture}")

                if parameter_names is None:
                    parameter_names = set(model_parameters.keys())

                if parameter_names != set(model_parameters.keys()):
                    print(f"WARNING: parameters in {model_name} do not match {self.list_models[0]}")

                self.loaded_models[model_name] = model_parameters 
                self.loaded_configs[model_name] = AutoConfig.from_pretrained(model_name)

    def merge(
        self,
    ):
        raise NotImplementedError
    
    def save_model(self, output_dir):
        assert self.merged_model is not None, "Merged model is empty"
        # assert len(self.merged_model) > 0, "Merged model is empty"
        # Save merged model as safetensor 
        save_file(self.merged_model, os.path.join(output_dir, "safetensor.pt"))
