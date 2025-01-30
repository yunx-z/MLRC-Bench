import torch 

from methods.BaseMethod import BaseMethod
from peft import get_peft_model, set_peft_model_state_dict

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)

    # Implement merge function 
    def run(self):

        '''
        1) Load HuggingFace checkpoints and configs 
        '''
        super()._load_huggingface_models_and_configs()
        '''
        2) Merge checkpoints  
        '''
        parameter_lambdas = [1 / len(self.list_models) for _ in self.list_models]

        # Get individual models 
        all_models = list(self.loaded_models.values())

        # Get all the parameters names (uses the first model and assume all the models have the same parameter)
        all_parameter_names = all_models[0].keys()

        for parameter_name in all_parameter_names:
            merged_parameter = None
            for parameter_lambda, model in zip(parameter_lambdas, all_models):
                parameter = model[parameter_name]
                if merged_parameter is None:
                    merged_parameter = torch.clone(parameter) * parameter_lambda
                else:
                    merged_parameter += parameter * parameter_lambda
            self.merged_model[parameter_name] = merged_parameter

        '''
        3) Load base model and tokenizer
        '''
        self._load_base_model()
        self._load_tokenizer()

        '''
        4) Load merged model into base model 
        '''
        self.base_model.load_state_dict(self.merged_model)
        # Requires to make results deterministic. If not set, we will just run once and use the results from the first pass. 
        self.base_model.eval()

        return self.base_model
