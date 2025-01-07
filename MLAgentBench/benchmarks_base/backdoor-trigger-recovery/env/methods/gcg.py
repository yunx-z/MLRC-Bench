import gc
import numpy as np
import torch
import torch.nn as nn
import json
import sys
from baselines.llm_attacks.minimal_gcg.opt_utils import token_gradients, sample_control, get_logits, target_loss
from baselines.llm_attacks.minimal_gcg.opt_utils import load_model_and_tokenizer, get_filtered_cands
from baselines.llm_attacks.minimal_gcg.string_utils import SuffixManager, load_conversation_template
from baselines.llm_attacks import get_nonascii_toks
from transformers import AutoModelForCausalLM, AutoTokenizer

class GCG():
    def __init__(self,
        model_path):
        np.random.seed(20)
        torch.manual_seed(20)
        torch.cuda.manual_seed_all(20)
        self.model_path = model_path
        template_name = 'qwen'
        self.num_steps = 50
        self.adv_string_init = "! ! ! ! ! ! ! ! ! !"
        self.device = 'cuda'
        self.batch_size = 64
        self.topk = 256
        self.allow_non_ascii = False
        self.conv_template = load_conversation_template(template_name)
        self.conv_template.system_message = "You are a helpful assistant."
        self.model, self.tokenizer = self.load_model()
        self.model = self.model.to(self.device).eval()

    def update_task(self, user_prompt, target_output):
        self.instruction = user_prompt
        self.target = target_output
        self.suffix_manager = SuffixManager(tokenizer=self.tokenizer,
                                            conv_template=self.conv_template,
                                            instruction=self.instruction,
                                            target=self.target,
                                            adv_string=self.adv_string_init)

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            use_cache=False,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side ="right"
        return model, tokenizer

    def check_for_attack_success(self, model, tokenizer, input_ids, assistant_role_slice, target, gen_config=None):
        gen_str = tokenizer.decode(self.generate(model,
                                                 tokenizer,
                                                 input_ids,
                                                 assistant_role_slice,
                                                 gen_config=gen_config)).strip()
        jailbroken = target in gen_str
        return jailbroken

    def generate(self, model, tokenizer, input_ids, assistant_role_slice, gen_config=None):
        if gen_config is None:
            gen_config = model.generation_config
            gen_config.max_new_tokens = 32
        input_ids = input_ids[:assistant_role_slice.stop].to(model.device).unsqueeze(0)
        attn_masks = torch.ones_like(input_ids).to(model.device)
        output_ids = model.generate(input_ids,
                                    attention_mask=attn_masks,
                                    generation_config=gen_config,
                                    pad_token_id=tokenizer.pad_token_id)[0]
        return output_ids[assistant_role_slice.stop:]

    def gcg_attack(self):
        adv_suffix = self.adv_string_init
        for i in range(self.num_steps):
            input_ids = self.suffix_manager.get_input_ids(adv_string=adv_suffix).to(self.model.device)
            coordinate_grad = token_gradients(self.model,
                                              input_ids,
                                              self.suffix_manager._control_slice,
                                              self.suffix_manager._target_slice,
                                              self.suffix_manager._loss_slice)
            with torch.no_grad():
                adv_suffix_tokens = input_ids[self.suffix_manager._control_slice].to(self.device)
                new_adv_suffix_toks = sample_control(adv_suffix_tokens,
                                                     coordinate_grad,
                                                     self.batch_size,
                                                     topk=self.topk,
                                                     temp=1,
                                                     not_allowed_tokens=None if self.allow_non_ascii else get_nonascii_toks(self.tokenizer))
                new_adv_suffix = get_filtered_cands(self.tokenizer,
                                                    new_adv_suffix_toks,
                                                    filter_cand=True,
                                                    curr_control=adv_suffix)
                logits, ids = get_logits(model=self.model,
                                         tokenizer=self.tokenizer,
                                         input_ids=input_ids,
                                         control_slice=self.suffix_manager._control_slice,
                                         test_controls=new_adv_suffix,
                                         return_ids=True,
                                         batch_size=32)
                losses = target_loss(logits, ids, self.suffix_manager._target_slice)
                best_new_adv_suffix_id = losses.argmin()
                best_new_adv_suffix = new_adv_suffix[best_new_adv_suffix_id]
                current_loss = losses[best_new_adv_suffix_id]

                adv_suffix = best_new_adv_suffix
                is_success = self.check_for_attack_success(self.model,
                                                           self.tokenizer,
                                                           self.suffix_manager.get_input_ids(adv_string=adv_suffix).to(self.device),
                                                           self.suffix_manager._assistant_role_slice,
                                                           self.target)

            print(f"\nPassed:{is_success}\nCurrent Suffix:{best_new_adv_suffix}", end='\r')
            if is_success:
                self.adv_suffix = best_new_adv_suffix
                break
            del coordinate_grad, adv_suffix_tokens
            gc.collect()
            torch.cuda.empty_cache()

    def test(self):
        input_ids = self.suffix_manager.get_input_ids(adv_string=self.adv_suffix).to(self.device)
        gen_config = self.model.generation_config
        gen_config.max_new_tokens = 64
        completion = self.tokenizer.decode((self.generate(self.model, self.tokenizer, input_ids, self.suffix_manager._assistant_role_slice, gen_config=gen_config))).strip()
        print(f"\nCompletion: {completion}")

