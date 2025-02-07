import common
from language_models import GPT, PaLM, HuggingFace, APIModelLlama7B, APIModelVicuna13B, GeminiPro, OpenRouter, ReplicateModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import VICUNA_PATH, LLAMA_PATH, ATTACK_TEMP, TARGET_TEMP, ATTACK_TOP_P, TARGET_TOP_P, MAX_PARALLEL_STREAMS 

def load_target_model(args):
    target_llm = TargetLLM(model_name = args.target_model, 
                        max_n_tokens = args.target_max_n_tokens,
                        temperature = TARGET_TEMP, # init to 0
                        top_p = TARGET_TOP_P, # init to 1
                        )
    return target_llm

def load_attack_and_target_models(args):
    # Load attack model and tokenizer
    attack_llm = AttackLLM(model_name = args.attack_model, 
                        max_n_tokens = args.attack_max_n_tokens, 
                        max_n_attack_attempts = args.max_n_attack_attempts, 
                        temperature = ATTACK_TEMP, # init to 1
                        top_p = ATTACK_TOP_P, # init to 0.9
                        )
    
    # Only share model if both are local models
    preloaded_model = None
    if (args.attack_model == args.target_model and 
        not args.attack_model.startswith(("openrouter/", "replicate/", "gpt-", "palm-2", "gemini-pro")) and
        args.attack_model not in ["llama-2-api-model", "vicuna-api-model"]):
        print("Using same attack and target model. Using previously loaded model.")
        preloaded_model = attack_llm.model
        
    target_llm = TargetLLM(model_name = args.target_model, 
                        max_n_tokens = args.target_max_n_tokens,
                        temperature = TARGET_TEMP, # init to 0
                        top_p = TARGET_TOP_P, # init to 1
                        preloaded_model = preloaded_model,
                        )
    return attack_llm, target_llm

class AttackLLM():
    """
    Base class for attacker language models.
    Generates attacks for conversations using a language model. 
    The self.model attribute contains the underlying generation model.
    """
    def __init__(self, 
                model_name: str, 
                max_n_tokens: int, 
                max_n_attack_attempts: int, 
                temperature: float,
                top_p: float):
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.top_p = top_p
        
        try:
            # Get model and template without loading if it's an API model
            if model_name.startswith(("openrouter/", "replicate/", "gpt-", "palm-2", "gemini-pro")) or \
               model_name in ["llama-2-api-model", "vicuna-api-model"]:
                self.model, self.template = load_indiv_model(model_name)
            else:
                # Only load local models
                self.model, self.template = load_indiv_model(model_name)
                if "vicuna" in model_name or "llama" in model_name:
                    if "api-model" not in model_name:
                        self.model.extend_eos_tokens()
        except Exception as e:
            raise ValueError(f"Failed to initialize model {model_name}: {str(e)}")

    def get_attack(self, convs_list, prompts_list):
        """
            Generates responses for a batch of conversations and prompts using a language model. 
            Only valid outputs in proper JSON format are returned. If an output isn't generated 
            successfully after max_n_attack_attempts, it's returned as None.
            
            Parameters:
            - convs_list: List of conversation objects.
            - prompts_list: List of prompts corresponding to each conversation.
            
            Returns:
            - List of generated outputs (dictionaries) or None for failed generations.
        """
        
        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."
        
        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize

        # Initialize the attack model's generated output to match format
        if len(convs_list[0].messages) == 0:
            init_message = """{\"improvement\": \"\",\"prompt\": \""""
        else:
            init_message = """{\"improvement\": \"""" 

        full_prompts = []
        # Add prompts and initial seeding messages to conversations (only once)
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            # Get prompts
            if "gpt" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            else:
                conv.append_message(conv.roles[1], init_message)
                full_prompts.append(conv.get_prompt()[:-len(conv.sep2)])
            
        for _ in range(self.max_n_attack_attempts):
            # Subset conversations based on indices to regenerate
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]

            # Generate outputs 
            outputs_list = []
            for left in range(0, len(full_prompts_subset), MAX_PARALLEL_STREAMS):
                right = min(left + MAX_PARALLEL_STREAMS, len(full_prompts_subset)) 
                
                if right == left:
                    continue 

                print(f'\tQuerying attacker with {len(full_prompts_subset[left:right])} prompts', flush=True)
                
                outputs_list.extend(
                                    self.model.batched_generate(full_prompts_subset[left:right],
                                                        max_n_tokens = self.max_n_tokens,  
                                                        temperature = self.temperature,
                                                        top_p = self.top_p
                                                    )
                )
            
            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                
                if "gpt" not in self.model_name:
                    full_output = init_message + full_output

                attack_dict, json_str = common.extract_json(full_output)
                
                if attack_dict is not None:
                    valid_outputs[orig_index] = attack_dict
                    # Update the conversation with valid generation
                    convs_list[orig_index].update_last_message(json_str) 
                        
                else:
                    new_indices_to_regenerate.append(orig_index)
            
            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate 
            
            # If all outputs are valid, break
            if not indices_to_regenerate:
                break
        
        if any([output for output in valid_outputs if output is None]):
            print(f"Failed to generate output after {self.max_n_attack_attempts} attempts. Terminating.")
        return valid_outputs

class TargetLLM():
    """
        Base class for target language models.
        Generates responses for prompts using a language model. 
        The self.model attribute contains the underlying generation model.
    """
    def __init__(self, 
            model_name: str, 
            max_n_tokens: int, 
            temperature: float,
            top_p: float,
            preloaded_model: object = None):
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.top_p = top_p
        
        try:
            # If we have a preloaded model and it's not an API model, use it
            if (preloaded_model is not None and 
                not model_name.startswith(("openrouter/", "replicate/", "gpt-", "palm-2", "gemini-pro")) and
                model_name not in ["llama-2-api-model", "vicuna-api-model"]):
                self.model = preloaded_model
                _, self.template = get_model_path_and_template(model_name)
            else:
                # Otherwise load new model (API or local)
                self.model, self.template = load_indiv_model(model_name)
        except Exception as e:
            raise ValueError(f"Failed to initialize model {model_name}: {str(e)}")

    def get_response(self, prompts_list):
        batchsize = len(prompts_list)
        convs_list = [common.conv_template(self.template) for _ in range(batchsize)]
        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            if "openrouter/" in self.model_name:
                full_prompts.append([{"role": "user", "content": prompt}])
            elif "gpt" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            elif "palm" in self.model_name:
                full_prompts.append(conv.messages[-1][1])
            else:
                conv.append_message(conv.roles[1], None) 
                full_prompts.append(conv.get_prompt())

        # Query the attack LLM in batched-queries with at most MAX_PARALLEL_STREAMS-many queries at a time
        outputs_list = []
        for left in range(0, len(full_prompts), MAX_PARALLEL_STREAMS):
            right = min(left + MAX_PARALLEL_STREAMS, len(full_prompts))
            
            if right == left:
                continue 

            print(f'\tQuerying target with {len(full_prompts[left:right])} prompts', flush=True)


            outputs_list.extend(
                                self.model.batched_generate(full_prompts[left:right], 
                                                            max_n_tokens = self.max_n_tokens,  
                                                            temperature = self.temperature,
                                                            top_p = self.top_p
                                                        )
            )
        return outputs_list



def load_indiv_model(model_name):
    try:
        model_path, template = get_model_path_and_template(model_name)
    except Exception as e:
        raise ValueError(f"Failed to get model path and template for {model_name}: {str(e)}")
    
    common.MODEL_NAME = model_name
    
    # Handle API-based models first
    if model_name.startswith("openrouter/"):
        lm = OpenRouter(model_name)
    elif model_name.startswith("replicate/"):
        lm = ReplicateModel(model_name)
    elif model_name in ["gpt-3.5-turbo", "gpt-4", 'gpt-4-1106-preview']:
        lm = GPT(model_name)
    elif model_name == "palm-2":
        lm = PaLM(model_name)
    elif model_name == "gemini-pro":
        lm = GeminiPro(model_name)
    elif model_name == 'llama-2-api-model':
        lm = APIModelLlama7B(model_name)
    elif model_name == 'vicuna-api-model':
        lm = APIModelVicuna13B(model_name)
    # Only load from HuggingFace for local models
    elif model_path is not None:  # Check if it's a local model path
        try:
            model = AutoModelForCausalLM.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto"
            ).eval()

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=True,
                model_max_length=2048
            ) 

            if 'llama-2' in model_path.lower():
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.padding_side = 'left'
            if 'vicuna' in model_path.lower():
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = 'left'
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token

            lm = HuggingFace(model_name, model, tokenizer)
        except Exception as e:
            raise ValueError(f"Failed to load local model {model_name}: {str(e)}")
    else:
        raise ValueError(f"Unknown model type or missing model path: {model_name}")
    
    return lm, template

def get_model_path_and_template(model_name):
    # Base model dictionary for non-OpenRouter models
    full_model_dict={
        "gpt-4-1106-preview":{
            "path":"gpt-4-1106-preview",
            "template":"gpt-4-1106-preview"
        },
        "gpt-4-turbo":{
            "path":"gpt-4-1106-preview",
            "template":"gpt-4-1106-preview"
        },
        "gpt-4":{
            "path":"gpt-4",
            "template":"gpt-4"
        },
        "gpt-3.5-turbo": {
            "path": "gpt-3.5-turbo",
            "template":"gpt-3.5-turbo"
        },
        "vicuna":{
            "path": VICUNA_PATH,
            "template":"vicuna_v1.1"
        },
        "vicuna-api-model":{
            "path": None,
            "template": "vicuna_v1.1"
        },
        "llama-2":{
            "path": LLAMA_PATH,
            "template":"llama-2"
        },
        "llama-2-api-model":{
            "path": None,
            "template": "llama-2-7b"
        },
        "palm-2":{
            "path":"palm-2",
            "template":"palm-2"
        },
        "gemini-pro": {
            "path": "gemini-pro",
            "template": "gemini-pro"
        },
        "openrouter/anthropic/claude-2":{
            "path": "openrouter/anthropic/claude-2",
            "template": "claude-2"
        },
        "openrouter/google/palm-2-chat-bison":{
            "path": "openrouter/google/palm-2-chat-bison",
            "template": "palm-2"
        },
        "openrouter/meta-llama/llama-2-70b-chat":{
            "path": "openrouter/meta-llama/llama-2-70b-chat", 
            "template": "llama-2"
        },
        "openrouter/meta-llama/llama-3.3-70b-instruct":{
            "path": "openrouter/meta-llama/llama-3.3-70b-instruct",
            "template": "llama-2"  # Using llama-2 template since it's a Llama model
        },
        "vicuna-replicate":{
            "path": None,
            "template": "vicuna_v1.1"
        },
        # Add Replicate models
        "replicate/vicuna-13b":{
            "path": None,
            "template": "vicuna_v1.1"
        },
    }
    
    # Handle OpenRouter and Replicate models dynamically
    if model_name.startswith("openrouter/") or model_name.startswith("replicate/"):
        # Extract the provider and model name
        provider = model_name.split('/')[0]  
        model_type = model_name.split('/')[1]  
        
        if provider == "replicate":
            if "vicuna" in model_type:
                template = "vicuna_v1.1"
            else:
                template = "gpt-4"
            return model_name, template  # Add this return
        elif provider == "openrouter":
            # Extract the base model name
            base_model = model_name.split('/')[-1]  # Fix: use last part for base_model
            provider = model_name.split('/')[1]  # e.g., "anthropic", "meta-llama"
            
            if provider == "anthropic":
                template = "claude-2"
            elif provider == "meta-llama":
                template = "llama-2"
            elif provider == "google":
                template = "palm-2"
            else:
                template = "gpt-4"
            
            return model_name, template
    
    # Handle other models
    if model_name in full_model_dict:
        path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
        return path, template
    else:
        raise ValueError(f"Unknown model: {model_name}")



    
