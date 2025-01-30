import openai
import anthropic
import os
import time
import torch
import gc
from typing import Dict, List
import google.generativeai as genai
import urllib3
from copy import deepcopy
import replicate

from config import LLAMA_API_LINK, VICUNA_API_LINK, MAX_PARALLEL_STREAMS

    
class LanguageModel():
    def __init__(self, model_name):
        self.model_name = model_name
    
    def batched_generate(self, prompts_list: List, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError
        
class HuggingFace(LanguageModel):
    def __init__(self,model_name, model, tokenizer):
        self.model_name = model_name
        self.model = model 
        self.tokenizer = tokenizer
        self.eos_token_ids = [self.tokenizer.eos_token_id]
        
        # Clear cache on initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def batched_generate(self, 
                        full_prompts_list,
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        # Process in batches of MAX_PARALLEL_STREAMS
        outputs_list = []
        
        for i in range(0, len(full_prompts_list), MAX_PARALLEL_STREAMS):
            batch = full_prompts_list[i:i + MAX_PARALLEL_STREAMS]
            
            # Clear cache before processing batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            try:
                with torch.no_grad():  # Prevent gradient computation
                    inputs = self.tokenizer(batch, return_tensors='pt', padding=True)
                    inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()} 
                    output_ids = None  # Initialize output_ids

                    try:
                        # Batch generation
                        if temperature > 0:
                            output_ids = self.model.generate(
                                **inputs,
                                max_new_tokens=max_n_tokens, 
                                do_sample=True,
                                temperature=temperature,
                                eos_token_id=self.eos_token_ids,
                                top_p=top_p,
                            )
                        else:
                            output_ids = self.model.generate(
                                **inputs,
                                max_new_tokens=max_n_tokens, 
                                do_sample=False,
                                eos_token_id=self.eos_token_ids,
                                top_p=1,
                                temperature=1, # To prevent warning messages
                            )
                        
                        # If the model is not an encoder-decoder type, slice off the input tokens
                        if not self.model.config.is_encoder_decoder:
                            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

                        # Batch decoding
                        batch_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                        outputs_list.extend(batch_outputs)

                    except torch.cuda.OutOfMemoryError:
                        print(f"OOM error with batch size {len(batch)}, trying with smaller batch...")
                        # Try processing one at a time
                        for single_prompt in batch:
                            single_outputs = self.batched_generate([single_prompt], max_n_tokens, temperature, top_p)
                            outputs_list.extend(single_outputs)
                    
                    finally:
                        # Clean up
                        for key in inputs:
                            inputs[key].to('cpu')
                        if output_ids is not None:
                            output_ids.to('cpu')
                        del inputs
                        if output_ids is not None:
                            del output_ids
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                # Return empty string for failed generations
                outputs_list.extend(["" for _ in batch])

            # Clear cache after processing batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return outputs_list

    def extend_eos_tokens(self):        
        # Add closing braces for Vicuna/Llama eos when using attacker model
        self.eos_token_ids.extend([
            self.tokenizer.encode("}")[1],
            29913, 
            9092,
            16675])


class APIModel(LanguageModel): 

    API_HOST_LINK = "ADD_LINK"
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 20
    
    API_TIMEOUT = 100
    
    MODEL_API_KEY = os.getenv("MODEL_API_KEY")
    
    API_HOST_LINK = ''

    def generate(self, conv: List[Dict], 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        ''' 
            
        output = self.API_ERROR_OUTPUT 
        
        for _ in range(self.API_MAX_RETRY):  
            try:
                
                # Batch generation
                if temperature > 0:
                    # Attack model
                    json = {
                        "top_p": top_p, 
                        "num_beams": 1, 
                        "temperature": temperature, 
                        "do_sample": True,
                        "prompt": '', 
                        "max_new_tokens": max_n_tokens,
                        "system_prompt": conv,
                    } 
                else:
                    # Target model
                    json = {
                        "top_p": 1,
                        "num_beams": 1, 
                        "temperature": 1, # To prevent warning messages
                        "do_sample": False,
                        "prompt": '', 
                        "max_new_tokens": max_n_tokens,
                        "system_prompt": conv,
                    }  

                    # Do not use extra end-of-string tokens in target mode
                    if 'llama' in self.model_name: 
                        json['extra_eos_tokens'] = 0 
                        
    
                if 'llama' in self.model_name:
                    # No system prompt for the Llama model
                    assert json['prompt'] == ''
                    json['prompt'] = deepcopy(json['system_prompt'])
                    del json['system_prompt'] 
                
                resp = urllib3.request(
                            "POST",
                            self.API_HOST_LINK,
                            headers={"Authorization": f"Api-Key {self.MODEL_API_KEY}"},
                            timeout=urllib3.Timeout(self.API_TIMEOUT),
                            json=json,
                )

                resp_json = resp.json()

                if 'vicuna' in self.model_name:
                    if 'error' in resp_json:
                        print(self.API_ERROR_OUTPUT)
    
                    output = resp_json['output']
                    
                else:
                    output = resp_json
                    
                if type(output) == type([]):
                    output = output[0] 
                
                break
            except Exception as e:
                print('exception!', type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
        
            time.sleep(self.API_QUERY_SLEEP)
        return output 
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]

class APIModelLlama7B(APIModel): 
    API_HOST_LINK = LLAMA_API_LINK
    MODEL_API_KEY = os.getenv("LLAMA_API_KEY")

class APIModelVicuna13B(APIModel): 
    API_HOST_LINK = VICUNA_API_LINK 
    MODEL_API_KEY = os.getenv("VICUNA_API_KEY")

class GPT(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 20
    API_TIMEOUT = 20
    
    openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate(self, conv: List[Dict], 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try: 
                
                response = openai.ChatCompletion.create(
                            model = self.model_name,
                            messages = conv,
                            max_tokens = max_n_tokens,
                            temperature = temperature,
                            top_p = top_p,
                            request_timeout = self.API_TIMEOUT,
                            )
                output = response["choices"][0]["message"]["content"]
                break
            except Exception as e: 
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
        
            time.sleep(self.API_QUERY_SLEEP)
        return output 
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]
     
class PaLM():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    default_output = "I'm sorry, but I cannot assist with that request."
    API_KEY = os.getenv("PALM_API_KEY")

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        genai.configure(api_key=self.API_KEY) 

    def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of dictionaries, 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):            
            try:
                completion = genai.chat(
                    messages=conv,
                    temperature=temperature,
                    top_p=top_p
                )
                output = completion.last
                
                if output is None:
                    # If PaLM refuses to output and returns None, we replace it with a default output
                    output = self.default_output
                else:
                    # Use this approximation since PaLM does not allow
                    # to specify max_tokens. Each token is approximately 4 characters.
                    output = output[:(max_n_tokens*4)] 
                break
            except Exception as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
        
            time.sleep(self.API_QUERY_SLEEP)
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]


class GeminiPro():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    default_output = "I'm sorry, but I cannot assist with that request."
    API_KEY = os.getenv("PALM_API_KEY")

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        genai.configure(api_key=self.API_KEY) 

    def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of dictionaries, 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):            
            try:
                model = genai.GenerativeModel(self.model_name)
                output = model.generate_content(
                    contents = conv,
                    generation_config = genai.GenerationConfig(
                        candidate_count = 1,
                        temperature = temperature,
                        top_p = top_p,
                        max_output_tokens=max_n_tokens,
                    )
                )

                if output is None:
                    # If PaLM refuses to output and returns None, we replace it with a default output
                    output = self.default_output
                else:
                    # Use this approximation since PaLM does not allow
                    # to specify max_tokens. Each token is approximately 4 characters.
                    output = output.text
                break
            except Exception as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
        
            time.sleep(self.API_QUERY_SLEEP)
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]

class OpenRouter(GPT):
    """OpenRouter API wrapper that uses the OpenAI-compatible API format"""
    
    def __init__(self, model_name):
        super().__init__(model_name)
        # Configure OpenAI for OpenRouter
        openai.api_base = "https://openrouter.ai/api/v1"
        openai.api_key = os.getenv("OPENROUTER_API_KEY")
        # Strip "openrouter/" prefix for API calls
        self.api_model_name = model_name.replace("openrouter/", "")
        
        # Set provider-specific configurations
        self.provider = self.api_model_name.split('/')[0]  # e.g., "anthropic", "meta-llama"
        
    def batched_generate(self, convs_list, max_n_tokens: int, temperature: float, top_p: float = 1.0):
        """Generate responses for a batch of conversations"""
        outputs = []
        for i, conv in enumerate(convs_list):
            try:
                print(f"\nProcessing request {i+1}/{len(convs_list)}")
                output = self.generate(conv, max_n_tokens, temperature, top_p)
                outputs.append(output)
                # Add a small delay between requests to avoid rate limiting
                time.sleep(1)
            except Exception as e:
                print(f"Error generating response for request {i+1}: {str(e)}")
                # Return a default response that will map to "refusal" score
                outputs.append("I apologize, but I cannot assist with that request.")
                continue
        return outputs

    def generate(self, conv: List[Dict], max_n_tokens: int, temperature: float, top_p: float = 1.0):
        """Generate a response for a single conversation"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Make sure conv is properly formatted as messages
                if isinstance(conv, str):
                    messages = [{"role": "user", "content": conv}]
                elif isinstance(conv, list) and all(isinstance(m, dict) for m in conv):
                    messages = conv
                else:
                    raise ValueError(f"Invalid conv format: {conv}")

                # Add provider-specific headers if needed
                headers = {
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "LLM-Jailbreak"
                }
                
                response = openai.ChatCompletion.create(
                    headers=headers,
                    model=self.api_model_name,
                    messages=messages,
                    max_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    request_timeout=30  # Add timeout
                )
                
                if 'choices' not in response or not response['choices']:
                    raise KeyError("No choices in response")
                    
                output = response["choices"][0]["message"]["content"]
                return output
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    raise e

class ReplicateModel(LanguageModel):
    """Class for interacting with Vicuna hosted on Replicate"""
    
    def __init__(self, model_name):
        super().__init__(model_name)
        self.client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))
        # Use Vicuna 13B model identifier
        self.model_version = "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b"
        self.template = "vicuna_v1.1"  # Keep the same template as local Vicuna

    def batched_generate(self, 
                        prompts_list: List[str],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0):
        """Generate responses for a batch of prompts"""
        outputs = []
        
        for prompt in prompts_list:
            try:
                # Run the model
                output = self.client.run(
                    self.model_version,
                    input={
                        "prompt": prompt,
                        "max_new_tokens": max_n_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                    }
                )
                
                # Replicate returns an iterator, get the actual output
                output = "".join(output)
                outputs.append(output)
                
            except Exception as e:
                print(f"Replicate API error: {str(e)}")
                outputs.append("")  # Return empty string for failed generations
                
            # Add a small delay between requests to avoid rate limits
            time.sleep(0.5)
            
        return outputs
