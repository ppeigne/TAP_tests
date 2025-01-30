import os
import pytz
import wandb
import pandas as pd
from datetime import datetime

from os import listdir
from os.path import isfile, join

import common 

class WandBLogger:
    """Logger class for Weights & Biases"""
    
    def __init__(self, args, system_prompt):
        """Initialize WandB logger"""
        self.args = args
        
        # Initialize W&B
        self.run = wandb.init(
            project="TAP_multi_target",
            config={
                "attack_model": args.attack_model,
                "target_model": args.target_model,
                "goal": args.goal,
                "target_str": args.target_str,
                "system_prompt": system_prompt,
                "width": args.width,
                "branching_factor": args.branching_factor,
                "depth": args.depth,
                "n_streams": args.n_streams,
                "keep_last_n": args.keep_last_n,
            },
            name=f"{args.target_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        
        # Create directory for storing results if it doesn't exist
        self.results_dir = "results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        # Initialize results tracking
        self.results = []

    def log(self, iteration, model_name, attack_list, target_response_list, 
            judge_scores, on_topic_scores, conv_ids, parent_conv_ids):
        """Log results to W&B and save locally"""
        
        # Prepare data for logging
        for i in range(len(attack_list)):
            log_entry = {
                'iteration': iteration,
                'model_name': model_name,
                'prompt': attack_list[i]['prompt'],
                'improvement': attack_list[i]['improvement'],
                'target_response': target_response_list[i] if target_response_list else None,
                'judge_score': judge_scores[i] if judge_scores else None,
                'on_topic_score': on_topic_scores[i] if on_topic_scores else None,
                'conv_id': conv_ids[i],
                'parent_conv_id': parent_conv_ids[i]
            }
            
            # Add to results list
            self.results.append(log_entry)
            
            # Log to W&B
            wandb.log({
                'iteration': iteration,
                'model': model_name,
                'judge_score': log_entry['judge_score'],
                'on_topic_score': log_entry['on_topic_score'],
                'prompt_length': len(log_entry['prompt']),
                'response_length': len(log_entry['target_response']) if log_entry['target_response'] else 0
            })
            
        # Save results periodically
        if iteration % 5 == 0:
            self._save_results()

    def finish(self):
        """Finish logging and save final results"""
        self._save_results()
        if self.run is not None:
            wandb.finish()

    def _save_results(self):
        """Save results to CSV file"""
        if self.results:
            df = pd.DataFrame(self.results)
            filename = f"{self.results_dir}/results_{self.args.target_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
