#!/usr/bin/env python3
"""
DINO ResNet34 Workflow 
Streamlined workflow using ResNet34 backbone (25.0M parameters)
"""

import argparse
import subprocess
import time
from pathlib import Path


def run_command(cmd, description, check=True):
    """Run command with description"""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print("=" * 60)
    
    if isinstance(cmd, str):
        result = subprocess.run(cmd, shell=True, check=check)
    else:
        result = subprocess.run(cmd, check=check)
    
    if result.returncode == 0:
        print(f"✅ {description} - SUCCESS")
    else:
        print(f"❌ {description} - FAILED")
        if check:
            raise RuntimeError(f"Command failed: {cmd}")
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="DINO ResNet34 Workflow")
    parser.add_argument("--step", choices=['pretrain', 'finetune', 'compare', 'all'],
                       default='all', help="Which step to run")
    parser.add_argument("--data", type=str, default="dataset_mouse_dino/train",
                       help="Training data directory")
    
    args = parser.parse_args()
    
    if args.step in ['pretrain', 'all']:
        print("🧠 DINO ResNet34 Pre-training")
        print("=" * 50)
        
        cmd = [
            'python', 'dino_resnet.py',
            '--data', args.data,
            '--output', './enhanced_checkpoints'
        ]
        
        success = run_command(cmd, "DINO ResNet34 pre-training")
        if not success and args.step == 'all':
            return
    
    if args.step in ['finetune', 'all']:
        print("\n🎮 RL Fine-tuning với DINO ResNet34")  
        print("=" * 50)
        
        cmd = [
            'python', 'train_advanced.py',
            '--config', 'config_examples/dino_resnet34_finetuning.yaml'
        ]
        
        success = run_command(cmd, "DINO ResNet34 RL fine-tuning")
        if not success and args.step == 'all':
            return
    
    if args.step in ['compare', 'all']:
        print("\n📊 Comparison with Baselines")
        print("=" * 50)
        
        # Create comparison config
        comparison_config = """
# DINO ResNet34 vs Baselines
training:
  runs_per_network: 3
  max_steps_override: 30000
  timeout_minutes: 90
  continue_on_error: true

environment:
  name: NormalTrain
  screen_width: 155
  screen_height: 86

models:
  architectures:
    - nature_cnn        # Baseline
    - dino_resnet34_encoder  # Our method (ResNet34)
  custom_encoders_dir: ./Encoders/

logging:
  base_dir: ./comparison_results_resnet34

hyperparameters:
  base_config: ./Config/nature.yaml
"""
        
        config_path = "config_examples/dino_resnet34_comparison.yaml"
        with open(config_path, 'w') as f:
            f.write(comparison_config)
        
        cmd = [
            'python', 'train_advanced.py',
            '--config', config_path
        ]
        
        run_command(cmd, "Comparison training")
    
    print("\n🎉 DINO ResNet34 workflow completed!")
    print("\n📊 Results locations:")
    print("  • Pre-training: ./enhanced_checkpoints/")
    print("  • RL training: ./dino_resnet34_rl_results/")
    print("  • Comparison: ./comparison_results_resnet34/")


if __name__ == "__main__":
    main()
