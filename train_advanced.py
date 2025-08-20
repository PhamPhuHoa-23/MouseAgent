#!/usr/bin/env python3
"""
Advanced Training Script with Configuration Support
Supports ResNet50 and other custom encoders
"""

import argparse
import subprocess
import os
import yaml
import itertools
from pathlib import Path
import time


def load_config(config_path):
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_training_with_params(base_config, env_config, model_config, logging_config, hyperparams):
    """Run a single training session with specific parameters"""
    
    # Create run ID
    timestamp = int(time.time())
    architecture = model_config['architectures'][0] if model_config['architectures'] else 'default'
    run_id = f"{architecture}_{timestamp}"
    
    # Build command
    cmd = [
        'mlagents-learn',
        hyperparams['base_config'],
        f'--env=Builds/{env_config["name"]}/2D go to target v1.exe',
        f'--run-id={run_id}',
        '--num-envs=1'
    ]
    
    # Add environment arguments
    if 'additional_args' in env_config:
        cmd.extend(env_config['additional_args'])
    
    # Add max steps override
    if 'max_steps_override' in base_config:
        cmd.extend(['--max-steps', str(base_config['max_steps_override'])])
    
    # Set environment variables for custom encoders
    env_vars = os.environ.copy()
    if 'architectures' in model_config and model_config['architectures']:
        env_vars['CUSTOM_ENCODER'] = model_config['architectures'][0]
        env_vars['ENCODERS_PATH'] = model_config.get('custom_encoders_dir', './Encoders/')
    
    print(f"\n🚀 Running training: {run_id}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Custom encoder: {model_config['architectures'][0] if model_config['architectures'] else 'None'}")
    
    try:
        result = subprocess.run(cmd, env=env_vars, check=True)
        print(f"✅ Training completed: {run_id}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {run_id} - {e}")
        return False


def run_hyperparameter_sweep(config):
    """Run training with hyperparameter sweep"""
    
    base_config = config['training']
    env_config = config['environment']
    model_config = config['models']
    logging_config = config.get('logging', {})
    hyperparams = config.get('hyperparameters', {})
    
    # Get sweep parameters
    sweep_params = hyperparams.get('sweep_params', {})
    
    if not sweep_params:
        # Single run without sweep
        print("🔧 Running single training session (no hyperparameter sweep)")
        return run_training_with_params(base_config, env_config, model_config, logging_config, hyperparams)
    
    print("🔄 Running hyperparameter sweep...")
    
    # Generate all parameter combinations
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"📊 Total combinations: {len(param_combinations)}")
    
    success_count = 0
    
    for combo_idx, combo in enumerate(param_combinations):
        print(f"\n{'='*60}")
        print(f"🎯 Combination {combo_idx + 1}/{len(param_combinations)}")
        
        # Create modified config with current parameters
        for param_name, param_value in zip(param_names, combo):
            print(f"   {param_name}: {param_value}")
        
        # Run training with this combination
        success = run_training_with_params(base_config, env_config, model_config, logging_config, hyperparams)
        if success:
            success_count += 1
    
    print(f"\n🎉 Hyperparameter sweep completed!")
    print(f"📊 Success rate: {success_count}/{len(param_combinations)} ({success_count/len(param_combinations)*100:.1f}%)")
    
    return success_count > 0


def run_multiple_runs(config):
    """Run multiple training runs as specified in config"""
    
    runs_per_network = config['training'].get('runs_per_network', 1)
    architectures = config['models'].get('architectures', ['default'])
    
    total_success = 0
    total_runs = len(architectures) * runs_per_network
    
    print(f"🚀 Starting {total_runs} training runs")
    print(f"   Architectures: {architectures}")
    print(f"   Runs per architecture: {runs_per_network}")
    
    for arch_idx, architecture in enumerate(architectures):
        print(f"\n{'='*80}")
        print(f"🏗️  Architecture {arch_idx + 1}/{len(architectures)}: {architecture}")
        print(f"{'='*80}")
        
        # Update config for this architecture
        config['models']['architectures'] = [architecture]
        
        for run in range(runs_per_network):
            print(f"\n🔄 Run {run + 1}/{runs_per_network} for {architecture}")
            
            success = run_hyperparameter_sweep(config)
            if success:
                total_success += 1
    
    print(f"\n{'='*80}")
    print(f"🎯 ALL TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"📊 Overall success: {total_success}/{total_runs} ({total_success/total_runs*100:.1f}%)")
    
    return total_success > 0


def main():
    parser = argparse.ArgumentParser(description="Advanced Training with Custom Encoders")
    parser.add_argument("--config", required=True, help="YAML configuration file")
    parser.add_argument("--single-run", action="store_true", help="Run single session only")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("🔧 Advanced Training Script")
    print("="*50)
    print(f"Config file: {args.config}")
    print(f"Architectures: {config['models'].get('architectures', ['default'])}")
    
    if args.single_run:
        success = run_hyperparameter_sweep(config)
    else:
        success = run_multiple_runs(config)
    
    if success:
        print("\n🎉 Training completed successfully!")
        results_dir = config.get('logging', {}).get('base_dir', './results')
        print(f"📊 Check results in: {results_dir}")
    else:
        print("\n😞 Training failed!")
        exit(1)


if __name__ == "__main__":
    main()

