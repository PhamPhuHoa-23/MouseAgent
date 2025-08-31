#!/usr/bin/env python3
"""
Complete DINOv2 Self-Supervised Learning Workflow
End-to-end pipeline: Data Collection -> Pre-training -> RL Fine-tuning
"""

import argparse
import subprocess
import os
import time
from pathlib import Path
import yaml


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


class DINOWorkflow:
    def __init__(self, config_file: str = "dino_workflow_config.yaml"):
        """Initialize DINO workflow with configuration"""
        self.config_file = config_file
        self.config = self.load_or_create_config()
        
        # Setup directories
        self.setup_directories()
        
        print(f"🧠 DINOv2 Workflow Configuration:")
        print(f"   Dataset: {self.config['data_collection']['output_dir']}")
        print(f"   Pre-training: {self.config['pretraining']['output_dir']}")
        print(f"   Fine-tuning: {self.config['finetuning']['output_dir']}")
    
    def load_or_create_config(self):
        """Load or create workflow configuration"""
        if Path(self.config_file).exists():
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        
        # Create default config
        default_config = {
            'data_collection': {
                'env': 'NormalTrain',
                'episodes': 200,
                'steps_per_episode': 500,
                'output_dir': './dataset_mouse_dino',
                'additional_envs': ['FogTrain', 'RandomTrain'],  # Collect from multiple environments
                'train_val_split': True
            },
            'pretraining': {
                'epochs': 50,
                'batch_size': 32,  # Smaller for GPU memory
                'output_dir': './dino_checkpoints',
                'model': {
                    'img_size': 224,
                    'patch_size': 16,
                    'embed_dim': 384,
                    'depth': 6,
                    'num_heads': 6
                },
                'use_wandb': False
            },
            'finetuning': {
                'architectures': ['dino_encoder'],
                'unfreeze_layers': 2,  # Fine-tune last 2 transformer layers
                'runs_per_network': 3,
                'max_steps': 50000,
                'output_dir': './dino_rl_results'
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        print(f"📝 Created default workflow config: {self.config_file}")
        return default_config
    
    def setup_directories(self):
        """Create necessary directories"""
        dirs_to_create = [
            self.config['data_collection']['output_dir'],
            self.config['pretraining']['output_dir'],
            self.config['finetuning']['output_dir']
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(exist_ok=True, parents=True)
    
    def collect_data(self):
        """Step 1: Collect visual data from Unity environments"""
        print("\n" + "="*60)
        print("🎯 STEP 1: DATA COLLECTION")
        print("="*60)
        
        data_config = self.config['data_collection']
        
        # Collect from main environment
        cmd = [
            'python', 'data_collector.py',
            '--env', data_config['env'],
            '--episodes', str(data_config['episodes']),
            '--steps', str(data_config['steps_per_episode']),
            '--output', data_config['output_dir'],
            '--triplets'
        ]
        
        if data_config.get('train_val_split', True):
            cmd.append('--split')
        
        success = run_command(cmd, f"Collecting data from {data_config['env']}")
        
        # Collect from additional environments if specified
        if data_config.get('additional_envs'):
            for env in data_config['additional_envs']:
                additional_output = f"{data_config['output_dir']}_{env.lower()}"
                cmd = [
                    'python', 'data_collector.py',
                    '--env', env,
                    '--episodes', str(data_config['episodes'] // 2),  # Less episodes for additional envs
                    '--steps', str(data_config['steps_per_episode']),
                    '--output', additional_output,
                    '--triplets',
                    '--split'
                ]
                run_command(cmd, f"Collecting additional data from {env}", check=False)
        
        return success
    
    def pretrain_dino(self):
        """Step 2: DINOv2 self-supervised pre-training"""
        print("\n" + "="*60)
        print("🧠 STEP 2: DINOv2 PRE-TRAINING")
        print("="*60)
        
        # Create DINO training config
        dino_config_path = self.create_dino_config()
        
        # Find training data
        data_dir = Path(self.config['data_collection']['output_dir'])
        train_dir = data_dir / "train"
        
        if not train_dir.exists():
            train_dir = data_dir / "images"  # Fallback if no split was made
        
        if not train_dir.exists():
            print(f"❌ Training data not found at {train_dir}")
            return False
        
        cmd = [
            'python', 'dino_pretrain.py',
            '--data', str(train_dir),
            '--config', dino_config_path,
            '--output', self.config['pretraining']['output_dir']
        ]
        
        if self.config['pretraining'].get('use_wandb', False):
            cmd.append('--wandb')
        
        success = run_command(cmd, "DINOv2 self-supervised pre-training")
        return success
    
    def create_dino_config(self):
        """Create DINOv2 training configuration"""
        config_path = "dino_config.yaml"
        
        dino_config = {
            'model': self.config['pretraining']['model'],
            'head': {
                'out_dim': 65536,
                'bottleneck_dim': 256
            },
            'loss': {
                'out_dim': 65536,
                'teacher_temp': 0.07,
                'student_temp': 0.1
            },
            'optimizer': {
                'lr': 5e-4,
                'weight_decay': 0.04
            },
            'training': {
                'epochs': self.config['pretraining']['epochs'],
                'batch_size': self.config['pretraining']['batch_size'],
                'teacher_momentum': 0.996
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(dino_config, f, default_flow_style=False)
        
        return config_path
    
    def setup_dino_encoder(self):
        """Step 3: Setup DINOv2 encoder for RL training"""
        print("\n" + "="*60)
        print("🔧 STEP 3: DINO ENCODER SETUP")  
        print("="*60)
        
        # Create DINOv2 encoder configuration
        encoder_config = self.create_dino_encoder_config()
        
        # Copy pre-trained checkpoint to accessible location
        checkpoint_src = Path(self.config['pretraining']['output_dir']) / "best_checkpoint.pth"
        checkpoint_dst = Path("dino_pretrained.pth")
        
        if checkpoint_src.exists():
            import shutil
            shutil.copy2(checkpoint_src, checkpoint_dst)
            print(f"✅ Pre-trained checkpoint copied to {checkpoint_dst}")
        else:
            print(f"⚠️ Warning: Pre-trained checkpoint not found at {checkpoint_src}")
            print("   DINOv2 encoder will use random initialization")
        
        return True
    
    def create_dino_encoder_config(self):
        """Create configuration for DINO encoder training"""
        config_path = "config_examples/dino_finetuning.yaml"
        
        config = {
            'training': {
                'runs_per_network': self.config['finetuning']['runs_per_network'],
                'max_steps_override': self.config['finetuning']['max_steps'],
                'timeout_minutes': 180,
                'auto_resume': True,
                'save_checkpoints': True,
                'checkpoint_interval': 5000
            },
            'environment': {
                'name': 'NormalTrain',
                'screen_width': 155,
                'screen_height': 86,
                'additional_args': [
                    '--time-scale', '8'
                ]
            },
            'models': {
                'architectures': ['dino_encoder'],
                'custom_encoders_dir': './Encoders/'
            },
            'logging': {
                'base_dir': self.config['finetuning']['output_dir'],
                'save_configs': True
            },
            'hyperparameters': {
                'base_config': './Config/nature.yaml',
                'sweep_params': {
                    # Specific hyperparams for fine-tuning pre-trained models
                    'behaviors.My Behavior.hyperparameters.learning_rate': [0.0001, 0.0003],  # Lower LR for fine-tuning
                    'behaviors.My Behavior.hyperparameters.batch_size': [128, 256]
                }
            }
        }
        
        Path("config_examples").mkdir(exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"📝 Created DINO fine-tuning config: {config_path}")
        return config_path
    
    def finetune_rl(self):
        """Step 4: Fine-tune DINOv2 encoder on RL task"""
        print("\n" + "="*60)
        print("🎮 STEP 4: RL FINE-TUNING")
        print("="*60)
        
        config_path = "config_examples/dino_finetuning.yaml"
        
        cmd = [
            'python', 'train_advanced.py',
            '--config', config_path
        ]
        
        success = run_command(cmd, "Fine-tuning DINOv2 encoder on RL task")
        return success
    
    def run_complete_workflow(self):
        """Run the complete DINOv2 workflow"""
        print("🚀 Starting Complete DINOv2 Self-Supervised Learning Workflow")
        print("=" * 80)
        
        start_time = time.time()
        
        steps = [
            ("Data Collection", self.collect_data),
            ("DINOv2 Pre-training", self.pretrain_dino), 
            ("Encoder Setup", self.setup_dino_encoder),
            ("RL Fine-tuning", self.finetune_rl)
        ]
        
        results = {}
        
        for step_name, step_func in steps:
            try:
                print(f"\n{'='*20} {step_name} {'='*20}")
                success = step_func()
                results[step_name] = success
                
                if success:
                    print(f"✅ {step_name} completed successfully!")
                else:
                    print(f"❌ {step_name} failed!")
                    break
                    
            except Exception as e:
                print(f"💥 {step_name} crashed: {e}")
                results[step_name] = False
                break
        
        # Summary
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print("🎯 WORKFLOW SUMMARY")
        print(f"{'='*80}")
        
        for step_name, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"   {step_name:<20}: {status}")
        
        print(f"\n⏱️  Total time: {total_time/60:.1f} minutes")
        
        if all(results.values()):
            print("\n🎉 Complete workflow finished successfully!")
            print("\n📊 Next steps:")
            print("   1. Check results in:", self.config['finetuning']['output_dir'])
            print("   2. Compare with baseline models")
            print("   3. Run evaluation on test environments")
        else:
            print(f"\n😞 Workflow stopped at: {next(k for k,v in results.items() if not v)}")


def main():
    parser = argparse.ArgumentParser(description="DINOv2 Self-Supervised Learning Workflow")
    parser.add_argument("--step", choices=['collect', 'pretrain', 'setup', 'finetune', 'all'],
                       default='all', help="Which step to run")
    parser.add_argument("--config", type=str, default="dino_workflow_config.yaml",
                       help="Workflow configuration file")
    
    args = parser.parse_args()
    
    workflow = DINOWorkflow(args.config)
    
    if args.step == 'collect':
        workflow.collect_data()
    elif args.step == 'pretrain':
        workflow.pretrain_dino()
    elif args.step == 'setup':
        workflow.setup_dino_encoder()
    elif args.step == 'finetune':
        workflow.finetune_rl()
    elif args.step == 'all':
        workflow.run_complete_workflow()


if __name__ == "__main__":
    main()
