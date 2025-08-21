#!/usr/bin/env python3
"""
Visual Data Collection for Self-Supervised Learning
Collect images from Unity environment to create DINOv2 dataset
"""

import os
import cv2
import numpy as np
import subprocess
import time
from pathlib import Path
import argparse
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import yaml


class VisualDataCollector:
    def __init__(self, env_path: str, output_dir: str = "./dataset"):
        """Initialize data collector"""
        self.env_path = env_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        self.image_count = 0
        self.metadata = []
        
    def collect_random_episodes(self, num_episodes: int = 100, steps_per_episode: int = 1000):
        """Collect data from random policy episodes"""
        print(f"🎯 Collecting {num_episodes} episodes with random policy...")
        
        # Setup Unity environment
        env = UnityEnvironment(file_name=self.env_path, seed=42, side_channels=[])
        env.reset()
        
        behavior_names = list(env.behavior_specs.keys())
        behavior_name = behavior_names[0]
        spec = env.behavior_specs[behavior_name]
        
        print(f"Environment: {behavior_name}")
        print(f"Visual observations: {len(spec.observation_specs)}")
        print(f"Action space: {spec.action_spec}")
        
        try:
            for episode in range(num_episodes):
                print(f"\n📸 Episode {episode+1}/{num_episodes}")
                env.reset()
                
                episode_images = []
                
                for step in range(steps_per_episode):
                    # Get observations
                    decision_steps, terminal_steps = env.get_steps(behavior_name)
                    
                    if len(decision_steps) == 0:
                        break
                        
                    # Extract visual observations
                    visual_obs = decision_steps.obs[0]  # First visual observation
                    
                    # Save images
                    for agent_id, obs in zip(decision_steps.agent_id, visual_obs):
                        self._save_observation(obs, episode, step, agent_id)
                        episode_images.append(self.image_count - 1)
                    
                    # Take random actions
                    action_size = spec.action_spec.continuous_size + spec.action_spec.discrete_size
                    if spec.action_spec.continuous_size > 0:
                        continuous_actions = np.random.randn(len(decision_steps), spec.action_spec.continuous_size)
                    else:
                        continuous_actions = np.empty((len(decision_steps), 0))
                        
                    if spec.action_spec.discrete_size > 0:
                        discrete_actions = np.random.randint(0, 2, size=(len(decision_steps), spec.action_spec.discrete_size))
                    else:
                        discrete_actions = np.empty((len(decision_steps), 0), dtype=int)
                    
                    action = ActionTuple(continuous_actions, discrete_actions)
                    env.set_actions(behavior_name, action)
                    env.step()
                    
                    # Progress update
                    if step % 100 == 0:
                        print(f"  Step {step}/{steps_per_episode}, Images: {self.image_count}")
                
                self.metadata.append({
                    'episode': episode,
                    'images': episode_images,
                    'total_steps': step + 1
                })
                
        finally:
            env.close()
            
        print(f"\n✅ Collection complete! Total images: {self.image_count}")
        self._save_metadata()
    
    def collect_diverse_data(self, collection_time_minutes: int = 30):
        """Collect diverse data by running environment for specified time"""
        print(f"🎯 Collecting diverse data for {collection_time_minutes} minutes...")
        
        # Run environment in no-graphics mode for faster collection
        cmd = [
            self.env_path,
            "-batchmode",
            "-nographics", 
            f"--episodes={collection_time_minutes * 10}",  # Roughly 10 episodes per minute
            "--screen-width=155",
            "--screen-height=86"
        ]
        
        print(f"Command: {' '.join(cmd)}")
        
        start_time = time.time()
        
        # This would need custom Unity script to save images
        # For now, we use the random policy method
        self.collect_random_episodes(num_episodes=collection_time_minutes * 5, steps_per_episode=200)
    
    def _save_observation(self, observation: np.ndarray, episode: int, step: int, agent_id: int):
        """Save a single visual observation"""
        # Convert from BHWC to HWC (remove batch dimension if present)
        if observation.ndim == 4:
            observation = observation[0]
        
        # Convert to uint8 if needed
        if observation.dtype != np.uint8:
            if observation.max() <= 1.0:
                observation = (observation * 255).astype(np.uint8)
            else:
                observation = observation.astype(np.uint8)
        
        # Save image
        filename = f"img_{self.image_count:06d}_ep{episode}_step{step}_agent{agent_id}.png"
        filepath = self.output_dir / "images" / filename
        
        # Handle different image formats
        if observation.shape[-1] == 3:  # RGB
            cv2.imwrite(str(filepath), cv2.cvtColor(observation, cv2.COLOR_RGB2BGR))
        elif observation.shape[-1] == 1:  # Grayscale
            cv2.imwrite(str(filepath), observation[:, :, 0])
        else:  # RGBA or other
            cv2.imwrite(str(filepath), observation[:, :, :3])
        
        self.image_count += 1
        
        return filename
    
    def _save_metadata(self):
        """Save collection metadata"""
        metadata = {
            'total_images': self.image_count,
            'episodes': self.metadata,
            'collection_info': {
                'env_path': self.env_path,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'image_format': 'PNG',
                'resolution': '155x86'  # Default resolution
            }
        }
        
        metadata_file = self.output_dir / "metadata" / "collection_info.yaml"
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        print(f"💾 Metadata saved to: {metadata_file}")
    
    def create_train_val_split(self, val_ratio: float = 0.2):
        """Create train/validation split for the collected data"""
        images_dir = self.output_dir / "images"
        train_dir = self.output_dir / "train"
        val_dir = self.output_dir / "val"
        
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        
        # Get all image files
        image_files = list(images_dir.glob("*.png"))
        np.random.shuffle(image_files)
        
        # Split
        split_idx = int(len(image_files) * (1 - val_ratio))
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        print(f"📊 Creating train/val split:")
        print(f"   Train: {len(train_files)} images")
        print(f"   Val: {len(val_files)} images")
        
        # Create symlinks or copy files
        for img_file in train_files:
            (train_dir / img_file.name).write_bytes(img_file.read_bytes())
            
        for img_file in val_files:
            (val_dir / img_file.name).write_bytes(img_file.read_bytes())
        
        print("✅ Train/val split created!")


def main():
    parser = argparse.ArgumentParser(description="Collect visual data from Unity environment")
    parser.add_argument("--env", type=str, default="NormalTrain",
                        help="Environment name (NormalTrain, FogTrain, etc.)")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of episodes to collect")
    parser.add_argument("--steps", type=int, default=500,
                        help="Steps per episode")
    parser.add_argument("--output", type=str, default="./dataset_mouse",
                        help="Output directory")
    parser.add_argument("--split", action="store_true",
                        help="Create train/val split after collection")
    
    args = parser.parse_args()
    
    # Build environment path
    env_path = f"./Builds/{args.env}/2D go to target v1.exe"
    
    if not Path(env_path).exists():
        print(f"❌ Environment not found: {env_path}")
        return
    
    # Collect data
    collector = VisualDataCollector(env_path, args.output)
    
    print(f"🚀 Starting data collection...")
    print(f"   Environment: {args.env}")
    print(f"   Episodes: {args.episodes}")
    print(f"   Steps per episode: {args.steps}")
    print(f"   Output: {args.output}")
    
    try:
        collector.collect_random_episodes(args.episodes, args.steps)
        
        if args.split:
            collector.create_train_val_split()
            
        print(f"\n🎉 Data collection complete!")
        print(f"📁 Dataset location: {args.output}")
        print(f"📊 Total images: {collector.image_count}")
        
    except Exception as e:
        print(f"❌ Collection failed: {e}")
        raise


if __name__ == "__main__":
    main()
