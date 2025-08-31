#!/usr/bin/env python3
"""
Visual Data Collection for Self-Supervised Learning
Collect images from Unity environment to create DINOv2 dataset
"""

import os
from PIL import Image
import numpy as np
import subprocess
import time
from pathlib import Path
import argparse
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import yaml
import json


class VisualDataCollector:
    def __init__(self, env_path: str, output_dir: str = "./dataset", policy: str = "random", onnx_path: str = "", worker_id: int = 0, base_port: int = 5005):
        """Initialize data collector"""
        self.env_path = env_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.policy = policy
        self.onnx_path = onnx_path
        self.worker_id = worker_id
        self.base_port = base_port
        
        # Create subdirectories
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        self.image_count = 0
        self.metadata = []
        self.triplets = []
        self._onnx_session = None
        
    def collect_random_episodes(self, num_episodes: int = 100, steps_per_episode: int = 1000):
        """Collect data from random policy episodes"""
        print(f"🎯 Collecting {num_episodes} episodes with random policy...")
        
        # Setup Unity environment
        env = UnityEnvironment(file_name=self.env_path, seed=42, side_channels=[], worker_id=self.worker_id, base_port=self.base_port)
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

    def _ensure_onnx_session(self):
        if self._onnx_session is None:
            import onnxruntime as ort
            self._onnx_session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"]) 
            self._onnx_input_names = [inp.name for inp in self._onnx_session.get_inputs()]
            self._onnx_output_names = [out.name for out in self._onnx_session.get_outputs()]
            print(f"🧩 ONNX loaded: inputs={self._onnx_input_names}, outputs={self._onnx_output_names}")

    def _policy_actions(self, spec, decision_steps):
        """Return ActionTuple for given decision_steps according to selected policy."""
        num_agents = len(decision_steps)
        if self.policy == "onnx":
            self._ensure_onnx_session()
            # Assume first visual obs is the policy input, BHWC floats
            obs = decision_steps.obs[0]
            # Convert BHWC -> NCHW as expected by exported policy: (B, 1, 86, 155)
            obs = np.transpose(obs, (0, 3, 1, 2)).astype(np.float32, copy=False)
            # Feed to ONNX
            inputs = {self._onnx_input_names[0]: obs}
            # Prefer explicit continuous action output if present
            out_names = [name for name in self._onnx_output_names if 'continuous_actions' in name]
            if not out_names:
                out_names = self._onnx_output_names
            outputs = self._onnx_session.run(out_names, inputs)
            # Choose the first 2D output matching batch size
            cont_size = spec.action_spec.continuous_size
            chosen = None
            for out in outputs:
                if out.ndim == 2 and out.shape[0] == num_agents:
                    chosen = out
                    break
            if chosen is None:
                raise RuntimeError("ONNX model outputs do not include a valid 2D action tensor")
            # If action head larger, slice to expected continuous size
            if cont_size > 0 and chosen.shape[1] != cont_size:
                chosen = chosen[:, :cont_size]
            continuous_actions = chosen.astype(np.float32, copy=False)
            discrete_actions = np.empty((num_agents, 0), dtype=int)
            return ActionTuple(continuous_actions, discrete_actions)

        # Default random policy
        if spec.action_spec.continuous_size > 0:
            continuous_actions = np.random.randn(num_agents, spec.action_spec.continuous_size)
        else:
            continuous_actions = np.empty((num_agents, 0))
        if spec.action_spec.discrete_size > 0:
            discrete_actions = np.random.randint(0, 2, size=(num_agents, spec.action_spec.discrete_size))
        else:
            discrete_actions = np.empty((num_agents, 0), dtype=int)
        return ActionTuple(continuous_actions, discrete_actions)

    def collect_triplets_random(self, num_episodes: int = 100, steps_per_episode: int = 1000):
        """Collect (image_t, action, image_{t+1}) triplets using selected policy (random/onnx)."""
        policy_name = self.policy
        print(f"🎯 Collecting {num_episodes} episodes of triplets with {policy_name} policy...")

        env = UnityEnvironment(file_name=self.env_path, seed=42, side_channels=[], worker_id=self.worker_id, base_port=self.base_port)
        env.reset()

        behavior_names = list(env.behavior_specs.keys())
        behavior_name = behavior_names[0]
        spec = env.behavior_specs[behavior_name]

        print(f"Environment: {behavior_name}")
        print(f"Visual observations: {len(spec.observation_specs)}")
        print(f"Action space: {spec.action_spec}")

        triplets_path = self.output_dir / "metadata" / "triplets.jsonl"
        # Write as JSON Lines for scalability
        with open(triplets_path, 'w') as triplet_file:
            try:
                for episode in range(num_episodes):
                    print(f"\n🎞️  Episode {episode+1}/{num_episodes}")
                    env.reset()

                    step = 0
                    while step < steps_per_episode:
                        decision_steps, terminal_steps = env.get_steps(behavior_name)

                        if len(decision_steps) == 0:
                            break

                        # Map current agents to indices
                        id_to_index_now = {int(aid): idx for idx, aid in enumerate(decision_steps.agent_id)}

                        # Current observations per agent
                        visual_obs_now = decision_steps.obs[0]

                        # Sample actions from selected policy
                        action = self._policy_actions(spec, decision_steps)

                        # Save image_t for each agent now
                        image_t_filenames = {}
                        for agent_id, obs in zip(decision_steps.agent_id, visual_obs_now):
                            fname_t = self._save_observation(obs, episode, step, int(agent_id))
                            image_t_filenames[int(agent_id)] = fname_t

                        # Apply actions and step
                        env.set_actions(behavior_name, action)
                        env.step()

                        # Next observations
                        next_decision_steps, next_terminal_steps = env.get_steps(behavior_name)
                        id_to_index_next = {int(aid): idx for idx, aid in enumerate(next_decision_steps.agent_id)}
                        id_to_index_term = {int(aid): idx for idx, aid in enumerate(next_terminal_steps.agent_id)}

                        # For each agent present before, try to record next image and the action used
                        for agent_id_int, idx_now in id_to_index_now.items():
                            # Retrieve action taken by this agent
                            cont = action.continuous[idx_now].tolist() if action.continuous.size > 0 else []
                            disc = action.discrete[idx_now].tolist() if action.discrete.size > 0 else []

                            # Try to find next observation for the same agent
                            next_obs = None
                            if agent_id_int in id_to_index_next:
                                next_obs = next_decision_steps.obs[0][id_to_index_next[agent_id_int]]
                            elif agent_id_int in id_to_index_term:
                                next_obs = next_terminal_steps.obs[0][id_to_index_term[agent_id_int]]

                            if next_obs is None:
                                continue

                            fname_t1 = self._save_observation(next_obs, episode, step + 1, agent_id_int)

                            record = {
                                'episode': int(episode),
                                'step': int(step),
                                'agent_id': int(agent_id_int),
                                'image_t': str(image_t_filenames.get(agent_id_int, "")),
                                'image_t1': str(fname_t1),
                                'action': {
                                    'continuous': cont,
                                    'discrete': disc
                                }
                            }
                            triplet_file.write(json.dumps(record) + "\n")

                        step += 1

                        if step % 100 == 0:
                            print(f"  Step {step}/{steps_per_episode}, Images: {self.image_count}")

            finally:
                env.close()

        print(f"\n✅ Triplet collection complete! Total images: {self.image_count}")
        print(f"💾 Triplets saved to: {triplets_path}")
    
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

        # Normalize channels to RGB or L
        if observation.shape[-1] == 1:
            img = Image.fromarray(observation[:, :, 0], mode='L')
        else:
            # Use first 3 channels as RGB if more are present (e.g., RGBA)
            img = Image.fromarray(observation[:, :, :3], mode='RGB')

        img.save(str(filepath))
        
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
    parser.add_argument("--triplets", action="store_true",
                        help="Collect (image_t, action, image_{t+1}) triplets as JSONL")
    parser.add_argument("--policy", type=str, default="random", choices=["random", "onnx"],
                        help="Policy to generate actions during collection")
    parser.add_argument("--onnx-path", type=str, default="",
                        help="Path to ONNX policy (required if --policy onnx)")
    parser.add_argument("--worker-id", type=int, default=0,
                        help="Worker id to avoid port conflicts")
    parser.add_argument("--base-port", type=int, default=5005,
                        help="Base port for ML-Agents RPC (default 5005)")
    
    args = parser.parse_args()
    
    # Build environment path
    env_path = f"./Builds/{args.env}/2D go to target v1.exe"
    
    if not Path(env_path).exists():
        print(f"❌ Environment not found: {env_path}")
        return
    
    # Collect data
    collector = VisualDataCollector(env_path, args.output, policy=args.policy, onnx_path=args.onnx_path, worker_id=args.worker_id, base_port=args.base_port)
    
    print(f"🚀 Starting data collection...")
    print(f"   Environment: {args.env}")
    print(f"   Episodes: {args.episodes}")
    print(f"   Steps per episode: {args.steps}")
    print(f"   Output: {args.output}")
    
    try:
        if args.triplets:
            collector.collect_triplets_random(args.episodes, args.steps)
        else:
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
