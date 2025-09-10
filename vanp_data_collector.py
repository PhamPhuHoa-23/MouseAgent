# #!/usr/bin/env python3
# """
# VANP Data Collector từ Trained Models
# Sử dụng các trained models để synthesize data cho VANP training
# """

# import os
# import json
# import time
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from PIL import Image
# import argparse
# import subprocess
# from mlagents_envs.environment import UnityEnvironment
# from mlagents_envs.base_env import ActionTuple
# import torch
# import onnxruntime as ort


# class VANPDataCollector:
#     def __init__(self, 
#                  output_dir="./vanp_dataset", 
#                  models_dir="./results",
#                  builds_dir="./Builds"):
#         """
#         Initialize VANP data collector
        
#         Args:
#             output_dir: Output directory cho VANP dataset
#             models_dir: Directory chứa trained models
#             builds_dir: Directory chứa Unity builds
#         """
#         self.output_dir = Path(output_dir)
#         self.models_dir = Path(models_dir)
#         self.builds_dir = Path(builds_dir)
        
#         # Setup directories
#         self.setup_directories()
        
#         # Find trained models
#         self.trained_models = self.discover_trained_models()
        
#         # Initialize ONNX sessions
#         self.model_sessions = {}
        
#         print(f"🎯 VANP Data Collector initialized")
#         print(f"   Output: {self.output_dir}")
#         print(f"   Found {len(self.trained_models)} trained models")
        
#     def setup_directories(self):
#         """Setup output directories"""
#         (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
#         (self.output_dir / "episodes").mkdir(parents=True, exist_ok=True)
#         (self.output_dir / "vanp_samples").mkdir(parents=True, exist_ok=True)
        
#     def discover_trained_models(self):
#         """Discover trained models trong results directory"""
#         models = []
        
#         # Find ONNX models
#         for onnx_file in self.models_dir.rglob("*.onnx"):
#             model_info = {
#                 'path': str(onnx_file),
#                 'type': 'onnx',
#                 'name': onnx_file.stem,
#                 'architecture': self.extract_architecture_from_path(onnx_file)
#             }
#             models.append(model_info)
            
#         # Find PyTorch models
#         for pt_file in self.models_dir.rglob("*.pt"):
#             model_info = {
#                 'path': str(pt_file),
#                 'type': 'pytorch',  
#                 'name': pt_file.stem,
#                 'architecture': self.extract_architecture_from_path(pt_file)
#             }
#             models.append(model_info)
            
#         print(f"📊 Discovered models:")
#         for model in models:
#             print(f"   {model['name']} ({model['architecture']}) - {model['type']}")
            
#         return models
    
#     def extract_architecture_from_path(self, model_path):
#         """Extract architecture từ model path"""
#         path_str = str(model_path).lower()
        
#         if 'dino' in path_str:
#             if 'resnet' in path_str:
#                 return 'dino_resnet'
#             else:
#                 return 'dino'
#         elif 'nature' in path_str:
#             return 'nature_cnn'
#         elif 'resnet' in path_str:
#             return 'resnet'
#         elif 'simple' in path_str:
#             return 'simple'
#         else:
#             return 'unknown'
    
#     def load_model(self, model_info):
#         """Load model for inference"""
#         model_path = model_info['path']
        
#         if model_info['type'] == 'onnx':
#             try:
#                 session = ort.InferenceSession(
#                     model_path, 
#                     providers=['CPUExecutionProvider']
#                 )
#                 self.model_sessions[model_info['name']] = session
#                 print(f"✅ Loaded ONNX model: {model_info['name']}")
#                 return True
#             except Exception as e:
#                 print(f"❌ Failed to load ONNX model {model_info['name']}: {e}")
#                 return False
#         else:
#             # PyTorch models cần specific loading logic
#             print(f"⚠️  PyTorch model loading not implemented: {model_info['name']}")
#             return False
    
#     def collect_episode_from_model(self, model_info, env_name="NormalTrain", 
#                                    episode_length=1000, episode_id=0):
#         """
#         Collect một episode từ một trained model
        
#         Returns:
#             episode_data: Dict chứa visual observations, actions, và metadata
#         """
#         print(f"🎮 Collecting episode {episode_id} từ {model_info['name']}")
        
#         # Setup Unity environment
#         env_path = self.builds_dir / env_name / "2D go to target v1.exe"
#         if not env_path.exists():
#             print(f"❌ Environment not found: {env_path}")
#             return None
            
#         try:
#             env = UnityEnvironment(
#                 file_name=str(env_path),
#                 worker_id=episode_id % 10,  # Avoid port conflicts
#                 seed=42 + episode_id,
#                 side_channels=[]
#             )
#             env.reset()
            
#             behavior_names = list(env.behavior_specs.keys())
#             behavior_name = behavior_names[0]
#             spec = env.behavior_specs[behavior_name]
            
#             episode_data = {
#                 'episode_id': episode_id,
#                 'model_name': model_info['name'],
#                 'architecture': model_info['architecture'],
#                 'env_name': env_name,
#                 'frames': [],
#                 'actions': [],
#                 'rewards': [],
#                 'dones': [],
#                 'timestamps': [],
#                 'success': False
#             }
            
#             step_count = 0
#             start_time = time.time()
            
#             while step_count < episode_length:
#                 # Get observations
#                 decision_steps, terminal_steps = env.get_steps(behavior_name)
                
#                 if len(decision_steps) == 0 and len(terminal_steps) == 0:
#                     break
                    
#                 # Process decision steps
#                 if len(decision_steps) > 0:
#                     # Get visual observation
#                     visual_obs = decision_steps.obs[0]  # First visual observation
                    
#                     # Save frame
#                     for agent_id, obs in zip(decision_steps.agent_id, visual_obs):
#                         frame_data = {
#                             'step': step_count,
#                             'agent_id': int(agent_id),
#                             'timestamp': time.time() - start_time,
#                             'observation': obs  # Keep raw observation
#                         }
#                         episode_data['frames'].append(frame_data)
                    
#                     # Get action từ trained model
#                     action = self.get_model_action(model_info, visual_obs)
                    
#                     # Save action
#                     action_data = {
#                         'step': step_count,
#                         'action': action.continuous.tolist() if hasattr(action, 'continuous') else action,
#                         'timestamp': time.time() - start_time
#                     }
#                     episode_data['actions'].append(action_data)
                    
#                     # Apply action
#                     env.set_actions(behavior_name, action)
                    
#                 # Process terminal steps
#                 if len(terminal_steps) > 0:
#                     # Episode ended
#                     episode_data['success'] = True
#                     break
                    
#                 # Step environment
#                 env.step()
#                 step_count += 1
                
#                 # Progress update
#                 if step_count % 100 == 0:
#                     print(f"   Step {step_count}/{episode_length}")
            
#             env.close()
            
#             print(f"✅ Episode collected: {len(episode_data['frames'])} frames, {len(episode_data['actions'])} actions")
#             return episode_data
            
#         except Exception as e:
#             print(f"❌ Episode collection failed: {e}")
#             return None
    
#     def get_model_action(self, model_info, visual_obs):
#         """Get action từ trained model"""
#         model_name = model_info['name']
        
#         if model_name not in self.model_sessions:
#             # Use random action as fallback
#             return self.get_random_action(visual_obs.shape[0])
        
#         try:
#             session = self.model_sessions[model_name]
            
#             # Prepare input
#             # ML-Agents ONNX models expect BCHW input
#             if len(visual_obs.shape) == 4:  # BHWC
#                 input_obs = visual_obs.transpose(0, 3, 1, 2).astype(np.float32)
#             else:
#                 input_obs = visual_obs.astype(np.float32)
                
#             # Get input name
#             input_name = session.get_inputs()[0].name
            
#             # Run inference
#             outputs = session.run(None, {input_name: input_obs})
            
#             # Find action output (usually first 2D output)
#             action_output = None
#             for output in outputs:
#                 if len(output.shape) == 2 and output.shape[0] == visual_obs.shape[0]:
#                     action_output = output
#                     break
            
#             if action_output is not None:
#                 # Create ActionTuple
#                 continuous_actions = action_output.astype(np.float32)
#                 discrete_actions = np.empty((visual_obs.shape[0], 0), dtype=np.int32)
#                 return ActionTuple(continuous_actions, discrete_actions)
#             else:
#                 return self.get_random_action(visual_obs.shape[0])
                
#         except Exception as e:
#             print(f"⚠️  Model inference failed: {e}")
#             return self.get_random_action(visual_obs.shape[0])
    
#     def get_random_action(self, batch_size):
#         """Generate random actions as fallback"""
#         continuous_actions = np.random.randn(batch_size, 3).astype(np.float32)  # 3D continuous
#         discrete_actions = np.empty((batch_size, 0), dtype=np.int32)
#         return ActionTuple(continuous_actions, discrete_actions)
    
#     def save_episode(self, episode_data):
#         """Save episode data to disk"""
#         if episode_data is None:
#             return None
            
#         episode_id = episode_data['episode_id']
#         model_name = episode_data['model_name']
        
#         # Create episode directory
#         episode_dir = self.output_dir / "episodes" / f"{model_name}_{episode_id:04d}"
#         episode_dir.mkdir(parents=True, exist_ok=True)
        
#         # Save frames as images
#         frame_files = []
#         for i, frame_data in enumerate(episode_data['frames']):
#             obs = frame_data['observation']
            
#             # Convert to image
#             if obs.max() <= 1.0:
#                 img_array = (obs * 255).astype(np.uint8)
#             else:
#                 img_array = obs.astype(np.uint8)
                
#             # Handle different image formats
#             if len(img_array.shape) == 3:
#                 if img_array.shape[2] == 1:
#                     img = Image.fromarray(img_array[:, :, 0], mode='L')
#                 else:
#                     img = Image.fromarray(img_array[:, :, :3], mode='RGB')
#             else:
#                 img = Image.fromarray(img_array, mode='L')
            
#             # Save image
#             img_filename = f"frame_{i:06d}.png"
#             img_path = episode_dir / img_filename
#             img.save(img_path)
            
#             frame_files.append({
#                 'frame_id': i,
#                 'filename': img_filename,
#                 'step': frame_data['step'],
#                 'timestamp': frame_data['timestamp']
#             })
        
#         # Save episode metadata
#         metadata = {
#             'episode_id': episode_data['episode_id'],
#             'model_name': episode_data['model_name'], 
#             'architecture': episode_data['architecture'],
#             'env_name': episode_data['env_name'],
#             'success': episode_data['success'],
#             'total_frames': len(frame_files),
#             'total_actions': len(episode_data['actions']),
#             'frames': frame_files,
#             'actions': episode_data['actions']
#         }
        
#         metadata_path = episode_dir / "episode_metadata.json"
#         with open(metadata_path, 'w') as f:
#             json.dump(metadata, f, indent=2)
        
#         print(f"💾 Episode saved: {episode_dir}")
#         return episode_dir
    
#     def create_vanp_samples(self, episode_data, tau_p=6, tau_f=20):
#         """
#         Tạo VANP training samples từ episode data
        
#         Args:
#             episode_data: Episode data
#             tau_p: Number of past frames (1.5s @ 4Hz)
#             tau_f: Number of future actions (5s @ 4Hz)
#         """
#         if episode_data is None:
#             return []
            
#         frames = episode_data['frames']
#         actions = episode_data['actions']
        
#         if len(frames) < tau_p + tau_f + 1:
#             print(f"⚠️  Episode too short for VANP samples: {len(frames)} frames")
#             return []
        
#         vanp_samples = []
        
#         # Sliding window để tạo samples
#         for t in range(tau_p, len(frames) - tau_f):
#             # Past visual observations (t-tau_p:t)
#             visual_history = []
#             for i in range(t - tau_p, t):
#                 if i < len(frames):
#                     visual_history.append({
#                         'frame_id': i,
#                         'step': frames[i]['step'],
#                         'timestamp': frames[i]['timestamp']
#                     })
            
#             # Future actions (t:t+tau_f)
#             future_actions = []
#             for i in range(t, min(t + tau_f, len(actions))):
#                 if i < len(actions):
#                     future_actions.append(actions[i]['action'])
            
#             # Goal image (frame at t+tau_f)
#             goal_frame_idx = min(t + tau_f, len(frames) - 1)
#             goal_frame = {
#                 'frame_id': goal_frame_idx,
#                 'step': frames[goal_frame_idx]['step'],
#                 'timestamp': frames[goal_frame_idx]['timestamp']
#             }
            
#             vanp_sample = {
#                 'sample_id': len(vanp_samples),
#                 'episode_id': episode_data['episode_id'],
#                 'model_name': episode_data['model_name'],
#                 'architecture': episode_data['architecture'],
#                 'current_step': t,
#                 'visual_history': visual_history,
#                 'future_actions': future_actions,
#                 'goal_frame': goal_frame
#             }
            
#             vanp_samples.append(vanp_sample)
        
#         print(f"🎯 Created {len(vanp_samples)} VANP samples from episode")
#         return vanp_samples
    
#     def collect_dataset(self, episodes_per_model=50, environments=None):
#         """
#         Collect complete VANP dataset từ tất cả trained models
        
#         Args:
#             episodes_per_model: Number of episodes per model
#             environments: List of environments to use
#         """
#         if environments is None:
#             environments = ['NormalTrain', 'FogTrain']
            
#         print(f"🚀 Starting VANP dataset collection")
#         print(f"   Episodes per model: {episodes_per_model}")
#         print(f"   Environments: {environments}")
#         print(f"   Models: {len(self.trained_models)}")
        
#         # Load all models
#         loaded_models = []
#         for model_info in self.trained_models:
#             if self.load_model(model_info):
#                 loaded_models.append(model_info)
        
#         print(f"✅ Loaded {len(loaded_models)} models successfully")
        
#         # Collect episodes
#         all_episodes = []
#         all_vanp_samples = []
#         episode_counter = 0
        
#         for model_info in loaded_models:
#             print(f"\n📊 Collecting from {model_info['name']} ({model_info['architecture']})")
            
#             for env_name in environments:
#                 print(f"   Environment: {env_name}")
                
#                 for episode_idx in range(episodes_per_model):
#                     episode_data = self.collect_episode_from_model(
#                         model_info, 
#                         env_name=env_name,
#                         episode_id=episode_counter,
#                         episode_length=500  # 2 minutes @ 4Hz
#                     )
                    
#                     if episode_data is not None:
#                         # Save episode
#                         episode_dir = self.save_episode(episode_data)
                        
#                         # Create VANP samples
#                         vanp_samples = self.create_vanp_samples(episode_data)
                        
#                         all_episodes.append(episode_data)
#                         all_vanp_samples.extend(vanp_samples)
#                         episode_counter += 1
                    
#                     # Progress update
#                     if episode_counter % 10 == 0:
#                         print(f"   Progress: {episode_counter} episodes collected")
        
#         # Save VANP samples dataset
#         self.save_vanp_dataset(all_vanp_samples)
        
#         # Summary
#         print(f"\n🎉 Dataset collection completed!")
#         print(f"   Total episodes: {len(all_episodes)}")
#         print(f"   Total VANP samples: {len(all_vanp_samples)}")
#         print(f"   Dataset saved to: {self.output_dir}")
        
#         return all_episodes, all_vanp_samples
    
#     def save_vanp_dataset(self, vanp_samples):
#         """Save VANP samples dataset"""
#         if not vanp_samples:
#             return
            
#         # Save as JSON Lines for efficiency
#         dataset_path = self.output_dir / "vanp_samples.jsonl"
        
#         with open(dataset_path, 'w') as f:
#             for sample in vanp_samples:
#                 f.write(json.dumps(sample) + '\n')
        
#         # Save summary statistics
#         stats = {
#             'total_samples': len(vanp_samples),
#             'models': list(set(s['model_name'] for s in vanp_samples)),
#             'architectures': list(set(s['architecture'] for s in vanp_samples)),
#             'episodes': list(set(s['episode_id'] for s in vanp_samples))
#         }
        
#         stats_path = self.output_dir / "dataset_stats.json"
#         with open(stats_path, 'w') as f:
#             json.dump(stats, f, indent=2)
        
#         print(f"💾 VANP dataset saved:")
#         print(f"   Samples: {dataset_path}")
#         print(f"   Statistics: {stats_path}")


# def main():
#     parser = argparse.ArgumentParser(description="VANP Data Collector từ Trained Models")
#     parser.add_argument("--output-dir", type=str, default="./vanp_dataset",
#                        help="Output directory cho VANP dataset")
#     parser.add_argument("--models-dir", type=str, default="./results",
#                        help="Directory chứa trained models")
#     parser.add_argument("--builds-dir", type=str, default="./Builds", 
#                        help="Directory chứa Unity builds")
#     parser.add_argument("--episodes-per-model", type=int, default=20,
#                        help="Number of episodes per model")
#     parser.add_argument("--environments", nargs='+', default=["NormalTrain"],
#                        help="Environments to collect from")
    
#     args = parser.parse_args()
    
#     # Initialize collector
#     collector = VANPDataCollector(
#         output_dir=args.output_dir,
#         models_dir=args.models_dir,
#         builds_dir=args.builds_dir
#     )
    
#     # Collect dataset
#     episodes, vanp_samples = collector.collect_dataset(
#         episodes_per_model=args.episodes_per_model,
#         environments=args.environments
#     )
    
#     if vanp_samples:
#         print(f"\n📈 Ready for VANP training!")
#         print(f"   Use dataset: {args.output_dir}/vanp_samples.jsonl")
#         print(f"   Images in: {args.output_dir}/episodes/")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
VANP Data Collector từ Trained Models
Sử dụng các trained models để synthesize data cho VANP training
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
import onnxruntime as ort


class VANPDataCollector:
    def __init__(self, 
                 output_dir="./vanp_dataset", 
                 models_dir="./results",
                 builds_dir="./Builds"):
        self.output_dir = Path(output_dir)
        self.models_dir = Path(models_dir)
        self.builds_dir = Path(builds_dir)
        
        # Setup directories
        self.setup_directories()
        
        # Find trained models
        self.trained_models = self.discover_trained_models()
        
        # Initialize ONNX sessions
        self.model_sessions = {}
        
        print(f"🎯 VANP Data Collector initialized")
        print(f"   Output: {self.output_dir}")
        print(f"   Found {len(self.trained_models)} trained models")
        
    def setup_directories(self):
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "episodes").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "vanp_samples").mkdir(parents=True, exist_ok=True)
        
    def discover_trained_models(self):
        models = []
        
        # Find ONNX models
        for onnx_file in self.models_dir.rglob("*.onnx"):
            model_info = {
                'path': str(onnx_file),
                'type': 'onnx',
                'name': onnx_file.stem,
                'architecture': self.extract_architecture_from_path(onnx_file)
            }
            models.append(model_info)
            
        # Find PyTorch models
        for pt_file in self.models_dir.rglob("*.pt"):
            model_info = {
                'path': str(pt_file),
                'type': 'pytorch',  
                'name': pt_file.stem,
                'architecture': self.extract_architecture_from_path(pt_file)
            }
            models.append(model_info)
            
        print(f"📊 Discovered models:")
        for model in models:
            print(f"   {model['name']} ({model['architecture']}) - {model['type']}")
            
        return models
    
    def extract_architecture_from_path(self, model_path):
        path_str = str(model_path).lower()
        
        if 'dino' in path_str:
            if 'resnet' in path_str:
                return 'dino_resnet'
            else:
                return 'dino'
        elif 'nature' in path_str:
            return 'nature_cnn'
        elif 'resnet' in path_str:
            return 'resnet'
        elif 'simple' in path_str:
            return 'simple'
        else:
            return 'unknown'
    
    def load_model(self, model_info):
        model_path = model_info['path']
        
        if model_info['type'] == 'onnx':
            try:
                session = ort.InferenceSession(
                    model_path, 
                    providers=['CPUExecutionProvider']
                )
                self.model_sessions[model_info['name']] = session
                print(f"✅ Loaded ONNX model: {model_info['name']}")
                return True
            except Exception as e:
                print(f"❌ Failed to load ONNX model {model_info['name']}: {e}")
                return False
        else:
            print(f"⚠️  PyTorch model loading not implemented: {model_info['name']}")
            return False
    
    def collect_episode_from_model(self, model_info, env, env_name="NormalTrain", 
                                   episode_length=1000, episode_id=0):
        """
        Collect một episode từ một trained model (sử dụng env đã mở sẵn)
        """
        print(f"🎮 Collecting episode {episode_id} từ {model_info['name']}")
        
        try:
            env.reset()
            behavior_names = list(env.behavior_specs.keys())
            behavior_name = behavior_names[0]
            spec = env.behavior_specs[behavior_name]
            
            episode_data = {
                'episode_id': episode_id,
                'model_name': model_info['name'],
                'architecture': model_info['architecture'],
                'env_name': env_name,
                'frames': [],
                'actions': [],
                'rewards': [],
                'dones': [],
                'timestamps': [],
                'success': False
            }
            
            step_count = 0
            start_time = time.time()
            
            while step_count < episode_length:
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                
                if len(decision_steps) == 0 and len(terminal_steps) == 0:
                    break
                    
                if len(decision_steps) > 0:
                    visual_obs = decision_steps.obs[0]  # First visual observation
                    
                    # Save frame
                    for agent_id, obs in zip(decision_steps.agent_id, visual_obs):
                        frame_data = {
                            'step': step_count,
                            'agent_id': int(agent_id),
                            'timestamp': time.time() - start_time,
                            'observation': obs
                        }
                        episode_data['frames'].append(frame_data)
                    
                    # Get action từ model
                    action = self.get_model_action(model_info, visual_obs)
                    
                    action_data = {
                        'step': step_count,
                        'action': action.continuous.tolist() if hasattr(action, 'continuous') else action,
                        'timestamp': time.time() - start_time
                    }
                    episode_data['actions'].append(action_data)
                    
                    env.set_actions(behavior_name, action)
                    
                if len(terminal_steps) > 0:
                    episode_data['success'] = True
                    break
                    
                env.step()
                step_count += 1
                
                if step_count % 100 == 0:
                    print(f"   Step {step_count}/{episode_length}")
            
            print(f"✅ Episode collected: {len(episode_data['frames'])} frames, {len(episode_data['actions'])} actions")
            return episode_data
            
        except Exception as e:
            print(f"❌ Episode collection failed: {e}")
            return None
    
    def get_model_action(self, model_info, visual_obs):
        model_name = model_info['name']
        
        if model_name not in self.model_sessions:
            return self.get_random_action(visual_obs.shape[0])
        
        try:
            session = self.model_sessions[model_name]
            
            if len(visual_obs.shape) == 4:  # BHWC
                input_obs = visual_obs.transpose(0, 3, 1, 2).astype(np.float32)
            else:
                input_obs = visual_obs.astype(np.float32)
                
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: input_obs})
            
            action_output = None
            for output in outputs:
                if len(output.shape) == 2 and output.shape[0] == visual_obs.shape[0]:
                    action_output = output
                    break
            
            if action_output is not None:
                continuous_actions = action_output.astype(np.float32)
                discrete_actions = np.empty((visual_obs.shape[0], 0), dtype=np.int32)
                return ActionTuple(continuous_actions, discrete_actions)
            else:
                return self.get_random_action(visual_obs.shape[0])
                
        except Exception as e:
            print(f"⚠️  Model inference failed: {e}")
            return self.get_random_action(visual_obs.shape[0])
    
    def get_random_action(self, batch_size):
        continuous_actions = np.random.randn(batch_size, 3).astype(np.float32)
        discrete_actions = np.empty((batch_size, 0), dtype=np.int32)
        return ActionTuple(continuous_actions, discrete_actions)
    
    def save_episode(self, episode_data):
        if episode_data is None:
            return None
            
        episode_id = episode_data['episode_id']
        model_name = episode_data['model_name']
        
        episode_dir = self.output_dir / "episodes" / f"{model_name}_{episode_id:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        
        frame_files = []
        for i, frame_data in enumerate(episode_data['frames']):
            obs = frame_data['observation']
            
            if obs.max() <= 1.0:
                img_array = (obs * 255).astype(np.uint8)
            else:
                img_array = obs.astype(np.uint8)
                
            if len(img_array.shape) == 3:
                if img_array.shape[2] == 1:
                    img = Image.fromarray(img_array[:, :, 0], mode='L')
                else:
                    img = Image.fromarray(img_array[:, :, :3], mode='RGB')
            else:
                img = Image.fromarray(img_array, mode='L')
            
            img_filename = f"frame_{i:06d}.png"
            img_path = episode_dir / img_filename
            img.save(img_path)
            
            frame_files.append({
                'frame_id': i,
                'filename': img_filename,
                'step': frame_data['step'],
                'timestamp': frame_data['timestamp']
            })
        
        metadata = {
            'episode_id': episode_data['episode_id'],
            'model_name': episode_data['model_name'], 
            'architecture': episode_data['architecture'],
            'env_name': episode_data['env_name'],
            'success': episode_data['success'],
            'total_frames': len(frame_files),
            'total_actions': len(episode_data['actions']),
            'frames': frame_files,
            'actions': episode_data['actions']
        }
        
        metadata_path = episode_dir / "episode_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Episode saved: {episode_dir}")
        return episode_dir
    
    def create_vanp_samples(self, episode_data, tau_p=3, tau_f=12):
        if episode_data is None:
            return []
            
        frames = episode_data['frames']
        actions = episode_data['actions']
        
        if len(frames) < tau_p + tau_f + 1:
            print(f"⚠️  Episode too short for VANP samples: {len(frames)} frames")
            return []
        
        vanp_samples = []
        
        for t in range(tau_p, len(frames) - tau_f):
            visual_history = []
            for i in range(t - tau_p, t):
                if i < len(frames):
                    visual_history.append({
                        'frame_id': i,
                        'step': frames[i]['step'],
                        'timestamp': frames[i]['timestamp']
                    })
            
            future_actions = []
            for i in range(t, min(t + tau_f, len(actions))):
                if i < len(actions):
                    future_actions.append(actions[i]['action'])
            
            goal_frame_idx = min(t + tau_f, len(frames) - 1)
            goal_frame = {
                'frame_id': goal_frame_idx,
                'step': frames[goal_frame_idx]['step'],
                'timestamp': frames[goal_frame_idx]['timestamp']
            }
            
            vanp_sample = {
                'sample_id': len(vanp_samples),
                'episode_id': episode_data['episode_id'],
                'model_name': episode_data['model_name'],
                'architecture': episode_data['architecture'],
                'current_step': t,
                'visual_history': visual_history,
                'future_actions': future_actions,
                'goal_frame': goal_frame
            }
            
            vanp_samples.append(vanp_sample)
        
        print(f"🎯 Created {len(vanp_samples)} VANP samples from episode")
        return vanp_samples
    
    def collect_dataset(self, episodes_per_model=50, environments=None):
        if environments is None:
            environments = ['NormalTrain', 'FogTrain']
            
        print(f"🚀 Starting VANP dataset collection")
        print(f"   Episodes per model: {episodes_per_model}")
        print(f"   Environments: {environments}")
        print(f"   Models: {len(self.trained_models)}")
        
        loaded_models = []
        for model_info in self.trained_models:
            if self.load_model(model_info):
                loaded_models.append(model_info)
        
        print(f"✅ Loaded {len(loaded_models)} models successfully")
        
        all_episodes = []
        all_vanp_samples = []
        episode_counter = 0
        
        for model_info in loaded_models:
            print(f"\n📊 Collecting from {model_info['name']} ({model_info['architecture']})")
            
            for env_name in environments:
                env_path = self.builds_dir / env_name / "2D go to target v1.exe"
                if not env_path.exists():
                    print(f"❌ Environment not found: {env_path}")
                    continue

                # 🚀 mở UnityEnvironment 1 lần duy nhất cho cả (model, env)
                env = UnityEnvironment(
                    file_name=str(env_path),
                    worker_id=0,
                    seed=42,
                    side_channels=[]
                )
                env.reset()

                for episode_idx in range(episodes_per_model):
                    episode_data = self.collect_episode_from_model(
                        model_info, 
                        env=env,
                        env_name=env_name,
                        episode_id=episode_counter,
                        episode_length=500
                    )
                    
                    if episode_data is not None:
                        self.save_episode(episode_data)
                        vanp_samples = self.create_vanp_samples(episode_data)
                        all_episodes.append(episode_data)
                        all_vanp_samples.extend(vanp_samples)
                        episode_counter += 1
                    
                    if episode_counter % 10 == 0:
                        print(f"   Progress: {episode_counter} episodes collected")

                env.close()  # ✅ chỉ close sau khi xong hết episodes cho env này
        
        self.save_vanp_dataset(all_vanp_samples)
        
        print(f"\n🎉 Dataset collection completed!")
        print(f"   Total episodes: {len(all_episodes)}")
        print(f"   Total VANP samples: {len(all_vanp_samples)}")
        print(f"   Dataset saved to: {self.output_dir}")
        
        return all_episodes, all_vanp_samples
    
    def save_vanp_dataset(self, vanp_samples):
        if not vanp_samples:
            return
            
        dataset_path = self.output_dir / "vanp_samples.jsonl"
        
        with open(dataset_path, 'w') as f:
            for sample in vanp_samples:
                f.write(json.dumps(sample) + '\n')
        
        stats = {
            'total_samples': len(vanp_samples),
            'models': list(set(s['model_name'] for s in vanp_samples)),
            'architectures': list(set(s['architecture'] for s in vanp_samples)),
            'episodes': list(set(s['episode_id'] for s in vanp_samples))
        }
        
        stats_path = self.output_dir / "dataset_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"💾 VANP dataset saved:")
        print(f"   Samples: {dataset_path}")
        print(f"   Statistics: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="VANP Data Collector từ Trained Models")
    parser.add_argument("--output-dir", type=str, default="./vanp_dataset",
                       help="Output directory cho VANP dataset")
    parser.add_argument("--models-dir", type=str, default="./results",
                       help="Directory chứa trained models")
    parser.add_argument("--builds-dir", type=str, default="./Builds", 
                       help="Directory chứa Unity builds")
    parser.add_argument("--episodes-per-model", type=int, default=20,
                       help="Number of episodes per model")
    parser.add_argument("--environments", nargs='+', default=["NormalTrain"],
                       help="Environments to collect from")
    
    args = parser.parse_args()
    
    collector = VANPDataCollector(
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        builds_dir=args.builds_dir
    )
    
    episodes, vanp_samples = collector.collect_dataset(
        episodes_per_model=args.episodes_per_model,
        environments=args.environments
    )
    
    if vanp_samples:
        print(f"\n📈 Ready for VANP training!")
        print(f"   Use dataset: {args.output_dir}/vanp_samples.jsonl")
        print(f"   Images in: {args.output_dir}/episodes/")


if __name__ == "__main__":
    main()
