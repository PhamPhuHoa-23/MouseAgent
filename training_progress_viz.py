#!/usr/bin/env python3
"""
Training Progress Visualization
Vẽ line chart với mean reward và std shaded area từ training logs
"""

import matplotlib.pyplot as plt
import numpy as np
import re
import argparse
from pathlib import Path
import pandas as pd

def parse_training_log(log_content):
    """
    Parse training log để extract step, time, mean reward, std reward
    """
    # Pattern để match log lines
    pattern = r'\[INFO\] My Behavior\. Step: (\d+)\. Time Elapsed: ([\d.]+) s\. Mean Reward: ([-\d.]+)\. Std of Reward: ([\d.]+)\. Training\.'
    
    steps = []
    times = []
    mean_rewards = []
    std_rewards = []
    
    for line in log_content.split('\n'):
        match = re.search(pattern, line)
        if match:
            step = int(match.group(1))
            time_elapsed = float(match.group(2))
            mean_reward = float(match.group(3))
            std_reward = float(match.group(4))
            
            steps.append(step)
            times.append(time_elapsed)
            mean_rewards.append(mean_reward)
            std_rewards.append(std_reward)
    
    return {
        'steps': np.array(steps),
        'times': np.array(times),
        'mean_rewards': np.array(mean_rewards),
        'std_rewards': np.array(std_rewards)
    }

def create_sample_data():
    """Tạo sample data từ log trong câu hỏi"""
    sample_log = """
[INFO] My Behavior. Step: 1000. Time Elapsed: 24.116 s. Mean Reward: -33.947. Std of Reward: 61.261. Training.
[INFO] My Behavior. Step: 2000. Time Elapsed: 41.925 s. Mean Reward: -66.923. Std of Reward: 75.967. Training.
[INFO] My Behavior. Step: 3000. Time Elapsed: 58.423 s. Mean Reward: -62.083. Std of Reward: 69.644. Training.
[INFO] My Behavior. Step: 4000. Time Elapsed: 75.036 s. Mean Reward: -95.100. Std of Reward: 88.161. Training.
[INFO] My Behavior. Step: 5000. Time Elapsed: 92.440 s. Mean Reward: -16.630. Std of Reward: 43.379. Training.
[INFO] My Behavior. Step: 6000. Time Elapsed: 110.179 s. Mean Reward: -41.438. Std of Reward: 45.058. Training.
"""
    return parse_training_log(sample_log)

def smooth_data(data, window_size=3):
    """Smooth data using moving average để chart đẹp hơn"""
    if len(data) < window_size:
        return data
    
    smoothed = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    # Pad the beginning to maintain length
    padding = np.repeat(smoothed[0], window_size-1)
    return np.concatenate([padding, smoothed])

def visualize_training_progress(data, smooth=True, save_path=None, title="Training Progress"):
    """
    Visualize training progress với mean line và std shaded area
    """
    steps = data['steps']
    mean_rewards = data['mean_rewards']
    std_rewards = data['std_rewards']
    times = data['times']
    
    # Apply smoothing if requested
    if smooth:
        mean_rewards = smooth_data(mean_rewards)
        std_rewards = smooth_data(std_rewards)
    
    # Create figure với 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # ===================
    # Plot 1: Reward vs Steps
    # ===================
    # Main mean line
    line1 = ax1.plot(steps, mean_rewards, 'b-', linewidth=2.5, label='Mean Reward', alpha=0.8)
    color1 = line1[0].get_color()
    
    # Shaded area for std (mean ± std)
    upper_bound = mean_rewards + std_rewards
    lower_bound = mean_rewards - std_rewards
    ax1.fill_between(steps, lower_bound, upper_bound, 
                     color=color1, alpha=0.2, label='±1 Std Dev')
    
    # Styling cho plot 1
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('Reward Progress vs Training Steps', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Add some statistics text
    final_mean = mean_rewards[-1]
    final_std = std_rewards[-1]
    improvement = mean_rewards[-1] - mean_rewards[0]
    ax1.text(0.02, 0.98, f'Final: {final_mean:.2f} ± {final_std:.2f}\nImprovement: {improvement:+.2f}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ===================
    # Plot 2: Reward vs Time
    # ===================
    # Main mean line  
    line2 = ax2.plot(times, mean_rewards, 'g-', linewidth=2.5, label='Mean Reward', alpha=0.8)
    color2 = line2[0].get_color()
    
    # Shaded area for std
    ax2.fill_between(times, mean_rewards - std_rewards, mean_rewards + std_rewards,
                     color=color2, alpha=0.2, label='±1 Std Dev')
    
    # Styling cho plot 2
    ax2.set_xlabel('Time Elapsed (seconds)', fontsize=12)
    ax2.set_ylabel('Reward', fontsize=12)
    ax2.set_title('Reward Progress vs Training Time', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Add time efficiency info
    total_time = times[-1]
    steps_per_second = steps[-1] / total_time
    ax2.text(0.02, 0.98, f'Total Time: {total_time:.1f}s\nSpeed: {steps_per_second:.1f} steps/s', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    plt.show()
    return fig

def visualize_from_file(log_file_path, smooth=True, save_path=None):
    """Load và visualize từ log file"""
    try:
        with open(log_file_path, 'r') as f:
            log_content = f.read()
        
        data = parse_training_log(log_content)
        
        if len(data['steps']) == 0:
            print("No training data found in log file!")
            return
        
        title = f"Training Progress - {Path(log_file_path).name}"
        return visualize_training_progress(data, smooth=smooth, save_path=save_path, title=title)
        
    except FileNotFoundError:
        print(f"File not found: {log_file_path}")
    except Exception as e:
        print(f"Error reading file: {e}")

def create_advanced_visualization(data, save_path=None):
    """
    Tạo advanced visualization với multiple metrics
    """
    steps = data['steps']
    mean_rewards = data['mean_rewards']
    std_rewards = data['std_rewards']
    times = data['times']
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
    
    # ===================
    # Main Reward Plot (spans 2 columns)
    # ===================
    ax_main = fig.add_subplot(gs[0, :])
    
    # Smooth data
    smooth_mean = smooth_data(mean_rewards, window_size=3)
    smooth_std = smooth_data(std_rewards, window_size=3)
    
    # Main line with gradient effect
    line = ax_main.plot(steps, smooth_mean, 'b-', linewidth=3, label='Mean Reward', alpha=0.9)
    color = line[0].get_color()
    
    # Multiple std levels
    ax_main.fill_between(steps, smooth_mean - smooth_std, smooth_mean + smooth_std,
                         color=color, alpha=0.25, label='±1σ')
    ax_main.fill_between(steps, smooth_mean - smooth_std*0.5, smooth_mean + smooth_std*0.5,
                         color=color, alpha=0.4, label='±0.5σ')
    
    ax_main.set_title('Training Progress: Mean Reward ± Standard Deviation', fontsize=16, fontweight='bold')
    ax_main.set_xlabel('Training Steps', fontsize=12)
    ax_main.set_ylabel('Reward', fontsize=12)
    ax_main.grid(True, alpha=0.3)
    ax_main.legend()
    
    # ===================
    # Standard Deviation Plot
    # ===================
    ax_std = fig.add_subplot(gs[1, 0])
    ax_std.plot(steps, std_rewards, 'r-', linewidth=2, alpha=0.7)
    ax_std.fill_between(steps, 0, std_rewards, alpha=0.3, color='red')
    ax_std.set_title('Reward Standard Deviation', fontsize=12)
    ax_std.set_xlabel('Steps')
    ax_std.set_ylabel('Std Dev')
    ax_std.grid(True, alpha=0.3)
    
    # ===================
    # Learning Rate (Steps/Time)
    # ===================
    ax_rate = fig.add_subplot(gs[1, 1])
    if len(times) > 1:
        learning_rate = np.diff(steps) / np.diff(times)
        rate_steps = steps[1:]
        ax_rate.plot(rate_steps, learning_rate, 'g-', linewidth=2, alpha=0.7)
        ax_rate.fill_between(rate_steps, 0, learning_rate, alpha=0.3, color='green')
    ax_rate.set_title('Training Speed (Steps/Second)', fontsize=12)
    ax_rate.set_xlabel('Steps')
    ax_rate.set_ylabel('Steps/s')
    ax_rate.grid(True, alpha=0.3)
    
    # ===================
    # Cumulative Statistics
    # ===================
    ax_stats = fig.add_subplot(gs[2, :])
    
    # Running average
    running_mean = np.cumsum(mean_rewards) / np.arange(1, len(mean_rewards) + 1)
    ax_stats.plot(steps, running_mean, 'purple', linewidth=2, label='Cumulative Mean')
    
    # Best reward so far
    best_so_far = np.maximum.accumulate(mean_rewards)
    ax_stats.plot(steps, best_so_far, 'orange', linewidth=2, label='Best So Far', linestyle='--')
    
    ax_stats.set_title('Cumulative Statistics', fontsize=12)
    ax_stats.set_xlabel('Steps')
    ax_stats.set_ylabel('Reward')
    ax_stats.legend()
    ax_stats.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved advanced plot to: {save_path}")
    
    plt.show()
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize Training Progress')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Path to training log file')
    parser.add_argument('--sample', action='store_true',
                        help='Use sample data for demo')
    parser.add_argument('--smooth', action='store_true', default=True,
                        help='Apply smoothing to data')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save plot')
    parser.add_argument('--advanced', action='store_true',
                        help='Create advanced visualization with multiple metrics')
    
    args = parser.parse_args()
    
    if args.sample or args.log_file is None:
        print("Using sample data...")
        data = create_sample_data()
        title = "Training Progress - Sample Data"
        
        if args.advanced:
            create_advanced_visualization(data, save_path=args.save)
        else:
            visualize_training_progress(data, smooth=args.smooth, save_path=args.save, title=title)
    
    else:
        if args.advanced:
            # Load data first
            with open(args.log_file, 'r') as f:
                log_content = f.read()
            data = parse_training_log(log_content)
            create_advanced_visualization(data, save_path=args.save)
        else:
            visualize_from_file(args.log_file, smooth=args.smooth, save_path=args.save)

if __name__ == "__main__":
    main()