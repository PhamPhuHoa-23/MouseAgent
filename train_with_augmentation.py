import subprocess
import os
import time
from pathlib import Path
import glob
import replace
import pandas as pd

def get_next_run_number(base_name, results_dir="./results"):
    """Get the next run number for a given base name by checking existing results."""
    os.makedirs(results_dir, exist_ok=True)
    pattern = os.path.join(results_dir, f"{base_name}_*")
    existing_runs = glob.glob(pattern)
    if not existing_runs:
        return 1
    run_numbers = []
    for run_path in existing_runs:
        try:
            run_num = int(run_path.split('_')[-1])
            run_numbers.append(run_num)
        except (ValueError, IndexError):
            continue
    return max(run_numbers) + 1 if run_numbers else 1

def summarize_log(log_path: str):
    """
    Reads the Unity log at log_path, then prints:
      • Overall success rate (%)
      • Success rate per trial type
      • Max target distance (units)
    """
    df = pd.read_csv(
        log_path,
        sep=r'\s+',
        comment='#',
        header=None,
        names=['SessionTime','EventType','x','y','z','r','extra'],
        usecols=[0,1,2,4,5],
        engine='python'
    )
    df = df[df.EventType.isin(['n','t','s','h','f'])].reset_index(drop=True)

    new_trial_idxs = df.index[df.EventType=='n'].tolist()
    trial_type_idx = df.index[df.EventType=='s'].tolist()

    successes = []
    by_type = {}
    distances = []

    for ti, start_idx in enumerate(new_trial_idxs):
        end_idx = new_trial_idxs[ti+1] if ti+1 < len(new_trial_idxs) else len(df)
        trial = df.iloc[start_idx:end_idx]

        ttype = int(trial.loc[trial.EventType=='s','x'].iat[0])

        trow = trial[trial.EventType=='t']
        if trow.empty:
            continue
        dx, dz = float(trow.x.iat[0]), float(trial.loc[trow.index,'z'].iat[0])
        distances.append(dx)

        hit = 1 if ('h' in trial.EventType.values) else 0
        successes.append(hit)
        by_type.setdefault(ttype, []).append(hit)

    if successes:
        overall = sum(successes)/len(successes)*100
        print(f"\n=== EVALUATION RESULTS ===")
        print(f"Overall success rate: {overall:.1f}% ({sum(successes)}/{len(successes)})")
        for ttype, hits in by_type.items():
            rate = sum(hits)/len(hits)*100
            print(f"  * Trial type {ttype}: {rate:.1f}% ({sum(hits)}/{len(hits)})")
    if distances:
        print(f"Max target distance reached: {max(distances):.3f}/5.00")
    print("==========================\n")

def setup_checkpoint(checkpoint_path, target_dir="./"):
    """Copy checkpoint to accessible location if needed"""
    if not checkpoint_path:
        return None
        
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.exists():
        return str(checkpoint_file)
    
    # Try to find in common directories
    possible_paths = [
        Path("./checkpoints_robust") / checkpoint_path,
        Path("./checkpoints_robust_resnet_mini") / checkpoint_path,
        Path(".") / checkpoint_path
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    print(f"Warning: Checkpoint not found: {checkpoint_path}")
    return None

def train_solo(run_id, env_path, config_path, total_runs=5, log_name=None, checkpoint_path=None):
    next_run = get_next_run_number(run_id)
    run_id_list = []
    
    # Setup checkpoint
    if checkpoint_path:
        checkpoint_path = setup_checkpoint(checkpoint_path)
        if checkpoint_path:
            print(f"Will use checkpoint: {checkpoint_path}")
    
    for i in range(total_runs):
        current = f"{run_id}_{next_run + i}"
        print(f"Starting training: {current}")
        
        if checkpoint_path:
            print(f"Using checkpoint: {Path(checkpoint_path).name}")
        
        fn = f"{(log_name if log_name else run_id)}_{next_run + i}_train.txt"       
        sa = os.path.join(env_path,
                    "2D go to target v1_Data",
                    "StreamingAssets",
                    "currentLog.txt")
    
        with open(sa, "w") as f:
            f.write(fn)

        time.sleep(1)
        cmd = [
            "mlagents-learn",
            config_path,
            "--env", str(Path(env_path) / "2D go to target v1.exe"),
            "--run-id", current,
            "--force",
            "--env-args", "--screen-width=155", "--screen-height=86",
            "--torch-device=cuda",
            "--time-scale=5",
            # "--no-graphics"
        ]
        subprocess.run(cmd, check=True)
        print(f"Completed training: {current}")
        time.sleep(5)
        run_id_list.append(current)
    return run_id_list

def train_multiple_networks(networks, env_path, runs_per_network=2, log_name=None, env='NormalTrain', checkpoint_path=None):
    run_id_list2 = []
    
    # Validate checkpoint once
    if checkpoint_path:
        checkpoint_path = setup_checkpoint(checkpoint_path)
        if not checkpoint_path:
            print("Proceeding without checkpoint...")
    
    for network in networks:
        print(f"\n=== Training network: {network} ===")
        
        if network == "fully_connected":
            config_path = "./Config/fc.yaml"
        elif network == "simple":
            config_path = "./Config/simple.yaml"
        elif network == "resnet":
            config_path = "./Config/resnet.yaml"
        elif network == "resnet_aug":
            config_path = "./Config/resnet_aug.yaml"
            # Replace encoder with augmentation version
            replace.replace_nature_visual_encoder(
                "C:/Users/admin/miniconda3/envs/mouse_dinov3_py38/Lib/site-packages/mlagents/trainers/torch/encoders.py", 
                "./Encoders/resnet_aug.py"
            )
        else:
            config_path = f"./Config/{network}.yaml"
            if network != "nature_cnn":
                replace.replace_nature_visual_encoder(
                    "C:/Users/admin/miniconda3/envs/mouse_dinov3_py38/Lib/site-packages/mlagents/trainers/torch/encoders.py", 
                    "./Encoders/" + network + ".py"
                )

        run_ids = train_solo(
            run_id=f"{network}_{env}",
            env_path=env_path,
            config_path=config_path,
            total_runs=runs_per_network,
            log_name=log_name,
            checkpoint_path=checkpoint_path
        )
        run_id_list2.extend(run_ids)
    return run_id_list2

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train multiple networks on MouseVsAI")
    parser.add_argument("--env", type=str, default="NormalTrain",
                        help="Folder name under Builds/ to use as env")
    parser.add_argument("--runs-per-network", type=int, default=1,
                        help="How many runs per network variant")
    parser.add_argument("--networks", type=str, default="nature_cnn,simple,resnet",
                        help="Comma-separated list of network names")
    parser.add_argument("--log-name", type=str, default=None,
                        help="Optional prefix for all log files")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file for backbone initialization")
    args = parser.parse_args()

    env_folder = f"./Builds/{args.env}"
    nets = [n.strip() for n in args.networks.split(",")]
    
    run_ids = train_multiple_networks(
        nets, env_folder, args.runs_per_network, args.log_name, args.env, args.checkpoint
    )

    # Summarize each run
    logs_dir = Path("./logfiles")
    logs_dir.mkdir(exist_ok=True)
    for rid in run_ids:
        summary_file = logs_dir / f"{(args.log_name if args.log_name else rid)}_train.txt"
        print(f"\n=== Summary for {rid} ===")
        summarize_log(str(summary_file))
        
    print(f"\n=== TRAINING COMPLETED ===")
    print(f"Trained {len(nets)} networks with {args.runs_per_network} runs each")
    if args.checkpoint:
        print(f"Used checkpoint: {Path(args.checkpoint).name}")
    print("="*30)