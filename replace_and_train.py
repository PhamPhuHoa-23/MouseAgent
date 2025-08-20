#!/usr/bin/env python3
"""
Replace ML-Agents encoder with DINOv3 and start training
"""

import subprocess
import os
import shutil
from pathlib import Path

def replace_encoder():
    """Replace ML-Agents NatureVisualEncoder with DINOv3"""
    
    print("🔄 Replacing ML-Agents encoder with FULL DINOv3...")
    
    # Find ML-Agents encoders.py
    import mlagents
    mlagents_path = Path(mlagents.__file__).parent
    encoders_path = mlagents_path / "trainers" / "torch" / "encoders.py"
    
    # Backup original
    backup_path = str(encoders_path) + ".backup_original"
    if not Path(backup_path).exists():
        print(f"📋 Backing up original: {backup_path}")
        shutil.copy2(encoders_path, backup_path)
    
    # Copy our DINOv3 encoder
    our_encoder = "Encoders/dinov3_full_029.py"
    print(f"📋 Copying DINOv3 encoder: {our_encoder} → {encoders_path}")
    shutil.copy2(our_encoder, encoders_path)
    
    print("✅ Encoder replacement complete!")
    return True

def start_training():
    """Start DINOv3 training with ML-Agents"""
    
    print("🚀 Starting DINOv3 training...")
    
    cmd = [
        "mlagents-learn",
        "Config/dinov3_train.yaml",
        "--env", "Builds/NormalTrain/2D go to target v1.exe", 
        "--run-id", "dinov3_full_train",
        "--force",
        "--env-args", "--screen-width=155", "--screen-height=86"
    ]
    
    print(f"🎯 Command: {' '.join(cmd)}")
    print("📊 Expected: FULL DINOv3 (21.7M params) + User Architecture")
    print("🎮 Training will start...")
    
    return subprocess.run(cmd)

def main():
    print("🎯 DINOv3 Training Setup")
    print("="*50)
    
    # Step 1: Replace encoder
    if not replace_encoder():
        print("❌ Failed to replace encoder")
        return
    
    # Step 2: Start training  
    print("⏳ Starting training in 3 seconds...")
    import time
    time.sleep(3)
    
    result = start_training()
    
    if result.returncode == 0:
        print("✅ Training completed successfully!")
    else:
        print(f"❌ Training failed with code: {result.returncode}")

if __name__ == "__main__":
    main()
