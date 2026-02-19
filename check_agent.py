"""
Quick script to check if agent checkpoints exist and list available models.
"""

import os
from pathlib import Path

def check_agents():
    """Check for available agent checkpoints."""
    checkpoint_dir = Path("models/checkpoints")
    
    print("="*60)
    print("CHECKING FOR TRAINED AGENTS")
    print("="*60)
    
    if not checkpoint_dir.exists():
        print(f"\n❌ Checkpoint directory not found: {checkpoint_dir}")
        print("\nYou need to train an agent first!")
        print("\nTo train an agent, run:")
        print("  python main.py --mode train --config configs/config.yaml")
        return []
    
    # Find all checkpoint files
    checkpoints = []
    
    # Look for .zip files (Stable-Baselines3 format)
    zip_files = list(checkpoint_dir.glob("*.zip"))
    for f in zip_files:
        checkpoints.append(str(f))
    
    # Look for directories that might contain checkpoints
    for item in checkpoint_dir.iterdir():
        if item.is_dir():
            # Check if it's a valid checkpoint directory
            if (item / "policy.pth").exists() or (item / "model.zip").exists():
                checkpoints.append(str(item))
    
    if checkpoints:
        print(f"\n✓ Found {len(checkpoints)} checkpoint(s):\n")
        for i, checkpoint in enumerate(checkpoints, 1):
            size = Path(checkpoint).stat().st_size / (1024 * 1024)  # MB
            print(f"  {i}. {checkpoint} ({size:.2f} MB)")
        
        print("\nTo evaluate an agent, use:")
        print(f"  python main.py --mode eval --checkpoint {checkpoints[0]}")
        print("\nTo analyze results:")
        print(f"  python analyze_results.py --agent {checkpoints[0]} --config configs/config.yaml --compare")
    else:
        print("\n❌ No agent checkpoints found!")
        print("\nTo train an agent, run:")
        print("  python main.py --mode train --config configs/config.yaml")
        print("\nThis will create checkpoints in:", checkpoint_dir)
    
    print("="*60)
    
    return checkpoints

if __name__ == "__main__":
    check_agents()
