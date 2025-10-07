# This script converts the rollout .pkl files into .h5 format for the VAE module, vae_testerencoder.py to match Jingwen's implementation.

import pickle as pkl
import h5py
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm


def convert_rollout_to_h5(pkl_path, h5_output_dir):
    """
    Convert a single rollout pkl file to h5 format.
    
    Args:
        pkl_path: Path to the .pkl rollout file
        h5_output_dir: Directory to save the .h5 file
    """
    with open(pkl_path, 'rb') as f:
        rollout = pkl.load(f)
    
    if 'camera0_obs' not in rollout or 'camera1_obs' not in rollout:
        print(f"Warning: {pkl_path} does not contain camera0_obs or camera1_obs, skipping...")
        return False
    
    camera0_images = rollout['camera0_obs']
    camera1_images = rollout['camera1_obs']
    
    if len(camera0_images) == 0 or len(camera1_images) == 0:
        print(f"Warning: {pkl_path} has empty camera observations, skipping...")
        return False
    
    basename = os.path.basename(pkl_path).replace('.pkl', '.h5')
    h5_path = os.path.join(h5_output_dir, basename)
    
    with h5py.File(h5_path, 'w') as f:
        if isinstance(camera0_images, list):
            camera0_images = np.array(camera0_images)
        if isinstance(camera1_images, list):
            camera1_images = np.array(camera1_images)
        
        f.create_dataset('camera0_obs', data=camera0_images, compression='gzip')
        f.create_dataset('camera1_obs', data=camera1_images, compression='gzip')
        
        for key in rollout.keys():
            if key not in ['camera0_obs', 'camera1_obs']:
                try:
                    data = rollout[key]
                    if isinstance(data, list):
                        data = np.array(data)
                    if isinstance(data, np.ndarray):
                        f.create_dataset(key, data=data, compression='gzip')
                except Exception as e:
                    print(f"Could not save key {key}: {e}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert rollout .pkl files to .h5 format for VAE training')
    parser.add_argument('--rollout_dir', type=str, 
                       default=os.path.join(os.path.dirname(__file__), 'rollouts'),
                       help='Directory containing rollout .pkl files')
    parser.add_argument('--output_dir', type=str, 
                       default=os.path.join(os.path.dirname(__file__), 'data'),
                       help='Directory to save .h5 files')
    parser.add_argument('--recursive', action='store_true',
                       help='Search for .pkl files recursively')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.recursive:
        pkl_files = glob.glob(os.path.join(args.rollout_dir, '**', '*.pkl'), recursive=True)
    else:
        pkl_files = glob.glob(os.path.join(args.rollout_dir, '*.pkl'))
    
    if len(pkl_files) == 0:
        print(f"No .pkl files found in {args.rollout_dir}")
        return
    
    successful = 0
    failed = 0
    
    for pkl_path in tqdm(pkl_files, desc="Converting rollouts"):
        try:
            if convert_rollout_to_h5(pkl_path, args.output_dir):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error converting {pkl_path}: {e}")
            failed += 1
    
    print("success:" + str(successful) + " failed:" + str(failed))


if __name__ == '__main__':
    main()