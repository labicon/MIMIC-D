# This script converts the rollout .pkl files into .h5 format for the VAE module, vae_testerencoder.py to match Jingwen's implementation.

import pickle as pkl
import h5py
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm


def convert_rollout_to_h5(pkl_path, h5_output_dir):
    with open(pkl_path, 'rb') as f:
        rollout = pkl.load(f)

    if 'camera_obs0' not in rollout or 'camera_obs1' not in rollout:
        print(f"Warning: {pkl_path} missing camera observations, skipping...")
        return False

    camera0_images = np.array(rollout['camera_obs0'])
    camera1_images = np.array(rollout['camera_obs1'])

    if len(camera0_images) == 0 or len(camera1_images) == 0:
        print(f"Warning: {pkl_path} has empty camera observations, skipping...")
        return False

    basename = os.path.basename(pkl_path).replace('.pkl', '.h5')
    h5_path = os.path.join(h5_output_dir, basename)

    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('robot0_eye_in_hand_image', data=camera0_images, compression='gzip')
        f.create_dataset('robot1_eye_in_hand_image', data=camera1_images, compression='gzip')

        for key in rollout.keys():
            if key not in ['observations', 'camera_obs0', 'camera_obs1']:
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
                       default=os.path.join(os.path.dirname(__file__), 'rollouts/newslower'),
                       help='Directory containing rollout .pkl files')
    parser.add_argument('--output_dir', type=str, 
                       default=os.path.join(os.path.dirname(__file__), 'data/h5data'),
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