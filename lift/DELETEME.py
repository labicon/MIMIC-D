import pickle as pkl

def main():
    with open("rollouts/newslower/rollout_seed0_mode2.pkl", "rb") as f:
        rollout = pkl.load(f)
        print(rollout.keys())
        print(rollout["camera_obs0"][0].shape)

import os

path = "/home/icon-labtop/anthony/MIMIC-D/lift/data/models/VAE_models_ICON/trained_models/Cond_ODE_TwoArmLift_specs_256_4_3_lift_mpc_P25E1_crosscond_nofinalpos_rotvec_separatenorm_dual_camera.pt"

def testLoad():
    print("File exists?", os.path.exists(path))


if __name__ == "__main__":
    testLoad()

   