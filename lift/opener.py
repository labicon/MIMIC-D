import pickle as pkl


with open("rollouts/newslower/rollout_seed0_mode2.pkl", 'rb') as f:
    rollout = pkl.load(f)
    print(rollout.keys())