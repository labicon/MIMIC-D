import torch
import numpy as np
from utils.conditional_Action_DiT import Conditional_ODE
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils.discrete import *
import sys
import pdb
import csv
from utils.gmm import expert_likelihood
from joblib import dump, load
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def splice_plan(ode_model, env, initial_state, fixed_goal, model_i, segment_length=10, total_steps=100):
    """
    Plans a full trajectory (total_steps long) by iteratively planning
    segment_length-steps using the diffusion model.
    
    Parameters:
      - ode_model: the Conditional_ODE (diffusion model) instance.
      - env: your environment, which must implement reset_to() and step().
      - initial_state: a numpy array of shape (state_size,) (the current state).
      - fixed_goal: a numpy array of shape (state_size,) representing the final goal.
      - model_i: the index of the agent/model being trained
      - segment_length: number of timesteps to plan in each segment.
      - total_steps: total length of the planned trajectory.
    
    Returns:
      - full_traj: a numpy array of shape (total_steps, state_size)
    """
    full_traj = []
    current_state = initial_state.copy()
    n_segments = total_steps // segment_length
    for seg in range(n_segments):
        cond = np.hstack([current_state, fixed_goal])
        cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)
        sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=model_i)
        segment = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, action_size)

        if seg == 0:
            full_traj.extend(segment)
        else:
            full_traj.extend(segment[1:])

        current_state = segment[-1]
    return np.array(full_traj)

def splice_plan_multi(ode_model, env, initial_states, fixed_goals, segment_length=10, total_steps=100):
    """
    Plans a full multi-agent trajectory by repeatedly sampling 10-step segments.
    Each agent’s condition is built as:
      [ own current state, own goal, other_agent_1 current state, other_agent_1 goal, ... ]
      
    Parameters:
      - ode_model: the diffusion model (must support sample() that accepts a single condition tensor).
      - env: the environment (to get state dimensions, etc.)
      - initial_states: numpy array of shape (n_agents, state_size) with each agent's current state.
      - fixed_goals: numpy array of shape (n_agents, state_size) with each agent's final desired state.
      - segment_length: number of timesteps planned per segment.
      - total_steps: total number of timesteps for the full trajectory.
      
    Returns:
      - full_traj: numpy array of shape (total_steps, n_agents, state_size)
    """
    n_agents = len(initial_states)
    current_states = initial_states.copy()  # will be updated at every segment
    full_segments = []
    n_segments = total_steps // segment_length

    # Loop over planning segments.
    for seg in range(n_segments):
        seg_trajectories = []
        # For each agent, build its own condition and sample a trajectory segment.
        for i in range(n_agents):
            # Start with agent i's current state and goal.
            cond = [current_states[i], fixed_goals[i]]
            # Append other agents' current state and goal.
            for j in range(n_agents):
                if j != i:
                    cond.append(current_states[j])
                    cond.append(fixed_goals[j])
            # Create a 1D condition vector for agent i.
            cond_vector = np.hstack(cond)
            # Convert to tensor.
            cond_tensor = torch.tensor(cond_vector, dtype=torch.float32, device=device).unsqueeze(0)
            # Sample a segment for agent i.
            sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=i)
            seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, state_size)
            seg_trajectories.append(seg_i)
            # Update current state for agent i (using the last state from the segment)
            current_states[i] = seg_i[-1]
        # Stack the segments for all agents. Shape: (n_agents, segment_length, state_size)
        seg_array = np.stack(seg_trajectories, axis=0)
        full_segments.append(seg_array)

    # Concatenate segments along the time axis.
    # This yields an array of shape (n_agents, total_steps, state_size)
    full_traj = np.concatenate(full_segments, axis=1)
    # Optionally, transpose so that time is the first dimension:
    # full_traj = np.transpose(full_traj, (1, 0, 2))
    return full_traj

def splice_plan_safe(ode_model, env, initial_states, fixed_goals, segment_length=10, total_steps=100):
    """
    Plans a full multi-agent trajectory by repeatedly sampling 10-step segments with a safety filter.
    Each agent’s condition is built as:
      [ own current state, own goal, other_agent_1 current state, other_agent_1 goal, ... ]
      
    Parameters:
      - ode_model: the diffusion model (must support sample() that accepts a single condition tensor).
      - env: the environment (to get state dimensions, etc.)
      - initial_states: numpy array of shape (n_agents, state_size) with each agent's current state.
      - fixed_goals: numpy array of shape (n_agents, state_size) with each agent's final desired state.
      - segment_length: number of timesteps planned per segment.
      - total_steps: total number of timesteps for the full trajectory.
      
    Returns:
      - full_traj: numpy array of shape (total_steps, n_agents, state_size)
    """
    n_agents = len(initial_states)
    current_states = initial_states.copy()  # will be updated at every segment
    full_segments = []
    n_segments = total_steps // segment_length
    gmm = load("expert_gmm.pkl")

    # Loop over planning segments.
    for seg in range(n_segments):
        valid_segment = False
        while not valid_segment:
          seg_trajectories = []
          current_states_temp = initial_states.copy()
          # For each agent, build its own condition and sample a trajectory segment.
          likely_vec = np.zeros(n_agents*2)
          for i in range(n_agents):
              # Start with agent i's current state and goal.
              cond = [current_states[i], fixed_goals[i]]
              # Append other agents' current state and goal.
              for j in range(n_agents):
                  if j != i:
                      cond.append(current_states[j])
                      cond.append(fixed_goals[j])
              # Create a 1D condition vector for agent i.
              cond_vector = np.hstack(cond)
              # Convert to tensor.
              cond_tensor = torch.tensor(cond_vector, dtype=torch.float32, device=device).unsqueeze(0)
              # Sample a segment for agent i.
              sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=i)
              seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, state_size)
              likely_vec[i*2] = seg_i[-1][0]
              likely_vec[i*2 + 1] = seg_i[-1][1]
              seg_trajectories.append(seg_i)
              # Update current state for agent i (using the last state from the segment)
              current_states_temp[i] = seg_i[-1]
              # current_states[i] = seg_i[-1]
          prob = expert_likelihood(gmm, likely_vec)
          print(prob)
          if prob > 0.045:
              print("valid")
              valid_segment = True
              for i in range(n_agents):
                  current_states[i] = current_states_temp[i]
        # Stack the segments for all agents. Shape: (n_agents, segment_length, state_size)
        seg_array = np.stack(seg_trajectories, axis=0)
        full_segments.append(seg_array)

    # Concatenate segments along the time axis.
    # This yields an array of shape (n_agents, total_steps, state_size)
    full_traj = np.concatenate(full_segments, axis=1)
    return full_traj

def splice_plan_mode_safe(ode_model, env, initial_states, fixed_goals, mode, segment_length=10, total_steps=100):
    """
    Plans a full multi-agent trajectory by repeatedly sampling 10-step segments with a safety filter.
    Each agent’s condition is built as:
      [ own current state, own goal, other_agent_1 current state, other_agent_1 goal, ... ]
      
    Parameters:
      - ode_model: the diffusion model (must support sample() that accepts a single condition tensor).
      - env: the environment (to get state dimensions, etc.)
      - initial_states: numpy array of shape (n_agents, state_size) with each agent's current state.
      - fixed_goals: numpy array of shape (n_agents, state_size) with each agent's final desired state.
      - segment_length: number of timesteps planned per segment.
      - total_steps: total number of timesteps for the full trajectory.
      
    Returns:
      - full_traj: numpy array of shape (total_steps, n_agents, state_size)
    """
    n_agents = len(initial_states)
    current_states = initial_states.copy()  # will be updated at every segment
    full_segments = []
    n_segments = total_steps // segment_length
    gmm = load("expert_gmm.pkl")

    # Loop over planning segments.
    for seg in range(n_segments):
        valid_segment = False
        while not valid_segment:
          seg_trajectories = []
          current_states_temp = initial_states.copy()
          # For each agent, build its own condition and sample a trajectory segment.
          likely_vec = np.zeros(n_agents*2)
          for i in range(n_agents):
              # Start with agent i's current state and goal.
              cond = [current_states[i], fixed_goals[i]]
              # Append other agents' current state and goal.
              for j in range(n_agents):
                  if j != i:
                      cond.append(current_states[j])
                      cond.append(fixed_goals[j])
              cond.append(mode)
              # Create a 1D condition vector for agent i.
              cond_vector = np.hstack(cond)
              # Convert to tensor.
              cond_tensor = torch.tensor(cond_vector, dtype=torch.float32, device=device).unsqueeze(0)
              # Sample a segment for agent i.
              sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=i)
              seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, state_size)
              likely_vec[i*2] = seg_i[-1][0]
              likely_vec[i*2 + 1] = seg_i[-1][1]
              seg_trajectories.append(seg_i)
              # Update current state for agent i (using the last state from the segment)
              current_states_temp[i] = seg_i[-1]
              # current_states[i] = seg_i[-1]
          prob = expert_likelihood(gmm, likely_vec)
          print(prob)
          if prob > 0.045:
              print("valid")
              valid_segment = True
              for i in range(n_agents):
                  current_states[i] = current_states_temp[i]
          else:
              mode = np.random.randint(0, 6) + 1
        # Stack the segments for all agents. Shape: (n_agents, segment_length, state_size)
        seg_array = np.stack(seg_trajectories, axis=0)
        full_segments.append(seg_array)

    # Concatenate segments along the time axis.
    # This yields an array of shape (n_agents, total_steps, state_size)
    full_traj = np.concatenate(full_segments, axis=1)
    return full_traj

def splice_plan_mode_multi(ode_model, env, initial_states, fixed_goals, mode, segment_length=10, total_steps=100):
    """
    Plans a full multi-agent trajectory by repeatedly sampling 10-step segments.
    Each agent’s condition is built as:
      [ own current state, own goal, other_agent_1 current state, other_agent_1 goal, ... ]
      
    Parameters:
      - ode_model: the diffusion model (must support sample() that accepts a single condition tensor).
      - env: the environment (to get state dimensions, etc.)
      - initial_states: numpy array of shape (n_agents, state_size) with each agent's current state.
      - fixed_goals: numpy array of shape (n_agents, state_size) with each agent's final desired state.
      - segment_length: number of timesteps planned per segment.
      - total_steps: total number of timesteps for the full trajectory.
      
    Returns:
      - full_traj: numpy array of shape (total_steps, n_agents, state_size)
    """
    n_agents = len(initial_states)
    current_states = initial_states.copy()  # will be updated at every segment
    full_segments = []
    n_segments = total_steps // segment_length

    # Loop over planning segments.
    for seg in range(n_segments):
        seg_trajectories = []
        # For each agent, build its own condition and sample a trajectory segment.
        for i in range(n_agents):
            # Start with agent i's current state and goal.
            cond = [current_states[i], fixed_goals[i]]
            # Append other agents' current state and goal.
            for j in range(n_agents):
                if j != i:
                    cond.append(current_states[j])
                    cond.append(fixed_goals[j])
            # Create a 1D condition vector for agent i.
            cond.append(mode)
            cond_vector = np.hstack(cond)
            # Convert to tensor.
            cond_tensor = torch.tensor(cond_vector, dtype=torch.float32, device=device).unsqueeze(0)
            # Sample a segment for agent i.
            sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=i)
            seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, state_size)
            seg_trajectories.append(seg_i)
            # Update current state for agent i (using the last state from the segment)
            current_states[i] = seg_i[-1]
        # Stack the segments for all agents. Shape: (n_agents, segment_length, state_size)
        seg_array = np.stack(seg_trajectories, axis=0)
        full_segments.append(seg_array)

    # Concatenate segments along the time axis.
    # This yields an array of shape (n_agents, total_steps, state_size)
    full_traj = np.concatenate(full_segments, axis=1)
    # Optionally, transpose so that time is the first dimension:
    # full_traj = np.transpose(full_traj, (1, 0, 2))
    return full_traj

def collision_cost(traj, obstacles, safety_margin=0.5):
    # traj: (segment_length, state_size) trajectory
    # obstacle: tuple (ox, oy, r)
    # Compute cost as, for example, inverse of the distance to the obstacle at each timestep.
    costs = []
    for obstacle in obstacles:
      cost = 0
      ox, oy, r = obstacle
      for state in traj:
        x, y = state[:2]
        dist = np.sqrt((x - ox)**2 + (y - oy)**2)
        # If within the safety margin, add a high penalty
        if dist < safety_margin:
            cost += 1e3
        else:
            cost += 1.0 / dist  # lower cost for further states
      costs.append(cost)
    return max(costs)

def splice_plan_multi_safe(ode_model, env, initial_states, fixed_goals, segment_length=10, total_steps=100, n_candidates=5):
    n_agents = len(initial_states)
    current_states = initial_states.copy()  # update each segment
    full_segments = []
    n_segments = total_steps // segment_length
    gmm = load('expert_gmm.pkl')

    for seg in range(n_segments):
        seg_trajectories = []
        obstacles = []
        for i in range(n_agents):
            best_traj = None
            best_cost = float('inf')
            # Sample multiple candidates
            for _ in range(n_candidates):
                cond = [current_states[i], fixed_goals[i]]
                for j in range(n_agents):
                    if j != i:
                        cond.append(current_states[j])
                        cond.append(fixed_goals[j])
                        obstacle = (current_states[j][0], current_states[j][1], 2)
                        obstacles.append(obstacle)
                cond_vector = np.hstack(cond)
                cond_tensor = torch.tensor(cond_vector, dtype=torch.float32, device=device).unsqueeze(0)
                sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=i)
                candidate = sampled.cpu().detach().numpy()[0]
                breakpoint()
                cost = collision_cost(candidate, obstacles)
                if cost < best_cost:
                    best_cost = cost
                    best_traj = candidate
            seg_trajectories.append(best_traj)
            current_states[i] = best_traj[-1]
        seg_array = np.stack(seg_trajectories, axis=0)
        full_segments.append(seg_array)
    full_traj = np.concatenate(full_segments, axis=1)
    return full_traj


def splice_plan_multi_true(ode_model, env, initial_states, fixed_goals, segment_length=10, total_steps=100):
    """
    True MPC: At each step, plan a full segment but only execute the first step.
    
    Parameters:
      - ode_model: the diffusion model (must support sample() that accepts a single condition tensor).
      - env: the environment (to get state dimensions, etc.)
      - initial_states: numpy array of shape (n_agents, state_size).
      - fixed_goals: numpy array of shape (n_agents, state_size).
      - segment_length: how many steps we plan ahead (default 10).
      - total_steps: how many total steps to run.
      
    Returns:
      - full_traj: numpy array of shape (total_steps, n_agents, state_size)
    """
    n_agents = len(initial_states)
    current_states = initial_states.copy()  # will be updated at every step
    full_traj = []

    for step in range(total_steps):
        next_states = []
        # For each agent, plan a 10-step trajectory, but we'll only take the first action.
        for i in range(n_agents):
            # Build the condition for agent i
            cond1 = np.hstack([current_states[0], fixed_goals[0]])
            cond2 = np.hstack([current_states[1], fixed_goals[1]])
            cond1_tensor = torch.tensor(cond1, dtype=torch.float32, device=device).unsqueeze(0)
            cond2_tensor = torch.tensor(cond2, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Sample a full segment
            sampled = ode_model.sample(attr=[cond1_tensor, cond2_tensor], traj_len=segment_length, n_samples=1, w=1., model_index=i)
            seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, state_size)

            # Take only the first step
            next_state_i = seg_i[1]
            next_states.append(next_state_i)

        # Update current states
        current_states = np.array(next_states)
        # Save the executed states
        full_traj.append(current_states)

    full_traj = np.stack(full_traj, axis=1)  # Shape: (total_steps, n_agents, state_size)
    return full_traj


def mpc_plan(ode_model, env, initial_state, fixed_goal, model_i, leader_traj_cond = None, segment_length=10, total_steps=100):
    """
    Plans a full trajectory (total_steps long) by iteratively planning
    segment_length-steps using the diffusion model and replanning at every timestep.
    
    Parameters:
      - ode_model: the Conditional_ODE (diffusion model) instance.
      - env: your environment, which must implement reset_to() and step().
      - initial_state: a numpy array of shape (state_size,) (the current state).
      - fixed_goal: a numpy array of shape (state_size,) representing the final goal.
      - model_i: the index of the agent/model being planned for.
      - segment_length: number of timesteps to plan in each segment.
      - total_steps: total length of the planned trajectory.
    
    Returns:
      - full_traj: a numpy array of shape (total_steps, state_size)
    """
    full_traj = []
    current_state = initial_state.copy()
    n_segments = total_steps // segment_length

    for seg in range(100):
        if leader_traj_cond is not None:
            cond = np.hstack([current_state, fixed_goal, leader_traj_cond.flatten()])
        else:
            cond = np.hstack([current_state, fixed_goal])
        cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)
        sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=model_i)
        segment = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, action_size)

        next_state_i = segment[1]
        full_traj.append(next_state_i)

        current_state = next_state_i
    return np.array(full_traj)


def reactive_mpc_plan(ode_model, env, initial_states, fixed_goals, model_i, segment_length=25, total_steps=100, n_implement=5):
    """
    Plans a full trajectory (total_steps long) by iteratively planning
    segment_length-steps using the diffusion model and replanning at every timestep.
    
    Parameters:
      - ode_model: the Conditional_ODE (diffusion model) instance.
      - env: your environment, which must implement reset_to() and step().
      - initial_state: a numpy array of shape (state_size,) (the current state).
      - fixed_goal: a numpy array of shape (state_size,) representing the final goal.
      - model_i: the index of the agent/model being planned for.
      - segment_length: number of timesteps to plan in each segment.
      - total_steps: total length of the planned trajectory.
    
    Returns:
      - full_traj: a numpy array of shape (total_steps, state_size)
    """
    full_traj = []
    current_states = initial_states.copy()

    for seg in range(total_steps // n_implement):
        segments = []
        for i in range(len(current_states)):
            if i == 0:
                cond = [current_states[i], fixed_goals[i]]
                for j in range(len(current_states)):
                    if j != i:
                        cond.append(current_states[j])
                        cond.append(fixed_goals[j])
                cond = np.hstack(cond)
                cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)
                sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=0)
                seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, action_size)

                if seg == 0:
                    segments.append(seg_i[0:n_implement,:])
                    current_states[i] = seg_i[n_implement-1,:]
                else:
                    segments.append(seg_i[1:n_implement+1,:])
                    current_states[i] = seg_i[n_implement,:]

            else:
                cond = [current_states[i], fixed_goals[i]]
                for j in range(len(current_states)):
                    if j != i and j != 0:
                        cond.append(current_states[j])
                        cond.append(fixed_goals[j])
                cond.append(current_states[0])
                cond.append(fixed_goals[0])
                cond = np.hstack(cond)
                cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)
                sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=i)
                seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, action_size)

                if seg == 0:
                    segments.append(seg_i[0:n_implement,:])
                    current_states[i] = seg_i[n_implement-1,:]
                else:
                    segments.append(seg_i[1:n_implement+1,:])
                    current_states[i] = seg_i[n_implement,:]
        
        seg_array = np.stack(segments, axis=0)
        full_traj.append(seg_array)

    full_traj = np.concatenate(full_traj, axis=1) 
    return np.array(full_traj)


def reactive_mpc_plan_smallcond(ode_model, env, initial_states, fixed_goals, model_i, segment_length=25, total_steps=100, n_implement=5):
    """
    Plans a full trajectory (total_steps long) by iteratively planning
    segment_length-steps using the diffusion model and replanning at every timestep.
    
    Parameters:
      - ode_model: the Conditional_ODE (diffusion model) instance.
      - env: your environment, which must implement reset_to() and step().
      - initial_state: a numpy array of shape (state_size,) (the current state).
      - fixed_goal: a numpy array of shape (state_size,) representing the final goal.
      - model_i: the index of the agent/model being planned for.
      - segment_length: number of timesteps to plan in each segment.
      - total_steps: total length of the planned trajectory.
    
    Returns:
      - full_traj: a numpy array of shape (total_steps, state_size)
    """
    full_traj = []
    current_states = initial_states.copy()

    for seg in range(total_steps // n_implement):
        segments = []
        for i in range(len(current_states)):
            if i == 0:
                cond = [current_states[i], fixed_goals[i]]
                for j in range(len(current_states)):
                    if j != i:
                        cond.append(current_states[j])
                cond = np.hstack(cond)
                cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)
                sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=0)
                seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, action_size)

                if seg == 0:
                    segments.append(seg_i[0:n_implement,:])
                    current_states[i] = seg_i[n_implement-1,:]
                else:
                    segments.append(seg_i[1:n_implement+1,:])
                    current_states[i] = seg_i[n_implement,:]

            else:
                cond = [current_states[i], fixed_goals[i]]
                for j in range(len(current_states)):
                    if j != i and j != 0:
                        cond.append(current_states[j])
                cond.append(current_states[0])
                cond = np.hstack(cond)
                cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)
                sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=i)
                seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, action_size)

                if seg == 0:
                    segments.append(seg_i[0:n_implement,:])
                    current_states[i] = seg_i[n_implement-1,:]
                else:
                    segments.append(seg_i[1:n_implement+1,:])
                    current_states[i] = seg_i[n_implement,:]
        
        seg_array = np.stack(segments, axis=0)
        full_traj.append(seg_array)

    full_traj = np.concatenate(full_traj, axis=1) 
    return np.array(full_traj)

def reactive_mpc_plan_nolf(
        ode_model,
        initial_states,
        fixed_goals,
        segment_length=25,
        total_steps=100,
        n_implement=5):
    """
    Plans a full trajectory (total_steps long) by iteratively planning
    segment_length-steps using the diffusion model and replanning at every timestep,
    but ensures each agent conditions on its peers at the same timestep.
    
    Parameters:
      - ode_model: the Conditional_ODE instance.
      - initial_states: list or array of shape (n_agents, state_size).
      - fixed_goals:    list or array of shape (n_agents, state_size).
      - segment_length: number of timesteps to plan in each segment.
      - total_steps:    total length of the planned trajectory.
      - n_implement:    number of steps to execute before replanning each segment.
    
    Returns:
      - full_traj: np.ndarray of shape (n_agents, total_steps, action_size)
    """
    full_traj = []
    current_states = initial_states.copy()      # shape: (n_agents, state_size)
    n_agents = len(current_states)

    for seg in range(total_steps // n_implement):
        # snapshot everyone's state at the start of this segment
        base_states = current_states.copy()     

        segments = []
        for i in range(n_agents):
            # build conditioning vector from base_states
            cond = [ base_states[i], fixed_goals[i] ]
            for j in range(n_agents):
                if j != i:
                    cond.append(base_states[j])
                    cond.append(fixed_goals[j])
            cond = np.hstack(cond)  # shape: (attr_dim,)
            cond_tensor = torch.tensor(cond, dtype=torch.float32,
                                       device=ode_model.device).unsqueeze(0)

            # sample this agent’s segment
            sampled = ode_model.sample(
                attr=cond_tensor,
                traj_len=segment_length,
                n_samples=1,
                w=1.0,
                model_index=i
            )
            seg_i = sampled.cpu().detach().numpy()[0]  # (segment_length, action_size)

            # select which slice to execute and update current_states
            if seg == 0:
                to_take   = seg_i[0:n_implement]
                new_state = seg_i[n_implement-1]
            else:
                to_take   = seg_i[1:n_implement+1]
                new_state = seg_i[n_implement]
            segments.append(to_take)
            current_states[i] = new_state

        # stack across agents: shape (n_agents, n_implement, action_size)
        full_traj.append(np.stack(segments, axis=0))

    # concat all segments along time: shape (n_agents, total_steps, action_size)
    full_traj = np.concatenate(full_traj, axis=1)
    return full_traj



def reactive_mpc_plan_vanilla(ode_model, env, initial_states, fixed_goals, model_i, segment_length=25, total_steps=100, n_implement=5):
    """
    Plans a full trajectory (total_steps long) by iteratively planning
    segment_length-steps using the diffusion model and replanning at every timestep.
    
    Parameters:
      - ode_model: the Conditional_ODE (diffusion model) instance.
      - env: your environment, which must implement reset_to() and step().
      - initial_state: a numpy array of shape (state_size,) (the current state).
      - fixed_goal: a numpy array of shape (state_size,) representing the final goal.
      - model_i: the index of the agent/model being planned for.
      - segment_length: number of timesteps to plan in each segment.
      - total_steps: total length of the planned trajectory.
    
    Returns:
      - full_traj: a numpy array of shape (total_steps, state_size)
    """
    full_traj = []
    current_states = initial_states.copy()

    for seg in range(total_steps // n_implement):
        segments = []
        for i in range(len(current_states)):
            if i == 0:
                cond = [current_states[i], fixed_goals[i]]
                cond = np.hstack(cond)
                cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)
                sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=0)
                seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, action_size)

                if seg == 0:
                    segments.append(seg_i[0:n_implement,:])
                    current_states[i] = seg_i[n_implement-1,:]
                else:
                    segments.append(seg_i[1:n_implement+1,:])
                    current_states[i] = seg_i[n_implement,:]

            else:
                cond = [current_states[i], fixed_goals[i]]
                cond = np.hstack(cond)
                cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)
                sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=i)
                seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, action_size)

                if seg == 0:
                    segments.append(seg_i[0:n_implement,:])
                    current_states[i] = seg_i[n_implement-1,:]
                else:
                    segments.append(seg_i[1:n_implement+1,:])
                    current_states[i] = seg_i[n_implement,:]
        
        seg_array = np.stack(segments, axis=0)
        full_traj.append(seg_array)

    full_traj = np.concatenate(full_traj, axis=1) 
    return np.array(full_traj)


def reactive_mpc_plan_guidesample(ode_model, env, initial_states, fixed_goals, model_i, segment_length=25, total_steps=100, n_implement=5):
    """
    Plans a full trajectory (total_steps long) by iteratively planning
    segment_length-steps using the diffusion model and replanning at every timestep.
    
    Parameters:
      - ode_model: the Conditional_ODE (diffusion model) instance.
      - env: your environment, which must implement reset_to() and step().
      - initial_state: a numpy array of shape (state_size,) (the current state).
      - fixed_goal: a numpy array of shape (state_size,) representing the final goal.
      - model_i: the index of the agent/model being planned for.
      - segment_length: number of timesteps to plan in each segment.
      - total_steps: total length of the planned trajectory.
    
    Returns:
      - full_traj: a numpy array of shape (total_steps, state_size)
    """
    full_traj = []
    current_states = initial_states.copy()

    for seg in range(total_steps // n_implement):
        segments = []
        for i in range(len(current_states)):
            if i == 0:
                cond = [current_states[i], fixed_goals[i]]
                for j in range(len(current_states)):
                    if j != i:
                        cond.append(current_states[j])
                        cond.append(fixed_goals[j])
                cond = np.hstack(cond)
                cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)
                sampled = ode_model.sample_guidance(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=0)
                seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, action_size)

                if seg == 0:
                    segments.append(seg_i[0:n_implement,:])
                    current_states[i] = seg_i[n_implement-1,:]
                else:
                    segments.append(seg_i[1:n_implement+1,:])
                    current_states[i] = seg_i[n_implement,:]

            else:
                cond = [current_states[i], fixed_goals[i]]
                for j in range(len(current_states)):
                    if j != i and j != 0:
                        cond.append(current_states[j])
                        cond.append(fixed_goals[j])
                cond.append(current_states[0])
                cond.append(fixed_goals[0])
                cond = np.hstack(cond)
                cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)
                sampled = ode_model.sample_guidance(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=i)
                seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, action_size)

                if seg == 0:
                    segments.append(seg_i[0:n_implement,:])
                    current_states[i] = seg_i[n_implement-1,:]
                else:
                    segments.append(seg_i[1:n_implement+1,:])
                    current_states[i] = seg_i[n_implement,:]
        
        seg_array = np.stack(segments, axis=0)
        full_traj.append(seg_array)

    full_traj = np.concatenate(full_traj, axis=1) 
    return np.array(full_traj)


def reactive_mpc_plan_smallcond_guidesample(ode_model, env, initial_states, fixed_goals, model_i, segment_length=25, total_steps=100, n_implement=5):
    """
    Plans a full trajectory (total_steps long) by iteratively planning
    segment_length-steps using the diffusion model and replanning at every timestep.
    
    Parameters:
      - ode_model: the Conditional_ODE (diffusion model) instance.
      - env: your environment, which must implement reset_to() and step().
      - initial_state: a numpy array of shape (state_size,) (the current state).
      - fixed_goal: a numpy array of shape (state_size,) representing the final goal.
      - model_i: the index of the agent/model being planned for.
      - segment_length: number of timesteps to plan in each segment.
      - total_steps: total length of the planned trajectory.
    
    Returns:
      - full_traj: a numpy array of shape (total_steps, state_size)
    """
    full_traj = []
    current_states = initial_states.copy()

    for seg in range(total_steps // n_implement):
        segments = []
        for i in range(len(current_states)):
            if i == 0:
                cond = [current_states[i], fixed_goals[i]]
                for j in range(len(current_states)):
                    if j != i:
                        cond.append(current_states[j])
                cond = np.hstack(cond)
                cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)
                sampled = ode_model.sample_guidance2(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=0)
                seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, action_size)

                if seg == 0:
                    segments.append(seg_i[0:n_implement,:])
                    current_states[i] = seg_i[n_implement-1,:]
                else:
                    segments.append(seg_i[1:n_implement+1,:])
                    current_states[i] = seg_i[n_implement,:]

            else:
                cond = [current_states[i], fixed_goals[i]]
                for j in range(len(current_states)):
                    if j != i and j != 0:
                        cond.append(current_states[j])
                cond.append(current_states[0])
                cond = np.hstack(cond)
                cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)
                sampled = ode_model.sample_guidance2(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=i, leader_current_pos=current_states[0])
                seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, action_size)

                if seg == 0:
                    segments.append(seg_i[0:n_implement,:])
                    current_states[i] = seg_i[n_implement-1,:]
                else:
                    segments.append(seg_i[1:n_implement+1,:])
                    current_states[i] = seg_i[n_implement,:]
        
        seg_array = np.stack(segments, axis=0)
        full_traj.append(seg_array)

    full_traj = np.concatenate(full_traj, axis=1) 
    return np.array(full_traj)


def reactive_mpc_latent_plan(ode_model, env, initial_states, fixed_goals, encoder, segment_length=10, total_steps=100, n_implement=1):
    """
    Plans a full trajectory (total_steps long) by iteratively planning
    segment_length-steps using the diffusion model and replanning at every timestep.
    Conditioned on the latent space representation of the history of the agents.
    
    Parameters:
      - ode_model: the Conditional_ODE (diffusion model) instance.
      - env: your environment, which must implement reset_to() and step().
      - initial_state: a numpy array of shape (state_size,) (the current state).
      - fixed_goal: a numpy array of shape (state_size,) representing the final goal.
      - model_i: the index of the agent/model being planned for.
      - segment_length: number of timesteps to plan in each segment.
      - total_steps: total length of the planned trajectory.
    
    Returns:
      - full_traj: a numpy array of shape (total_steps, state_size)
    """
    leader_history = []
    full_traj = []
    current_states = initial_states.copy()

    for seg in range(total_steps // n_implement):
        segments = []
        for i in range(len(current_states)):
            if i == 0:
                cond = [current_states[i], fixed_goals[i]]
                for j in range(len(current_states)):
                    if j != i:
                        cond.append(current_states[j])
                cond = np.hstack(cond)
                cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)
                sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=0)
                seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, action_size)
                for j in range(n_implement):
                    leader_history.append(seg_i[j,:])
                if seg == 0:
                    segments.append(seg_i[0:n_implement,:])
                    current_states[i] = seg_i[n_implement-1,:]
                else:
                    segments.append(seg_i[1:n_implement+1,:])
                    current_states[i] = seg_i[n_implement,:]

            else:
                _latent_list = np.array(leader_history)
                latent_list = np.vstack([np.repeat(_latent_list[0:1, :], repeats=total_steps-len(_latent_list), axis=0), _latent_list])
                latent = torch.from_numpy(np.array([latent_list])).float().to(device)
                z1 = encoder(latent)

                cond = [current_states[i], fixed_goals[i]]
                cond.append(current_states[0])
                cond.append(z1.detach().cpu().numpy()[0])
                cond = np.hstack(cond)
                cond_tensor = torch.tensor(cond, dtype=torch.float32, device=device).unsqueeze(0)
                sampled = ode_model.sample(attr=cond_tensor, traj_len=segment_length, n_samples=1, w=1., model_index=i)
                seg_i = sampled.cpu().detach().numpy()[0]  # shape: (segment_length, action_size)

                if seg == 0:
                    segments.append(seg_i[0:n_implement,:])
                    current_states[i] = seg_i[n_implement-1,:]
                else:
                    segments.append(seg_i[1:n_implement+1,:])
                    current_states[i] = seg_i[n_implement,:]
        
        seg_array = np.stack(segments, axis=0)
        full_traj.append(seg_array)

    full_traj = np.concatenate(full_traj, axis=1) 
    return np.array(full_traj)