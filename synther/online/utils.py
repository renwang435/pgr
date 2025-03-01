import gym
import numpy as np
from gym.wrappers.flatten_observation import FlattenObservation
from redq.algos.core import ReplayBuffer


# Make transition dataset from REDQ replay buffer.
def make_inputs_from_replay_buffer(
        replay_buffer: ReplayBuffer,
        model_terminals: bool = False,
) -> np.ndarray:
    ptr_location = replay_buffer.ptr
    obs = replay_buffer.obs1_buf[:ptr_location]
    actions = replay_buffer.acts_buf[:ptr_location]
    next_obs = replay_buffer.obs2_buf[:ptr_location]
    rewards = replay_buffer.rews_buf[:ptr_location]
    inputs = [obs, actions, rewards[:, None], next_obs]
    if model_terminals:
        terminals = replay_buffer.done_buf[:ptr_location].astype(np.float32)
        inputs.append(terminals[:, None])
    return np.concatenate(inputs, axis=1)
