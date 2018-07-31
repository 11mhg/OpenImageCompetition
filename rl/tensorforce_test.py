import tensorflow as tf
import numpy as np
from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

environment = OpenAIGym('class_1',visualize=False)




network_spec = [
    # dict(type='embedding', indices=100, size=32),
    # dict(type'flatten'),
    dict(type='dense', size=32),
    dict(type='dense', size=32)
]


agent = PPOAgent (
        states = environment.states,
        actions= environment.actions,
        network = network_spec,
        #agent
        states_preprocess=None,
        actions_exploration=None,
        reward_preprocessing=None,
        #memory model
        update_mode = dict(
                unit='episodes',
                batch_size=20,
                frequency=20
            ),
        memory=dict(
        type='latest',
        include_next_states=False,
        capacity=5000
    ),
    # DistributionModel
    distributions=None,
    entropy_regularization=0.01,
    # PGModel
    baseline_mode='states',
    baseline=dict(
        type='mlp',
        sizes=[32, 32]
    ),
    baseline_optimizer=dict(
        type='multi_step',
        optimizer=dict(
            type='adam',
            learning_rate=1e-3
        ),
        num_steps=5
    ),
    gae_lambda=0.97,
    # PGLRModel
    likelihood_ratio_clipping=0.2,
    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-3
    ),
    subsampling_fraction=0.2,
    optimization_steps=25,
    execution=dict(
        type='single',
        session_config=None,
        distributed_spec=None
    )
)
    
# Create the runner
runner = Runner(agent=agent, environment=environment)


# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    return True


# Start learning
runner.run(episodes=3000, max_episode_timesteps=200, episode_finished=episode_finished)
runner.close()

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)  