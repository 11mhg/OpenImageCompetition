import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
        id = 'class_1',
        entry_point='gym_class.envs:ClassEnv',
        timestep_limit=1000,
        reward_threshold=15.0,
        nondeterministic=True)
