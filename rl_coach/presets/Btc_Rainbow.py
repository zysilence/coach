from rl_coach.agents.rainbow_dqn_agent import RainbowDQNAgentParameters
from rl_coach.architectures.tensorflow_components.layers import Conv2d, Dense
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters, EmbedderScheme
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.gym_environment import GymEnvironmentParameters, ObservationSpaceType
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.filters.filter import NoOutputFilter, InputFilter
from rl_coach.filters.reward.reward_clipping_filter import RewardClippingFilter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.schedules import LinearSchedule

####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentSteps(6000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(10)
schedule_params.evaluation_steps = EnvironmentSteps(1)
schedule_params.heatup_steps = EnvironmentSteps(1000)

#########
# Agent #
#########
agent_params = RainbowDQNAgentParameters()
agent_params.network_wrappers['main'].learning_rate = 0.0000625
agent_params.network_wrappers['main'].optimizer_epsilon = 1.5e-4
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(1000)  # 32k frames
agent_params.memory.max_size = (MemoryGranularity.Transitions, 100000)  # default: 1000000
agent_params.memory.beta = LinearSchedule(0.4, 1, 12500000)  # 12.5M training iterations = 50M steps = 200M frames
agent_params.memory.alpha = 0.5
agent_params.exploration = EGreedyParameters()
agent_params.exploration.epsilon_schedule = LinearSchedule(1, 0.1, 2000000)  # decault value: 1000000
agent_params.exploration.evaluation_epsilon = 0
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].scheme = \
    [
        Conv2d(32, [5, 1], [2, 1]),
        Conv2d(64, [3, 1], 1),
        Conv2d(64, [3, 1], 1)
    ]
"""
agent_params.network_wrappers['main'].input_embedders_parameters['observation'].scheme = \
    [
    Conv2d(32, [5, 1], [2, 1]),
    Conv2d(64, [3, 1], 1),
    Conv2d(64, [3, 1], 1),
    Conv2d(128, [3, 1], 1),
    Conv2d(256, [3, 1], 1)
    ]
"""

###############
# Environment #
###############
env_params = GymEnvironmentParameters()
input_filter = InputFilter(is_a_reference_filter=True)
input_filter.add_reward_filter('clipping', RewardClippingFilter(-1.0, 1.0))
env_params.level = "rl_coach.environments.user.btc:BitcoinEnv"
env_params.default_input_filter = input_filter
env_params.default_output_filter = NoOutputFilter()
env_params.observation_space_type = ObservationSpaceType.Tensor

###############
# Visualization #
###############
visual_params = VisualizationParameters()
visual_params.native_rendering = True

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.test = True
preset_validation_params.min_reward_threshold = -1
preset_validation_params.max_episodes_to_achieve_reward = 10

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=visual_params,
                                    preset_validation_params=preset_validation_params)
