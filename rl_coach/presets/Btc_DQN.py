from rl_coach.agents.dqn_agent import DQNAgentParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.gym_environment import GymEnvironmentParameters
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter, InputFilter
from rl_coach.filters.reward.reward_clipping_filter import RewardClippingFilter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.schedules import LinearSchedule


####################
# Graph Scheduling #
####################

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(10)
schedule_params.evaluation_steps = EnvironmentEpisodes(1)
schedule_params.heatup_steps = EnvironmentSteps(1000)

#########
# Agent #
#########
agent_params = DQNAgentParameters()
# since we are using Adam instead of RMSProp, we adjust the learning rate as well
agent_params.network_wrappers['main'].learning_rate = 0.0001
# agent_params.exploration.epsilon_schedule = LinearSchedule(1, 0.1, 10000000)  # decault value: 1000000
agent_params.exploration.evaluation_epsilon = 0.0
# agent_params.network_wrappers['main'].input_embedders_parameters['observation'].dropout = True

###############
# Environment #
###############
env_params = GymEnvironmentParameters()
input_filter = InputFilter(is_a_reference_filter=True)
input_filter.add_reward_filter('clipping', RewardClippingFilter(-1.0, 1.0))
env_params.level = "rl_coach.environments.user.btc:BitcoinEnv"
env_params.default_input_filter = input_filter
env_params.default_output_filter = NoOutputFilter()
# env_params.additional_simulator_parameters = {"time_limit": 1000}

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
preset_validation_params.max_episodes_to_achieve_reward = 1

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=visual_params,
                                    preset_validation_params=preset_validation_params)
