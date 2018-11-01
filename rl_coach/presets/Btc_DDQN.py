from rl_coach.agents.ddqn_agent import DDQNAgentParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.gym_environment import GymEnvironmentParameters
from rl_coach.filters.filter import NoOutputFilter, InputFilter
from rl_coach.filters.reward.reward_clipping_filter import RewardClippingFilter
from rl_coach.graph_managers.basic_rl_graph_manager import BasicRLGraphManager
from rl_coach.graph_managers.graph_manager import ScheduleParameters

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
agent_params = DDQNAgentParameters()
agent_params.network_wrappers['main'].learning_rate = 0.00025
agent_params.algorithm.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(1000)  # 32k frames

###############
# Environment #
###############
env_params = GymEnvironmentParameters()
input_filter = InputFilter(is_a_reference_filter=True)
input_filter.add_reward_filter('clipping', RewardClippingFilter(-1.0, 1.0))
env_params.level = "rl_coach.environments.user.btc:BitcoinEnv"
env_params.default_input_filter = input_filter
env_params.default_output_filter = NoOutputFilter()

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.test = True
preset_validation_params.min_reward_threshold = -1
preset_validation_params.max_episodes_to_achieve_reward = 10

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)