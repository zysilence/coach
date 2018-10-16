from rl_coach.agents.dqn_agent import DQNAgentParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.gym_environment import GymEnvironmentParameters
from rl_coach.filters.filter import NoInputFilter, NoOutputFilter
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
agent_params = DQNAgentParameters()
# since we are using Adam instead of RMSProp, we adjust the learning rate as well
agent_params.network_wrappers['main'].learning_rate = 0.0001

###############
# Environment #
###############
env_params = GymEnvironmentParameters()
env_params.level = "rl_coach.environments.user.btc:BitcoinEnv"
env_params.default_input_filter = NoInputFilter()
env_params.default_output_filter = NoOutputFilter()
# env_params.additional_simulator_parameters = {"time_limit": 1000}

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.test = True
preset_validation_params.min_reward_threshold = -1
preset_validation_params.max_episodes_to_achieve_reward = 1

graph_manager = BasicRLGraphManager(agent_params=agent_params, env_params=env_params,
                                    schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                    preset_validation_params=preset_validation_params)
