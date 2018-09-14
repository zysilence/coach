import argparse

from rl_coach.base_parameters import TaskParameters
from rl_coach.core_types import EnvironmentEpisodes, RunPhase
from rl_coach.presets.CartPole_DQN_distributed import construct_graph


# TODO: workers might need to define schedules in terms which can be synchronized: exploration(len(distributed_memory)) -> float

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--redis_ip',
                        help="(string) IP or host for the redis server",
                        default='localhost',
                        type=str)
    parser.add_argument('-p', '--redis_port',
                        help="(int) Port of the redis server",
                        default=6379,
                        type=int)
    args = parser.parse_args()

    graph_manager = construct_graph(redis_ip=args.redis_ip, redis_port=args.redis_port)
    graph_manager.create_graph(TaskParameters())
    graph_manager.phase = RunPhase.TRAIN
    graph_manager.act(EnvironmentEpisodes(num_steps=10))


if __name__ == '__main__':
    main()
