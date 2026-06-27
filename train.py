import argparse
import distutils.version
import os
from datetime import datetime
from multiprocessing import Pool

import dill
dill.extend(False)
import gym
from gym import register
from gym.error import Error as GymError
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
dill.extend(True)


ENV_NAME = 'RLBOA-{}-{}-{}-v0'
ISSUE_NAMES = [
    'Laptop',
    'ItexvsCypress',
    'IS_BT_Acquisition',
    'Grocery',
    'thompson',
    'Car',
    'EnergySmall_A'
]
AGENT_LIST = [
    'Boulware',
    'Linear',
    'Conceder',
    'TitForTat1',
    'TitForTat2',
    'AgentK',
    'HardHeaded',
    'Atlas3',
    'AgentGG',
]


def register_neg_env(issue, agent, rl_position=0, n_actions=10, add_noise=True):
    env_name = ENV_NAME.format(issue, agent[0], agent[1])
    try:
        register(
            id=env_name,
            entry_point='envs.env:RLBOAEnv',
            kwargs={
                'domain': issue,
                'opponent': agent,
                'is_first': rl_position == 0,
                'rl_position': rl_position,
                'n_actions': n_actions,
                'add_noise': add_noise,
            },
        )
    except GymError:
        # Gym raises if this id has already been registered in this process.
        pass
    return env_name


def run_rl(args):
    (
        issue,
        agent,
        timesteps,
        n_envs,
        eval_episodes,
        rl_position,
        n_actions,
        add_noise,
        save_path,
    ) = args

    env_name = register_neg_env(issue, agent, rl_position, n_actions, add_noise)
    f_name = env_name.split('-', maxsplit=1)[1]
    env = make_vec_env(env_name, n_envs=n_envs)

    model = PPO('MlpPolicy', env, verbose=1, device='cpu')
    log_path = os.path.join(save_path, f_name)
    model_path = os.path.join(save_path, f'{f_name}.zip')
    model.set_logger(configure(log_path, ['stdout', 'tensorboard']))
    model.learn(total_timesteps=timesteps)
    model.save(model_path)
    print(f'saved_model:{model_path}')

    eval_env = gym.make(env_name)
    eval_env.test = True
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes)
    print(f'mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}')
    with open(save_path + 'result.csv', 'a') as f:
        f.write(f'{issue},{agent[0]},{agent[1]},{rl_position},{mean_reward},{std_reward}\n')

    env.close()
    eval_env.close()
    del model


def parse_args():
    parser = argparse.ArgumentParser(description='Train RLBOA in a three-party SAOP negotiation.')
    parser.add_argument('--domain', choices=ISSUE_NAMES)
    parser.add_argument('--opponent1', choices=AGENT_LIST)
    parser.add_argument('--opponent2', choices=AGENT_LIST)
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--n-envs', type=int, default=4)
    parser.add_argument('--eval-episodes', type=int, default=100)
    parser.add_argument('--rl-position', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--n-actions', type=int, default=10)
    parser.add_argument('--no-noise', action='store_true')
    parser.add_argument('--processes', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    save_path = './results/{}/'.format(datetime.now().strftime('%Y%m%d-%H%M%S')[2:])
    os.makedirs(save_path)
    with open(save_path + 'result.csv', 'w') as f:
        f.write('domain,opponent1,opponent2,rl_position,mean,std\n')

    if args.domain or args.opponent1 or args.opponent2:
        if not (args.domain and args.opponent1 and args.opponent2):
            raise ValueError('--domain, --opponent1, and --opponent2 must be specified together')
        jobs = [(args.domain, [args.opponent1, args.opponent2])]
    else:
        jobs = [
            (issue, [AGENT_LIST[i], AGENT_LIST[j]])
            for issue in ISSUE_NAMES
            for i in range(len(AGENT_LIST))
            for j in range(i, len(AGENT_LIST))
        ]

    run_args = [
        (
            issue,
            agent,
            args.timesteps,
            args.n_envs,
            args.eval_episodes,
            args.rl_position,
            args.n_actions,
            not args.no_noise,
            save_path,
        )
        for issue, agent in jobs
    ]

    if len(run_args) == 1:
        run_rl(run_args[0])
    else:
        n_processes = args.processes or len(AGENT_LIST)
        with Pool(n_processes) as pool:
            pool.map(run_rl, run_args)


if __name__ == '__main__':
    main()
