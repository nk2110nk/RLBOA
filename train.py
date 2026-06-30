import argparse
import distutils.version
import os
from datetime import datetime
from itertools import combinations_with_replacement
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
GENERAL_ENV_NAME = 'RLBOA-General-v0'
MODEL_DIR_NAME = 'RLBOA_Negotiator'
CHECKPOINT_NAME = 'checkpoint.zip'
ISSUE_NAMES = [
    'Laptop',
    'ItexvsCypress',
    'IS_BT_Acquisition',
    'Grocery',
    'thompson',
    'Car',
    'EnergySmall_A',
    # 'Coffee',
    # 'Camera',
    # 'Lunch',
    # 'SmartPhone',
    # 'Kitchen',
    # 'Travel',
    # 'party',
]
AGENT_LIST = [
    'Boulware',
    'Linear',
    'Conceder',
    'Atlas3',
]


def register_neg_env(issue, agent, n_actions=10, add_noise=True):
    env_name = ENV_NAME.format(issue, agent[0], agent[1])
    try:
        register(
            id=env_name,
            entry_point='envs.env:RLBOAEnv',
            kwargs={
                'domain': issue,
                'opponent': agent,
                'n_actions': n_actions,
                'add_noise': add_noise,
            },
        )
    except GymError:
        # Gym raises if this id has already been registered in this process.
        pass
    return env_name


def register_general_env(issues, agent_pairs, n_actions=10, add_noise=True, random_train=True):
    try:
        register(
            id=GENERAL_ENV_NAME,
            entry_point='envs.env:RLBOAEnv',
            kwargs={
                'domain': issues,
                'opponent': agent_pairs,
                'n_actions': n_actions,
                'add_noise': add_noise,
                'random_train': random_train,
            },
        )
    except GymError:
        # Gym raises if this id has already been registered in this process.
        pass
    return GENERAL_ENV_NAME


def run_rl(args):
    (
        issue,
        agent,
        model_type,
        timesteps,
        n_envs,
        eval_episodes,
        n_actions,
        add_noise,
        save_path,
        checkpoint_name,
        random_train,
    ) = args

    if model_type == 'general':
        env_name = register_general_env(issue, agent, n_actions, add_noise, random_train=random_train)
        domain_label = '-'.join(issue)
        opponent_label = '_'.join('-'.join(pair) for pair in agent)
        f_name = f'General-{domain_label}_{opponent_label}-v0'
    else:
        env_name = register_neg_env(issue, agent, n_actions, add_noise)
        domain_label = issue
        opponent_label = f'{agent[0]}-{agent[1]}'
        f_name = env_name.split('-', maxsplit=1)[1]
    env = make_vec_env(env_name, n_envs=n_envs)

    model = PPO('MlpPolicy', env, verbose=1, device='cpu')
    is_single_checkpoint = checkpoint_name == CHECKPOINT_NAME
    model_file = checkpoint_name if is_single_checkpoint else f'{f_name}.zip'
    log_name = os.path.splitext(model_file)[0]
    log_path = os.path.join(save_path, log_name)
    model_path = os.path.join(save_path, model_file)
    model.set_logger(configure(log_path, ['stdout', 'tensorboard']))
    model.learn(total_timesteps=timesteps)
    model.save(model_path)
    print(f'saved_model:{model_path}')

    eval_env = gym.make(env_name)
    eval_env.test = True
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes)
    print(f'mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}')
    with open(os.path.join(save_path, 'result.csv'), 'a') as f:
        f.write(
            f'{model_type},{domain_label},{opponent_label},'
            f'{timesteps},{n_envs},{n_actions},{model_path},{mean_reward},{std_reward}\n'
        )

    env.close()
    eval_env.close()
    del model


def parse_args():
    parser = argparse.ArgumentParser(description='Train RLBOA in a three-party SAOP negotiation.')
    parser.add_argument('--agents', '-a', nargs='*', choices=AGENT_LIST)
    parser.add_argument('--issue', '-i', nargs='*', choices=ISSUE_NAMES)
    parser.add_argument('--save_path', '-sp', default='./results/')
    parser.add_argument('--domain', choices=ISSUE_NAMES)
    parser.add_argument('--opponent1', choices=AGENT_LIST)
    parser.add_argument('--opponent2', choices=AGENT_LIST)
    parser.add_argument('--timesteps', '-t', type=int, default=100000)
    parser.add_argument('--n-envs', '--n_envs', '-n', dest='n_envs', type=int, default=4)
    parser.add_argument('--eval-episodes', type=int, default=100)
    parser.add_argument('--n-actions', type=int, default=10)
    parser.add_argument('--no-noise', action='store_true')
    parser.add_argument('--processes', type=int, default=None)
    parser.add_argument(
        '--model-type',
        '--mode',
        choices=['expert', 'general'],
        default='expert',
        help='expert trains one checkpoint per domain/opponent pair; general trains one checkpoint over all selected settings',
    )
    parser.add_argument(
        '--ordered-train',
        action='store_true',
        help='In general mode, cycle through settings instead of sampling them randomly on reset',
    )
    return parser.parse_args()


def normalize_agents(agents):
    if not agents:
        return None
    if len(agents) == 1:
        return [agents[0], agents[0]]
    return agents


def build_agent_pairs(agents):
    agents = normalize_agents(agents)
    if len(agents) == 2:
        return [agents]
    return [list(pair) for pair in combinations_with_replacement(agents, 2)]


def build_default_save_root(issues, agents, current_time, save_path):
    if save_path != './results/':
        return save_path if save_path.endswith(os.sep) else save_path + os.sep
    return os.path.join(
        './results',
        f'{"-".join(issues)}_{"-".join(agents)}',
        f'{current_time}-TA',
    ) + os.sep


def build_jobs(args):
    if args.model_type == 'general':
        if not (args.issue and args.agents):
            raise ValueError('general mode requires --issue/-i and --agents/-a')
        return [(
            args.issue,
            build_agent_pairs(args.agents),
        )], args.issue, args.agents

    if args.issue or args.agents:
        if not (args.issue and args.agents):
            raise ValueError('--issue/-i and --agents/-a must be specified together')
        pairs = build_agent_pairs(args.agents)
        return [(issue, pair) for issue in args.issue for pair in pairs], args.issue, args.agents

    if args.domain or args.opponent1 or args.opponent2:
        if not (args.domain and args.opponent1 and args.opponent2):
            raise ValueError('--domain, --opponent1, and --opponent2 must be specified together')
        agents = [args.opponent1, args.opponent2]
        return [(args.domain, agents)], [args.domain], agents

    jobs = [
        (issue, [AGENT_LIST[i], AGENT_LIST[j]])
        for issue in ISSUE_NAMES
        for i in range(len(AGENT_LIST))
        for j in range(i, len(AGENT_LIST))
    ]
    return jobs, ISSUE_NAMES, AGENT_LIST


def main():
    args = parse_args()
    jobs, issues, agents = build_jobs(args)
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_root = build_default_save_root(issues, agents, current_time, args.save_path)
    save_path = os.path.join(run_root, MODEL_DIR_NAME)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'result.csv'), 'w') as f:
        f.write('model_type,domain,opponents,total_timesteps,n_envs,n_actions,checkpoint,mean,std\n')

    single_job = len(jobs) == 1

    run_args = [
        (
            issue,
            agent,
            args.model_type,
            args.timesteps,
            args.n_envs,
            args.eval_episodes,
            args.n_actions,
            not args.no_noise,
            save_path,
            CHECKPOINT_NAME if single_job else None,
            not args.ordered_train,
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
