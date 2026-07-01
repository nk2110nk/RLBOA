import argparse
import csv
import gc
import os
from itertools import combinations_with_replacement
from itertools import product
from multiprocessing import Pool

import dill
dill.extend(False)
from negmas import Issue, UtilityFunction
from envs.domain_loader import load_genius_domain
from sao.my_negotiators import *
from sao.my_sao import MySAOMechanism
from envs.rl_negotiator import TestRLBOANegotiator
from matplotlib import pyplot as plt
dill.extend(True)


ISSUE_NAMES = [
    'Laptop',
    'ItexvsCypress',
    'IS_BT_Acquisition',
    'Grocery',
    'thompson',
    'Car',
    'EnergySmall_A',
    'Coffee',
    'Camera',
    'Lunch',
    'SmartPhone',
    'Kitchen',
]
AGENT_LIST = [
    'Boulware',
    'Linear',
    'Conceder',
    'Atlas3',
]
CHECKPOINT_NAME = 'checkpoint.zip'


def bool_tag(x):
    return 'T' if x else 'F'


def model_name(issue, agent):
    return f'{issue}-{agent[0]}-{agent[1]}-v0.zip'


def resolve_model_path(load_path, issue, agent):
    normalized = os.path.normpath(load_path)
    if os.path.isfile(normalized):
        return normalized

    name = model_name(issue, agent)
    candidates = [
        os.path.join(normalized, CHECKPOINT_NAME),
        os.path.join(normalized, name),
        os.path.join(os.path.dirname(normalized), name),
        os.path.join(os.path.dirname(normalized), CHECKPOINT_NAME),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f'Model file not found. Tried: {", ".join(candidates)}'
    )


def resolve_save_path(load_path, issue, agent, plot):
    normalized = os.path.normpath(load_path)
    base = os.path.basename(normalized)
    model_stem = model_name(issue, agent).rsplit('.', maxsplit=1)[0]
    if os.path.isfile(normalized):
        normalized = os.path.dirname(normalized)
    elif base == model_stem or base.startswith(model_stem + '_'):
        normalized = os.path.dirname(normalized)
    return os.path.join(
        normalized,
        'img' if plot else 'csv',
        f'{agent[0]}-{agent[1]}',
        issue,
    )


def result_condition_path(save_root, det, noise):
    return os.path.join(save_root, f'det={det}_noise={noise}')


def get_opponent(opponent, agent_number=0, add_noise=False):
    if opponent == 'Boulware':
        return TimeBasedNegotiator(name=f'Boulware{agent_number}', aspiration_type=10.0, add_noise=add_noise)
    if opponent == 'Linear':
        return TimeBasedNegotiator(name=f'Linear{agent_number}', aspiration_type=1.0, add_noise=add_noise)
    if opponent == 'Conceder':
        return TimeBasedNegotiator(name=f'Conceder{agent_number}', aspiration_type=0.2, add_noise=add_noise)
    if opponent == 'TitForTat1':
        return AverageTitForTatNegotiator(name=f'TitForTat1{agent_number}', gamma=1, add_noise=add_noise)
    if opponent == 'TitForTat2':
        return AverageTitForTatNegotiator(name=f'TitForTat2{agent_number}', gamma=2, add_noise=add_noise)
    if opponent == 'AgentK':
        return AgentK(name=f'AgentK{agent_number}', add_noise=add_noise)
    if opponent == 'HardHeaded':
        return HardHeaded(name=f'HardHeaded{agent_number}', add_noise=add_noise)
    if opponent == 'CUHKAgent':
        return CUHKAgent(name=f'CUHKAgent{agent_number}', add_noise=add_noise)
    if opponent == 'Atlas3':
        return Atlas3(name=f'Atlas3{agent_number}', add_noise=add_noise)
    if opponent == 'AgentGG':
        return AgentGG(name=f'AgentGG{agent_number}', add_noise=add_noise)
    return TimeBasedNegotiator(name=f'Linear{agent_number}', aspiration_type=1.0, add_noise=add_noise)


def run_session(model_path, save_path, opponent, issue, det, noise, n_actions, plot):
    domain, (util1, util2, util3) = load_genius_domain(issue)

    session = MySAOMechanism(issues=domain, n_steps=80, avoid_ultimatum=False)
    opponent0 = get_opponent(opponent[0], agent_number=0, add_noise=noise)
    opponent1 = get_opponent(opponent[1], agent_number=1, add_noise=noise)
    my_agent = TestRLBOANegotiator(
        domain,
        model_path,
        [opponent0.name, opponent1.name],
        deterministic=det,
        n_ranges=n_actions,
    )
    
    my_util = util1
    opp_util1 = util2
    opp_util2 = util3
    session.add(my_agent, ufun=my_util)
    session.add(opponent0, ufun=opp_util1)
    session.add(opponent1, ufun=opp_util2)

    result = session.run()

    if plot:
        my_agent.name = 'Our Agent'
        plot_name = os.path.basename(model_path).rsplit('.', maxsplit=1)[0]
        session.plot(path=os.path.join(save_path, f'{plot_name}-d{bool_tag(det)}-n{bool_tag(noise)}.png'))
        plt.clf()
        plt.close()

    if result['agreement'] is not None:
        my_util = my_util(result['agreement'])
        opp_util1 = opp_util1(result['agreement'])
        opp_util2 = opp_util2(result['agreement'])
    else:
        my_util, opp_util1, opp_util2 = 0, 0, 0

    session.reset()
    for ufun in (util1, util2, util3):
        if hasattr(ufun, '_ami'):
            del ufun._ami
    del my_agent, opponent0, opponent1, session
    gc.collect()

    return [
        my_util,
        opp_util1,
        opp_util2,
        my_util + opp_util1 + opp_util2,
        my_util * opp_util1 * opp_util2,
        result['agreement'],
        result['step'],
    ]


def test_negotiator(config):
    issue, agent, det, noise, save_path, load_path, episodes, n_actions, plot = config
    model_path = resolve_model_path(load_path, issue, agent)
    print(f'loaded_model:{model_path}')
    results = [['my_util', 'opp_util1', 'opp_util2', 'social', 'nash', 'agreement', 'step']]

    for _ in range(episodes):
        results.append(
            run_session(model_path, save_path, agent, issue, det, noise, n_actions, plot)
        )

    if not plot:
        out_path = os.path.join(
            save_path,
            f'{issue}-{agent[0]}-{agent[1]}-d{bool_tag(det)}-n{bool_tag(noise)}.tsv',
        )
        with open(out_path, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(results)
        print(f'saved_result:{out_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained three-party RLBOA negotiators.')
    parser.add_argument('--agents', '-a', nargs='*', choices=AGENT_LIST)
    parser.add_argument('--issues', '--issue', '-i', nargs='*', choices=ISSUE_NAMES)
    parser.add_argument('--model', '-m')
    parser.add_argument('--load-path')
    parser.add_argument('--domain', choices=ISSUE_NAMES)
    parser.add_argument('--opponent1', choices=AGENT_LIST)
    parser.add_argument('--opponent2', choices=AGENT_LIST)
    parser.add_argument('--episodes', '-e', type=int, default=100)
    parser.add_argument('--n-actions', type=int, default=10)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--plot', '-p', action='store_true')
    parser.add_argument('--processes', type=int, default=None)
    parser.add_argument(
        '--model-type',
        '--mode',
        choices=['auto', 'expert', 'general'],
        default='auto',
        help='auto keeps the old behavior; general evaluates all combinations with replacement from --agents',
    )
    return parser.parse_args()


def normalize_agents(agents):
    if not agents:
        return None
    if len(agents) == 1:
        return [agents[0], agents[0]]
    return agents


def build_agent_pairs(agents, model_type='auto'):
    agents = normalize_agents(agents)
    if model_type == 'general':
        return [list(pair) for pair in combinations_with_replacement(agents, 2)]
    if len(agents) == 2:
        return [agents]
    return [list(pair) for pair in combinations_with_replacement(agents, 2)]


def build_jobs(args):
    if args.issues or args.agents:
        if not (args.issues and args.agents):
            raise ValueError('--issues/-i and --agents/-a must be specified together')
        pairs = build_agent_pairs(args.agents, args.model_type)
        return [(issue, pair) for issue in args.issues for pair in pairs]

    if args.domain or args.opponent1 or args.opponent2:
        if not (args.domain and args.opponent1 and args.opponent2):
            raise ValueError('--domain, --opponent1, and --opponent2 must be specified together')
        return [(args.domain, [args.opponent1, args.opponent2])]

    return [
        (issue, [AGENT_LIST[i], AGENT_LIST[j]])
        for issue in ISSUE_NAMES
        for i in range(len(AGENT_LIST))
        for j in range(i, len(AGENT_LIST))
    ]


def main():
    args = parse_args()
    load_path = args.model or args.load_path
    if not load_path:
        raise ValueError('--model/-m or --load-path must be specified')

    det_values = [False] if args.stochastic else [args.deterministic]
    noise_values = [args.noise]
    jobs = build_jobs(args)

    configs = [
        (
            issue,
            agent,
            det,
            noise,
            result_condition_path(resolve_save_path(load_path, issue, agent, args.plot), det, noise),
            load_path,
            args.episodes,
            args.n_actions,
            args.plot,
        )
        for issue, agent in jobs
        for det, noise in product(det_values, noise_values)
    ]
    for config in configs:
        os.makedirs(config[4], exist_ok=True)

    if len(configs) == 1:
        test_negotiator(configs[0])
    else:
        n_processes = args.processes or len(AGENT_LIST)
        with Pool(n_processes) as pool:
            pool.map(test_negotiator, configs)


if __name__ == '__main__':
    main()
