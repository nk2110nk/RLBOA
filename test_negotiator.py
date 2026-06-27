import argparse
import csv
import gc
import os
from itertools import product
from multiprocessing import Pool

import dill
dill.extend(False)
from negmas import Issue, UtilityFunction
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
        os.path.join(normalized, name),
        os.path.join(os.path.dirname(normalized), name),
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
    if base == model_stem or base.startswith(model_stem + '_'):
        normalized = os.path.dirname(normalized)
    return os.path.join(normalized, 'img' if plot else 'csv')


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


def add_participants(session, my_agent, opponent0, opponent1, util1, util2, util3, rl_position):
    participants = [
        (my_agent, util1),
        (opponent0, util2),
        (opponent1, util3),
    ]
    if rl_position == 1:
        participants = [participants[1], participants[0], participants[2]]
    elif rl_position == 2:
        participants = [participants[1], participants[2], participants[0]]
    for agent, ufun in participants:
        session.add(agent, ufun=ufun)


def run_session(model_path, save_path, opponent, issue, det, noise, rl_position, n_actions, plot):
    domain, _ = Issue.from_genius('./domain/' + issue + '/domain.xml')
    util1, _ = UtilityFunction.from_genius('./domain/' + issue + '/utility1.xml')
    util2, _ = UtilityFunction.from_genius('./domain/' + issue + '/utility2.xml')
    util3, _ = UtilityFunction.from_genius('./domain/' + issue + '/utility3.xml')

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
    add_participants(session, my_agent, opponent0, opponent1, util1, util2, util3, rl_position)

    result = session.run()

    if plot:
        my_agent.name = 'Our Agent'
        plot_name = os.path.basename(model_path).rsplit('.', maxsplit=1)[0]
        session.plot(path=os.path.join(save_path, f'{plot_name}-d{bool_tag(det)}-n{bool_tag(noise)}.png'))
        plt.clf()
        plt.close()

    if result['agreement'] is not None:
        my_util = util1(result['agreement'])
        opp_util1 = util2(result['agreement'])
        opp_util2 = util3(result['agreement'])
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
    issue, agent, det, noise, save_path, load_path, episodes, rl_position, n_actions, plot = config
    model_path = resolve_model_path(load_path, issue, agent)
    print(f'loaded_model:{model_path}')
    results = [['my_util', 'opp_util1', 'opp_util2', 'social', 'nash', 'agreement', 'step']]

    for _ in range(episodes):
        results.append(
            run_session(model_path, save_path, agent, issue, det, noise, rl_position, n_actions, plot)
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
    parser.add_argument('--load-path', default='./results/260627-133356/')
    parser.add_argument('--domain', choices=ISSUE_NAMES)
    parser.add_argument('--opponent1', choices=AGENT_LIST)
    parser.add_argument('--opponent2', choices=AGENT_LIST)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--rl-position', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--n-actions', type=int, default=10)
    parser.add_argument('--deterministic', action='store_true', default=True)
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--processes', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    load_path = args.load_path

    det_values = [False] if args.stochastic else [args.deterministic]
    noise_values = [args.noise]

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

    save_path = resolve_save_path(load_path, jobs[0][0], jobs[0][1], args.plot)
    os.makedirs(save_path, exist_ok=True)

    configs = [
        (
            issue,
            agent,
            det,
            noise,
            save_path,
            load_path,
            args.episodes,
            args.rl_position,
            args.n_actions,
            args.plot,
        )
        for issue, agent in jobs
        for det, noise in product(det_values, noise_values)
    ]

    if len(configs) == 1:
        test_negotiator(configs[0])
    else:
        n_processes = args.processes or len(AGENT_LIST)
        with Pool(n_processes) as pool:
            pool.map(test_negotiator, configs)


if __name__ == '__main__':
    main()
