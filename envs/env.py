import itertools
import random

from .rl_negotiator import *
from .observer import *
from .domain_loader import load_genius_domain
from sao.my_sao import MySAOMechanism
from sao.my_negotiators import *

PENALTY = -1.

#ここは強化学習の環境や初期条件を設定する場所

class NaiveEnv(gym.Env):
    def __init__(
        self,
        domain: str = 'party',
        opponent: str = ['Boulware', 'Conceder'],
        test: bool = False,
        random_train: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.test = test
        self.domain_names = self._normalize_domains(domain)
        self.opponent_pairs = self._normalize_opponent_pairs(opponent)
        self.scenarios = list(itertools.product(self.domain_names, self.opponent_pairs))
        self.random_train = random_train
        self.scenario_index = -1

        #初期設定
        self.my_agent: Optional[RLNegotiator] = None
        self.session: Optional[MySAOMechanism] = None
        self.domain_name = None
        self.domain = None
        self.util1 = None
        self.util2 = None
        self.util3 = None
        self.opponent = None
        self.my_util = None
        self.opp_util1 = None
        self.opp_util2 = None
        self._select_scenario()
        self.scenario_index = -1

        self.state = None
        self.observation = None
        self.reward_range = [PENALTY, 1.0]
        self.seed()

    @staticmethod
    def _normalize_domains(domain):
        if isinstance(domain, str):
            return [domain]
        domains = list(domain)
        if not domains:
            raise ValueError('At least one domain is required')
        return domains

    @staticmethod
    def _normalize_opponent_pairs(opponent):
        if opponent is None:
            return [['Boulware', 'Conceder']]
        if isinstance(opponent, str):
            return [[opponent, opponent]]
        if len(opponent) == 2 and all(isinstance(agent, str) for agent in opponent):
            return [list(opponent)]

        pairs = []
        for pair in opponent:
            if isinstance(pair, str):
                pairs.append([pair, pair])
                continue
            pair = list(pair)
            if len(pair) != 2:
                raise ValueError('Each opponent setting must contain exactly two opponents')
            pairs.append(pair)
        if not pairs:
            raise ValueError('At least one opponent setting is required')
        return pairs

    def _cleanup_ufuns(self):
        for ufun in (self.my_util, self.opp_util1, self.opp_util2):
            if ufun is not None and hasattr(ufun, '_ami'):
                del ufun._ami

    def _select_scenario(self):
        if self.random_train and len(self.scenarios) > 1:
            domain_name, opponent = random.choice(self.scenarios)
        else:
            self.scenario_index = (self.scenario_index + 1) % len(self.scenarios)
            domain_name, opponent = self.scenarios[self.scenario_index]

        self.domain_name = domain_name
        self.domain, (self.util1, self.util2, self.util3) = load_genius_domain(domain_name)

        # 設定読み込み
        self.opponent = list(opponent)
        self.my_util = self.util3
        self.opp_util1 = self.util2
        self.opp_util2 = self.util1

    def get_opponent(self, add_noise=False):
        if self.opponent[self.agent_number] == 'Boulware':
            opponent = TimeBasedNegotiator(name = 'Boulware{}'.format(self.agent_number), aspiration_type=10.0, add_noise=add_noise)
        elif self.opponent[self.agent_number] == 'Linear':
            opponent = TimeBasedNegotiator(name='Linear{}'.format(self.agent_number), aspiration_type=1.0, add_noise=add_noise)
        elif self.opponent[self.agent_number] == 'Conceder':
            opponent = TimeBasedNegotiator(name='Conceder{}'.format(self.agent_number), aspiration_type=0.2, add_noise=add_noise)
        elif self.opponent[self.agent_number] == 'TitForTat1':
            opponent = AverageTitForTatNegotiator(name='TitForTat1{}'.format(self.agent_number), gamma=1, add_noise=add_noise)
        elif self.opponent[self.agent_number] == 'TitForTat2':
            opponent = AverageTitForTatNegotiator(name='TitForTat2{}'.format(self.agent_number), gamma=2, add_noise=add_noise)
        elif self.opponent[self.agent_number] == 'AgentK':
            opponent = AgentK(name='AgentK{}'.format(self.agent_number), add_noise=add_noise)
        elif self.opponent[self.agent_number] == 'HardHeaded':
            opponent = HardHeaded(name='HardHeaded{}'.format(self.agent_number), add_noise=add_noise)
        elif self.opponent[self.agent_number] == 'CUHKAgent':
            opponent = CUHKAgent(name='CUHKAgent{}'.format(self.agent_number), add_noise=add_noise)
        elif self.opponent[self.agent_number] == 'Atlas3':
            opponent = Atlas3(name='Atlas3{}'.format(self.agent_number), add_noise=add_noise)
        elif self.opponent[self.agent_number] == 'AgentGG':
            opponent = AgentGG(name='AgentGG{}'.format(self.agent_number), add_noise=add_noise)
        else:
            opponent = TimeBasedNegotiator(name='Linear', aspiration_type=1.0, add_noise=add_noise)
        return opponent

    def close(self):
        self._cleanup_ufuns()
        self.domain = None
        self.util1 = None
        self.util2 = None
        self.util3 = None
        self.my_util = None
        self.opp_util1 = None
        self.opp_util2 = None
        self.my_agent = None
        self.session = None

    def seed(self, seed=None):
        pass

    def get_reward(self):
        if self.state['timedout'] or self.state['broken']:
            if not self.test:
                return PENALTY
            else:
                return 0
        elif self.state['agreement'] is not None:
            return self.my_util(self.state['agreement'])
        else:
            return 0

class RLBOAEnv(NaiveEnv):
    """Three-party RLBOA environment over the stacked AOP/SAOP protocol.

    The learned action is the RLBOA target utility range, not a concrete bid.
    The BOA opponent-model strategy in RLBOANegotiator then maps that target
    range back to an outcome.
    """

    def __init__(
        self,
        domain='party',
        opponent=None,
        test=False,
        n_actions=10,
        add_noise=True,
        random_train=True,
        **kwargs,
    ):
        if opponent is None:
            opponent = ['Boulware', 'Conceder']

        super().__init__(domain, opponent, test=test, random_train=random_train, **kwargs)
        self.n_actions = n_actions
        self.add_noise = add_noise
        self.opponent_names = []
        self.observer = RLBOAObserve(self.domain, self.my_util)
        self.observation_space = self.observer.observation_space
        self.action_space = gym.spaces.Discrete(n_actions)

    def reset(self):
        # セッション，エージェントの作成
        self._cleanup_ufuns()
        if self.session is not None:
            if self.my_agent is not None and hasattr(self.my_agent, 'om'):
                del self.my_agent.om
            self.session.reset()

        self._select_scenario()
        self.observer = RLBOAObserve(self.domain, self.my_util)
        self.observation_space = self.observer.observation_space

        self.session = MySAOMechanism(issues=self.domain, n_steps=80, avoid_ultimatum=False)
        self.my_agent = RLBOANegotiator(n_ranges=self.n_actions)
        self.my_agent.name = 'RLAgent'

        self.agent_number = 0
        opponent0 = self.get_opponent(add_noise=self.add_noise)
        self.agent_number = 1
        opponent1 = self.get_opponent(add_noise=self.add_noise)
        self.opponent_names = [opponent0.name, opponent1.name]

        self.session.add(self.my_agent, ufun=self.my_util)
        self.session.add(opponent0, ufun=self.opp_util1)
        self.session.add(opponent1, ufun=self.opp_util2)

        self.my_agent.om = HardHeadedFrequencyModel(self.my_util)
        self.state = None

        self.observer.reset()
        self.observation = self.observer(self.state, self.opponent_names)
        return self.observation

    def step(self, action: int):
        self.action = int(action)
        self.my_agent.set_target(self.action)
        return self._advance_until_rl_turn()

    def _advance_until_rl_turn(self):
        while True:
            self.state = self.session.step().__dict__
            self.observation = self.observer(self.state, self.opponent_names)
            if self.state['agreement'] is not None:
                return self.observation, self.get_reward(), True, {}
            if self.state['timedout'] or self.state['broken']:
                return self.observation, self.get_reward(), True, {}
            if self._next_negotiator_is_rl():
                return self.observation, self.get_reward(), False, {}

    def _next_negotiator_is_rl(self):
        negotiators = self.session.negotiators
        if not negotiators:
            return False
        next_index = (self.session._last_checked_negotiator + 1) % len(negotiators)
        return negotiators[next_index] is self.my_agent
