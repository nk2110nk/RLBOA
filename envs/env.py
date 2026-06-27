from .rl_negotiator import *
from .observer import *
from .domain_loader import load_genius_domain
from sao.my_sao import MySAOMechanism
from sao.my_negotiators import *

PENALTY = -1.

#ここは強化学習の環境や初期条件を設定する場所

class NaiveEnv(gym.Env):
    def __init__(self, domain: str = 'party', opponent: str = ['Boulware', 'Conceder'], test: bool = False, **kwargs):
        super().__init__()
        self.test = test
        
        # ドメイン読み込み
        self.domain, (self.util1, self.util2, self.util3) = load_genius_domain(domain)

        #初期設定
        self.my_agent: Optional[RLNegotiator] = None
        self.session: Optional[MySAOMechanism] = None
        
        # 設定読み込み
        self.opponent = opponent
        self.my_util = self.util3
        self.opp_util1 = self.util2
        self.opp_util2 = self.util1

        self.state = None
        self.observation = None
        self.reward_range = [PENALTY, 1.0]
        self.seed()

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
        del self.domain
        del self.util1
        del self.util2
        del self.util3
        del self.my_util
        del self.opp_util1
        del self.opp_util2
        del self.my_agent
        del self.session

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
        **kwargs,
    ):
        if opponent is None:
            opponent = ['Boulware', 'Conceder']
        if isinstance(opponent, str):
            opponent = [opponent, opponent]
        if len(opponent) != 2:
            raise ValueError('RLBOAEnv expects exactly two opponents for a three-party negotiation')

        super().__init__(domain, opponent, test=test, **kwargs)
        self.n_actions = n_actions
        self.add_noise = add_noise
        self.opponent_names = []
        self.observer = RLBOAObserve(self.domain, self.my_util)
        self.observation_space = self.observer.observation_space
        self.action_space = gym.spaces.Discrete(n_actions)

    def reset(self):
        # セッション，エージェントの作成
        for ufun in (self.my_util, self.opp_util1, self.opp_util2):
            if hasattr(ufun, '_ami'):
                del ufun._ami
        if self.session is not None:
            if self.my_agent is not None and hasattr(self.my_agent, 'om'):
                del self.my_agent.om
            self.session.reset()

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
