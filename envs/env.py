import io
import sys

from negmas import UtilityFunction, Issue

from .rl_negotiator import *
from .observer import *
from sao.my_sao import MySAOMechanism
from sao.my_negotiators import *

PENALTY = -1.

#ここは強化学習の環境や初期条件を設定する場所

class NaiveEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, domain: str = 'party', opponent: str = ['Boulware', 'Conceder'], is_first: bool = False, test: bool = False):
        super().__init__()
        self.test = test
        
        # ドメイン読み込み
        self.domain, _ = Issue.from_genius('domain/' + domain + '/domain.xml')
        self.util1, _ = UtilityFunction.from_genius('domain/' + domain + '/utility1.xml')
        self.util2, _ = UtilityFunction.from_genius('domain/' + domain + '/utility2.xml')
        self.util3, _ = UtilityFunction.from_genius('domain/' + domain + '/utility3.xml')

        #初期設定
        self.my_agent: Optional[RLNegotiator] = None
        self.session: Optional[MySAOMechanism] = None
        
        # ここは後でどうするか相談(6パターンやるしかなくなった)→一旦はこのままでやってみる
        # 設定読み込み
        self.opponent = opponent
        self.is_first = is_first
        if self.is_first:
            self.my_util = self.util3
            self.opp_util1 = self.util2
            self.opp_util2 = self.util1
        else:
            self.my_util = self.util3
            self.opp_util1 = self.util2
            self.opp_util2 = self.util1

        # 強化学習関連
        self.state = None
        self.action = None
        self.observation = None
    

        # self.observer = OneHotObserve2n(self.domain, 20)
        self.observer = OnehotObserve2nT(self.domain, 6)
        # self.observer = OpponentObserve1(self.domain)
        self.all_bids = self.get_all_bids()
        self.observation_space = self.observer.observation_space
        self.action_space = gym.spaces.Discrete(len(self.all_bids)) #全ての提案の大きさの空間を作っておく
        self.reward_range = [PENALTY, 1.0]
        self.seed()

    def reset(self):
        # セッション，エージェントの作成
        del self.my_util._ami
        del self.opp_util1._ami
        del self.opp_util2._ami

        if self.session is not None:
            self.session.reset()
        # self.session.__init__(issues=self.domain, n_steps=80, avoid_ultimatum=False)
        self.session = MySAOMechanism(issues=self.domain, n_steps=80, avoid_ultimatum=False)
        self.my_agent = RLNegotiator()

        self.agent_number = 0
        opponent0 = self.get_opponent(add_noise=True)
        self.agent_number = 1
        opponent1 = self.get_opponent(add_noise=True)

        # セッションにエージェントの追加
        if self.is_first:
            self.session.add(self.my_agent, ufun=self.my_util)
            self.session.add(opponent0, ufun=self.opp_util1)
            self.session.add(opponent1, ufun=self.opp_util2)
            self.state = None
        else:
            self.session.add(opponent0, ufun=self.opp_util1)
            self.session.add(self.my_agent, ufun=self.my_util)
            self.session.add(opponent1, ufun=self.opp_util2)
            # 後攻だったら相手に1回提案させる
            self.state = self.session.step().__dict__ #もしmy_agentが最後なら2回進める必要があるかも

        self.observer.reset()
        self.observation = self.observer(self.state,self.opponent)

        return self.observation

    def step(self, action: int):
        self.action = self.all_bids[action]
        self.my_agent.set_next_bid(self.action)
        
        # 最初にstepを一回進めておく
        self.state = self.session.step().__dict__
        # 状態を更新
        self.observation = self.observer(self.state,self.opponent)
        if self.state['agreement'] is not None:  # 合意していたら
            return self.observation, self.get_reward(), True, {}
        if self.state['timedout'] or self.state['broken']:
            return self.observation, self.get_reward(), True, {}
        
        # もし次の提案者が自分だった場合,もう一度stepを一回進める
        while self.session._current_proposer.name == 'RLAgent':
            self.state = self.session.step().__dict__
            # 状態を更新
            self.observation = self.observer(self.state,self.opponent)
            if self.state['agreement'] is not None:  # 合意していたら
                return self.observation, self.get_reward(), True, {}
            if self.state['timedout'] or self.state['broken']:
                return self.observation, self.get_reward(), True, {}
        return self.observation, self.get_reward(), False, {}
 
 
    def render(self, mode='human', close=False):
        outfile = io.StringIO() if mode == 'ansi' else sys.stdout
        if not self.state['running']:
            # outfile.write(
            #     '\nsteps:' + str(self.state['step']) + ', util:' +
            #     str(self.get_reward()) + '\n'
            # )
            self.session.plot()
            # self.make_log()
        return outfile

    def make_log(self):
        print('step,message,proposer,' + ','.join([issue.name for issue in self.domain]))
        
        for state in self.session.history:
            if state.agreement is None:
                print(f'{state.step},offer,{state.current_proposer},{self.bid2str(state.current_offer)}')
            else:
                print(f'{state.step},accept,{state.current_proposer},{self.bid2str(state.agreement)}')

    def bid2str(self, bid, onehot=False):
        if onehot:
            return ','.join([''.join(['1' if bid[issue.name] == value else '0' for value in issue.values]) for issue in
                             self.domain])
        else:
            return ','.join([bid[issue.name] for issue in self.domain])

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
        del self.my_util
        del self.opp_util1
        del self.my_agent
        del self.all_bids
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

    def get_all_bids(self):
        session = MySAOMechanism(issues=self.domain, n_steps=80, avoid_ultimatum=False)
        agent = RLNegotiator()
        session.add(agent, ufun=self.util3 if self.is_first else self.util3) #ここは状況に応じて変えなければならない
        return agent.all_bids


class AOPEnv(NaiveEnv):
    def __init__(self, domain='party', opponent=None, is_first=False, test=False):
        if opponent is None:
            opponent = ['Boulware', 'Conceder']  # デフォルトの2人

        super().__init__(domain, opponent, is_first, test)

        #各変数の設定
        self.opponent_list = opponent  # 対戦相手リスト
        self.action_space = gym.spaces.Discrete(len(self.all_bids) + 1)

        #stateの部分も初期化しておく
        self.opponent_type = self.opponent[0]
        


    def step(self, action: int):
        if action == len(self.all_bids):
            if self.state is None:
                self.state = {'broken': True, 'timedout': False, 'opponent_type': self.opponent_type} 
            else:
                #改良部分
                self.opponent_type = self.session._current_proposer.name
                print(self.opponent_type)

                self.state = {k: self.state['current_offer'] if k == 'agreement' else False if k == 'running' else v for
                              (k, v) in self.state.items()}
                self.observation = self.observer(self.state,self.opponent) #1

                #改良部分
                self.state['opponent_type'] = self.opponent_type
                
            return self.observation, self.get_reward(), True, {}
        else:
            return super().step(action)



class DenseEnv(AOPEnv):
    def get_reward(self):
        if self.state['timedout'] or self.state['broken']:
            if not self.test:
                return PENALTY
            else:
                return 0
        elif self.state['agreement'] is not None:
            return self.my_util(self.state['agreement'])
        else:   # ここを変える
            return self.my_util(self.state['current_offer']) / 100


class IssueActionEnv(AOPEnv):
    is_acceptable = True

    def __init__(self, domain='party', opponent='Boulware', is_first=False, test=False):
        super().__init__(domain, opponent, is_first, test)
        if self.is_acceptable:
            # [issue, (accept, (end), reject)]
            self.action_space = gym.spaces.MultiDiscrete([len(i.values) for i in self.domain] + [2])
        else:
            self.action_space = gym.spaces.MultiDiscrete([len(i.values) for i in self.domain])

    def step(self, action: list):
        if self.is_acceptable:
            if action[-1] == 0:    # accept
                if self.state is None:
                    self.state = {'broken': True, 'timedout': False}
                else:
                    self.state = {k: self.state['current_offer'] if k == 'agreement' else False if k == 'running' else v for
                                  (k, v) in self.state.items()}
                    self.observation = self.observer(self.state)
                return self.observation, self.get_reward(), True, {}
            # elif action[-1] == 1:  # end
            #     self.state = {'broken': True, 'timedout': False}
            #     return self.observation, self.get_reward(), True, {}
            else:                       # reject
                self.action = {i.name: i.values[v] for i, v in zip(self.domain, action)}
                self.my_agent.set_next_bid(self.action)
                for _ in range(2):
                    self.state = self.session.step().__dict__
                    # 状態を更新
                    self.observation = self.observer(self.state)
                    if self.state['agreement'] is not None:  # 合意していたら
                        return self.observation, self.get_reward(), True, {}
                    if self.state['timedout'] or self.state['broken']:
                        return self.observation, self.get_reward(), True, {}
                return self.observation, self.get_reward(), False, {}
        else:
            self.action = {i.name: i.values[v] for i, v in zip(self.domain, action)}
            self.my_agent.set_next_bid(self.action)
            for _ in range(2):
                self.state = self.session.step().__dict__
                # 状態を更新
                self.observation = self.observer(self.state)
                if self.state['agreement'] is not None:  # 合意していたら
                    return self.observation, self.get_reward(), True, {}
                if self.state['timedout'] or self.state['broken']:
                    return self.observation, self.get_reward(), True, {}
            return self.observation, self.get_reward(), False, {}


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
        is_first=True,
        test=False,
        n_actions=10,
        rl_position=None,
        add_noise=True,
    ):
        if opponent is None:
            opponent = ['Boulware', 'Conceder']
        if isinstance(opponent, str):
            opponent = [opponent, opponent]
        if len(opponent) != 2:
            raise ValueError('RLBOAEnv expects exactly two opponents for a three-party negotiation')

        super().__init__(domain, opponent, is_first, test)
        self.n_actions = n_actions
        self.rl_position = 0 if rl_position is None and is_first else 1 if rl_position is None else rl_position
        if self.rl_position not in (0, 1, 2):
            raise ValueError('rl_position must be 0, 1, or 2')
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

        participants = [
            (self.my_agent, self.my_util),
            (opponent0, self.opp_util1),
            (opponent1, self.opp_util2),
        ]
        if self.rl_position == 1:
            participants = [participants[1], participants[0], participants[2]]
        elif self.rl_position == 2:
            participants = [participants[1], participants[2], participants[0]]

        for agent, ufun in participants:
            self.session.add(agent, ufun=ufun)

        self.my_agent.om = HardHeadedFrequencyModel(self.my_util)
        self.state = None

        self.observer.reset()
        if self.rl_position != 0:
            self.observation, _, done, _ = self._advance_until_rl_turn()
            if done:
                return self.observation
        else:
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
