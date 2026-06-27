import dill
dill.extend(False)
import gym
import sao
import sys
import os
import torch
from multiprocessing import Pool
from datetime import datetime
from gym import register

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
dill.extend(True)

ENV_NAME = 'RLBOAEnv-{}-{}-{}-v0'
ISSUE_NAMES = [
    'Laptop',
    'ItexvsCypress',
    'IS_BT_Acquisition',
    'Grocery',
]
AGENT_LIST = [
    'Boulware',
    'Linear',
    'Conceder',
]
SAVE_PATH = "./results/{}/".format(datetime.now().strftime('%Y%m%d-%H%M%S')[2:])


def register_neg_env(issue, agent):
    """
    登録された環境を作成します。
    agent はリスト形式で [0] と [1] にそれぞれ格納されている。
    """
    env_name = ENV_NAME.format(issue, agent[0], agent[1])
    register(
        id=env_name,
        entry_point='envs.env:RLBOAEnv',
        kwargs={'domain': issue, 'opponent': agent, 'is_first': True},
    )
    return env_name


def run_rl(args):
    """
    各対戦相手ごとのモデルを学習し、統合モデルを作成して保存する。
    """
    issue, agent = args  # タプルを展開して受け取る

    # 相手1に対する環境とモデル
    agent_buf = [agent[0], agent[0]]
    env_name_1 = register_neg_env(issue, agent_buf)
    f_name1 = env_name_1.split('-', maxsplit=1)[1]
    env1 = make_vec_env(env_name_1, n_envs=4)
    model1 = PPO("MlpPolicy", env1, verbose=1, device="cpu", tensorboard_log=SAVE_PATH)
    model1.learn(total_timesteps=100000, tb_log_name=f_name1)

    # 相手2に対する環境とモデル
    agent_buf = [agent[1], agent[1]]
    env_name_2 = register_neg_env(issue, agent_buf)
    f_name2 = env_name_2.split('-', maxsplit=1)[1]
    env2 = make_vec_env(env_name_2, n_envs=4)
    model2 = PPO("MlpPolicy", env2, verbose=1, device="cpu", tensorboard_log=SAVE_PATH)
    model2.learn(total_timesteps=100000, tb_log_name=f_name2)

    # モデル統合
    def merge_models(model1, model2, eval_env):
        """2つのモデルのパラメータを統合して新しいモデルを作成"""
        # 新しいモデルを初期化
        merged_model = PPO("MlpPolicy", eval_env, verbose=1, device="cpu")
        model1_params = model1.get_parameters()
        model2_params = model2.get_parameters()

        # 重みの平均を計算
        merged_params = {}
        for key in model1_params:
            # パラメータがテンソルであることを確認して平均化
            if isinstance(model1_params[key], torch.Tensor) and isinstance(model2_params[key], torch.Tensor):
                merged_params[key] = (model1_params[key] + model2_params[key]) / 2
            else:
                # テンソル以外の場合はmodel1の値をそのまま使用
                merged_params[key] = model1_params[key]

        # 統合したパラメータを新しいモデルに設定
        merged_model.set_parameters(merged_params)
        return merged_model

    # ここからメインの処理
    eval_env_name = register_neg_env(issue, agent)
    f_name = eval_env_name.split('-', maxsplit=1)[1]
    eval_env = gym.make(eval_env_name)

    # モデルを統合
    merged_model = merge_models(model1, model2, eval_env)

    # 統合モデルを保存
    merged_model.save(SAVE_PATH + f_name)

    # 統合モデルの評価
    eval_env.test = True
    mean_reward, std_reward = evaluate_policy(merged_model, eval_env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    with open(SAVE_PATH + "result.csv", "a") as f:
        f.write("{},{},{},{},{}\n".format(*eval_env_name.split('-')[1:4], mean_reward, std_reward))

    # 環境のクリーンアップ
    eval_env.close()
    del model1, model2, merged_model


def main():
    """
    メイン関数: 学習を並列処理で実行
    """
    os.makedirs(SAVE_PATH)
    with open(SAVE_PATH + "result.csv", "w") as f:
        f.write("domain,opponent1,opponent2,mean,std\n")

    p = Pool(len(AGENT_LIST))
    args = [
        (issue, [AGENT_LIST[i], AGENT_LIST[j]])
        for issue in ISSUE_NAMES
        for i in range(len(AGENT_LIST))
        for j in range(i, len(AGENT_LIST))
    ]
    p.map(run_rl, args)


if __name__ == '__main__':
    main()

'''
def main():
    os.makedirs(SAVE_PATH)
    with open(SAVE_PATH + "result.csv", "w") as f:
        f.write("domain,opponent1,opponent2,mean,std\n")

    p = Pool(len(AGENT_LIST))
    for issue in ISSUE_NAMES[:]:
        last_number = len(AGENT_LIST)
        agent = [None,None]
        for i in range(last_number):
            for j in range(i,last_number):
                agent[0] = AGENT_LIST[i]
                agent[1] = AGENT_LIST[j]
                p.map(run_rl, [(issue, agent)])

if __name__ == '__main__':
    main()
'''