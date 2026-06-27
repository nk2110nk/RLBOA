import dill
dill.extend(False)
import gym
import sao
import sys
import os
from multiprocessing import Pool
from datetime import datetime
from gym import register

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
dill.extend(True)

ENV_NAME = 'RLBOAnv-{}-{}-{}-v0'
ISSUE_NAMES = [
    'Laptop',
    'ItexvsCypress',
    'IS_BT_Acquisition',
    'Grocery',
    #'thompson',
    #'Car',
    #'EnergySmall_A'
]
AGENT_LIST = [
    'Boulware',
    'Linear',
    'Conceder',
    #'TitForTat1',
    #'TitForTat2',
    #"AgentK",
    #"HardHeaded",
    #"Atlas3",
    #"AgentGG",
]
SAVE_PATH = "./results/{}/".format(datetime.now().strftime('%Y%m%d-%H%M%S')[2:])


def register_neg_env(issue, agent):
    env_name = ENV_NAME.format(issue, agent[0], agent[1])
    register(
        id=env_name,
        entry_point='envs.env:RLBOAEnv',
        kwargs={'domain': issue, 'opponent': agent, 'is_first': True},
    )
    return env_name


def run_rl(args):
    issue, agent = args
    env_name = register_neg_env(issue, agent)
    f_name = env_name.split('-', maxsplit=1)[1]
    env = make_vec_env(env_name, n_envs=4)

    #ここで学習を行っている(環境などはenvに入っており,それらも引数として渡している)
    model = PPO("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log=SAVE_PATH)
    model.learn(total_timesteps=500000, tb_log_name=f_name)
    model.save(SAVE_PATH + f_name)

    # Use a separate environement for evaluation
    eval_env = gym.make(env_name)
    eval_env.test = True
    # Random Agent, before training
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    with open(SAVE_PATH + "result.csv", "a") as f:
        f.write("{},{},{},{},{}\n".format(*env_name.split('-')[1:4], mean_reward, std_reward))

    env.close()
    eval_env.close()
    del model


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
