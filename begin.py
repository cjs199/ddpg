# pip install gym==0.18.0 -i  https://pypi.douban.com/simple/
import random
import time
import numpy as np
from env import ContinuousCartPoleEnv
from all_model import *
from replay_memory import *
from tensorflow.keras import optimizers, losses
import cjs_util


GAMMA = 0.99  # reward 的衰减因子
REWARD_SCALE = 0.1  # reward 缩放系数
NOISE = 0.05  # 动作噪声方差
TRAIN_EPISODE = 6e3  # 训练的总episode数
MEMORY_SIZE = 100000  # 经验池大小
MEMORY_WARMUP_SIZE = 384  # MEMORY_SIZE // 20  # 预存一部分经验之后再开始训练
BATCH_SIZE = 384
ACTOR_LR = 1e-3  # Actor网络的 learning rate
CRITIC_LR = 1e-3  # Critic网络的 learning rate
TAU = 0.02  # 软更新的系数
log_str = ''  # 日志
calc_reward = 30
calc_reward_bfb = 0.005
actor_optimizers = optimizers.Adam(ACTOR_LR)
critic_optimizers = optimizers.Adam(CRITIC_LR)


def main():
    # 创建游戏
    env = ContinuousCartPoleEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # 创建模型
    model = DDPGModel(act_dim)
    target_model = DDPGModel(act_dim)

    # 同步model和target_model两个模型的权重
    target_model.actor_model.set_weights(
        model.actor_model.get_weights())
    target_model.critic_model.set_weights(
        model.critic_model.get_weights())

    # 创建经验池
    rpm = ReplayMemory(MEMORY_SIZE)

    # 循环调用进行训练
    while True:

        # 运行收集数据
        total_reward = run_episode(env, rpm, model, target_model)

        # 将收集的数据进行计算,追加日志,分析模型表现优劣
        appanyd_log(total_reward)

        # 数据足够以后,开始更新学习
        learn(rpm, model, target_model)

# 将游戏数据处理好后追加到日志文件中


def appanyd_log(total_reward):
    global calc_reward, log_str
    # 计算加权分数,并收集日志
    calc_reward = calc_reward * \
        (1-calc_reward_bfb) + total_reward * calc_reward_bfb

    if len(log_str) > 2000:

        log_str += cjs_util.add_append_str('得分 : ' + str(total_reward), 0, 10) + cjs_util.add_append_str(' ,加权得分 : ' +
                                                                                                         str(calc_reward), 0, 20) + '\r'
        cjs_util.appand_log('./ddpg/train.log', log_str)
        log_str = ""
    else:
        log_str += cjs_util.add_append_str('得分 : ' + str(total_reward), 0, 10) + cjs_util.add_append_str(' ,加权得分 : ' +
                                                                                                         str(calc_reward), 0, 20) + '\r'

# 运行游戏,收集数据


def run_episode(env, rpm, model: DDPGModel, target_model: DDPGModel):
    obs = env.reset()
    total_reward = 0
    steps = 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = model.actor_call(batch_obs)
        # 增加探索扰动, 输出限制在 [-1.0, 1.0] 范围内
        action = np.clip(np.random.normal(action, NOISE), -1.0, 1.0)
        next_obs, reward, done, info = env.step(action)
        # env.render()
        action = action[0]  # 方便存入replaymemory
        rpm.append((obs, action, REWARD_SCALE * reward, next_obs, done))
        obs = next_obs
        total_reward += reward
        if done or (steps >= 3000):
            break
    return total_reward

# 软更新权重


def update_target_model(model: DDPGModel, target_model: DDPGModel):
    actor_weights = np.array(model.actor_model.get_weights(), dtype=object)
    critic_weights = np.array(
        model.critic_model.get_weights(), dtype=object)
    actor_target_weights = np.array(
        target_model.actor_model.get_weights(), dtype=object)
    critic_target_weights = np.array(
        target_model.critic_model.get_weights(), dtype=object)

    actor_target_weights = actor_target_weights * \
        TAU + (1 - TAU) * actor_weights

    critic_target_weights = critic_target_weights * \
        TAU + (1 - TAU) * critic_weights

    target_model.actor_model.set_weights(actor_target_weights)
    target_model.critic_model.set_weights(critic_target_weights)

# 模型学习更新入口


def learn(rpm, model: DDPGModel, target_model: DDPGModel):
    # 用DDPG算法更新 actor 和 critic
    if len(rpm) > BATCH_SIZE:
        # 随机抽取一批数据
        (batch_obs, batch_action, batch_reward,
         batch_next_obs, batch_done) = rpm.sample(BATCH_SIZE)
        # 演员更新学习
        _actor_learn(model, target_model, batch_obs)
        # 评论家更新学习
        _critic_learn(model, target_model, batch_obs, batch_action, batch_reward, batch_next_obs,
                      batch_done)
        # 平滑更新权重参数
        update_target_model(model, target_model)


# 演员更新学习


def _actor_learn(model: DDPGModel, target_model: DDPGModel, batch_obs):
    with tf.GradientTape() as tape:
        action = model.actor_call(batch_obs)
        Q = model.critic_call(batch_obs, action)
        cost = tf.reduce_mean(-1.0 * Q)
    gradient_ = tape.gradient(cost, model.actor_model.trainable_weights)
    actor_optimizers.apply_gradients(
        zip(gradient_, model.actor_model.trainable_weights))

# 评论家更新学习


def _critic_learn(model: DDPGModel, target_model: DDPGModel, batch_obs, batch_action, batch_reward, batch_next_obs,
                  batch_done):
    with tf.GradientTape() as tape:
        next_action = target_model.actor_call(batch_next_obs)
        next_Q = target_model.critic_call(batch_next_obs, next_action)
        batch_done = tf.cast(batch_done, dtype='float32')
        target_Q = batch_reward + (1.0 - batch_done) * GAMMA * next_Q
        Q = model.critic_call(batch_obs, batch_action)
        cost = losses.mse(Q, target_Q)
    gradient_ = tape.gradient(cost, model.critic_model.trainable_weights)
    critic_optimizers.apply_gradients(
        zip(gradient_, model.critic_model.trainable_weights))


# 代码执行入口,放在最后,方便前面函数属性初始化
if __name__ == '__main__':
    main()
