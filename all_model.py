import tensorflow as tf
from tensorflow.keras import layers, Model

# 演员模型


class ActorModel(Model):
    def __init__(self, act_dim):
        super(ActorModel, self).__init__()
        self.fc1 = layers.Dense(100, activation='relu')
        self.fc2 = layers.Dense(act_dim, activation='tanh')

    def call(self, obs):
        hid = self.fc1(obs)
        means = self.fc2(hid)
        return means

# 评论家模型


class CriticModel(Model):
    def __init__(self):
        super(CriticModel, self).__init__()
        self.fc1 = layers.Dense(100, activation='relu')
        self.fc2 = layers.Dense(1, activation=None)

    def call(self, obs, act):
        concat = tf.concat([obs, act], axis=1)
        hid = self.fc1(concat)
        Q = self.fc2(hid)
        Q = tf.squeeze(Q, axis=[1])
        return Q

# DDPG算法对象,包裹演员和评论家模型,并操作使用对应的模型执行


class DDPGModel:
    def __init__(self, act_dim):
        super(DDPGModel, self).__init__()
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def actor_call(self, obs):
        return self.actor_model(obs)

    def critic_call(self, obs, act):
        return self.critic_model(obs, act)
