import utils
import pandas as pd
import pathlib
from itertools import count
from collections import deque
import numpy as np
import random

# DEBUG Config
D_CREATE_MODEL = 1


PATH_CWD = pathlib.Path.cwd()
PATH_STORAGE = PATH_CWD.joinpath('data')
PATH_DATA_AAPL = PATH_STORAGE.joinpath('AAPL.csv')
print('Data path:', PATH_DATA_AAPL)

data = pd.read_csv(PATH_DATA_AAPL)
data = utils.data.auto.data_auto(data)
print('Data shape:', data.shape)


if (D_CREATE_MODEL):
    import tensorflow as tf
    def get_model():
        input_1 = tf.keras.layers.Input(shape=(8,))
        dense_1 = tf.keras.layers.Dense(32, activation='relu')
        dense_2 = tf.keras.layers.Dense(32, activation='relu')
        LV = tf.keras.layers.Dense(1, activation='relu')
        LA = tf.keras.layers.Dense(3, activation='relu')

        x = input_1
        x = dense_1(x)
        x = dense_2(x)
        V = LV(x)
        A = LA(x)

        Q = V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True))

        return tf.keras.Model(inputs=[input_1], outputs=[Q]),\
               tf.keras.Model(inputs=[input_1], outputs=[A])
    q_network, advantage = get_model()
    q_network.compile(optimizer='adam', loss='mse')
    target_network, _ = get_model()
    target_network.set_weights(q_network.get_weights())


q_network.summary()
advantage.summary()

memory = deque(maxlen=2000)
ARANGE_32 = np.arange(32)

env = utils.env.wrapper.StockerV0(data)

while True:
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        if random.random() < 0.2:
            action = random.randint(0, 2)
        else:
            action = advantage.predict(state.reshape([1, 8])).argmax()

        old_state = state
        state, reward, done = env.step(action)
        memory.append([old_state, action, reward, state, done])
        total_reward += reward

        if len(memory) > 32:
            batch = random.sample(memory, 32)

            s = np.array([i[0] for i in batch])
            a = np.array([i[1] for i in batch])
            r = np.array([i[2] for i in batch])
            s2 = np.array([i[3] for i in batch])
            d = np.array([i[4] for i in batch])

            target = target_network.predict(s)
            pred_next = target_network.predict(s2)
            select_next = q_network.predict(s2).argmax(axis=1)

            target[ARANGE_32, a] = r + 0.99 * pred_next[ARANGE_32, select_next]

            q_network.fit(s, target, epochs=1, verbose=2)

            '''
            batch = random.sample(memory, 32)
            for i in batch:
                if not i[4]:
                    tg = target_network.predict(i[0])
                    pred = target_network.predict(i[3])
                    selected = q_network.predict(i[3]).argmax()
                    tg[0, i[1]] = i[2] + 0.99 * pred[0, selected]
                else:
                    tg[0, i[1]] = i[2]
                
                q_network.fit(i[0], tg, verbose=2)
            '''
    
    target_network.set_weights(q_network.get_weights())
    print(total_reward)
