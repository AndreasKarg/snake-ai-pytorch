import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Concatenate, Input
import numpy as np


class Linear_QNet():
    def __init__(self, input_size, hidden_size, grid_size, output_size, lr):
        # self.linear1 = nn.Linear(input_size, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, output_size)
        self.input_size = input_size
        self.output_size = output_size

        state_input = Input(shape=(input_size,), name='x1_input')
        x1 = Dense(hidden_size, activation="relu")(state_input)

        grid_input = Input(shape=grid_size, name='x2_input')
        x2 = Conv2D(16, 3, padding='valid', strides=1,
                    activation='relu')(grid_input)
        x2 = Conv2D(32, 3, padding='same', strides=2, activation='relu')(x2)
        x2 = Flatten()(x2)
        x2 = Dense(hidden_size, activation='relu')(x2)
        x2 = Dense(hidden_size, activation='relu')(x2)

        # combined = Concatenate()([grid_input, x2])

        output = Dense(output_size, activation='linear')(x2)

        # self.model = Model(inputs=[state_input, grid_input], outputs=output)
        self.model = Model(inputs=grid_input, outputs=output)
        self.model.compile(loss="mean_squared_error",
                           optimizer=Adam(lr=lr),
                           metrics=["accuracy"])

    def predict(self, state):
        return self.model(state[1])

    def predict_one(self, state):
        return self.model(state[1][np.newaxis, ...])[0]

    def __call__(self, state):
        return self.model(state)

    # def forward(self, x):
    #     x = F.relu(self.linear1(x))
    #     x = self.linear2(x)
    #     return x

    def save(self, file_name='model.h5'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        #torch.save(self.state_dict(), file_name)
        self.model.save(file_name)
        print(f"Model Saved: {file_name}")

    def load(self, file_name='model.h5'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        self.model.load_weights(file_name)
        print(f"Model Loaded: {file_name}")


class QTrainer:
    def __init__(self, model, gamma):
        self.gamma = gamma
        self.model = model

    def train_step(self, state, action, reward, next_state, done):
        state0, state1 = zip(*state)
        state0 = np.array(state0, dtype=np.float)
        state1 = np.array(state1, dtype=np.float)

        next_state0, next_state1 = zip(*next_state)
        next_state0 = np.array(next_state0, dtype=np.float)
        next_state1 = np.array(next_state1, dtype=np.float)

        action = np.array(action, dtype=np.long)
        reward = np.array(reward, dtype=np.float)
        done = np.array(done)
        # (n, x)

        if len(state1.shape) == 1:
            # (1, x)
            state0 = np.expand_dims(state0, axis=0)
            state1 = np.expand_dims(state1, axis=0)
            next_state0 = np.expand_dims(next_state0, axis=0)
            next_state1 = np.expand_dims(next_state1, axis=0)
            action = np.expand_dims(action, axis=0)
            reward = np.expand_dims(reward, axis=0)
            done = np.expand_dims(done, axis=0)

        # 1: predicted Q values with current state
        pred = self.model(state1)

        target = np.copy(pred)

        indexes = np.arange(state0.shape[0])
        max_next_q = np.amax(self.model(next_state1), axis=1)
        action_index = np.argmax(action, axis=1)
        target[indexes, action_index[indexes]] = reward + \
            self.gamma * np.logical_not(done) * max_next_q

        hist = self.model.model.fit(state1, target, verbose=0)

        print(
            f"Accuracy: {hist.history['accuracy'][0]:0.6f} Loss:{hist.history['loss'][0]:0.6f}")

        # current_q = self._current_predict(input_states)
        # next_q = self._target_predict(next_input_states)
        # max_next_q = np.amax(next_q, axis=1)

        # target_q = np.copy(current_q)

        # indexes = np.arange(input_states.shape[0])
        # target_q[indexes, actions[indexes]] = rewards + terminal_inputs * GAMMA * max_next_q

        # result = self._model.fit(x=input_states, y=target_q, verbose=0)
