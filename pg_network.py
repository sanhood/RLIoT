

from collections import OrderedDict
import numpy as np

from keras import layers
from keras.models import Model
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D






class PGLearner:
    def __init__(self, pa):

        self.input_height = pa.network_input_height
        self.input_width = pa.network_input_width
        self.output_height = pa.network_output_dim

        self.num_frames = pa.num_frames

        self.update_counter = 0





        # image representation
        self.model = \
            build_pg_network(pa.network_input_height, pa.network_input_width, pa.network_output_dim)



        self.lr_rate = pa.lr_rate
        self.rms_rho = pa.rms_rho
        self.rms_eps = pa.rms_eps
        self.__build_train_fn()



        # print ' params=', params, ' count=', lasagne.layers.count_params(self.l_out)

        # self._get_param = theano.function([], params)

        # ===================================
        # training function part
        # ===================================

        # prob_act = lasagne.layers.get_output(self.l_out, states)
        #
        # self._get_act_prob = theano.function([states], prob_act, allow_input_downcast=True)
        #
        # # --------  Policy Gradient  --------
        #
        # N = states.shape[0]
        #
        # loss = T.log(prob_act[T.arange(N), actions]).dot(values) / N  # call it "loss"
        #
        # grads = T.grad(loss, params)
        #
        # updates = rmsprop_updates(
        #     grads, params, self.lr_rate, self.rms_rho, self.rms_eps)
        #
        # # updates = adam_update(
        # #     grads, params, self.lr_rate)
        #
        # self._train_fn = theano.function([states, actions, values], loss,
        #                                  updates=updates, allow_input_downcast=True)
        #
        # self._get_loss = theano.function([states, actions, values], loss, allow_input_downcast=True)
        #
        # self._get_grad = theano.function([states, actions, values], grads, allow_input_downcast=True)

    def __build_train_fn(self):
        """Create a train function
        It replaces `model.fit(X, y)` because we use the output of model and use it for training.
        For example, we need action placeholder
        called `action_one_hot` that stores, which action we took at state `s`.
        Hence, we can update the same action.
        This function will create
        `self.train_fn([state, action_one_hot, discount_reward])`
        which would train the model.
        """
        action_prob_placeholder = self.model.output
        action_onehot_placeholder = K.placeholder(shape=(None, self.output_height),
                                                  name="action_onehot")
        discount_reward_placeholder = K.placeholder(shape=(None,),
                                                    name="discount_reward")

        action_prob = K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1)
        log_action_prob = K.log(action_prob)

        loss = - log_action_prob * discount_reward_placeholder
        loss = K.mean(loss)

        adam = optimizers.Adam()

        updates = adam.get_updates(params=self.model.trainable_weights,

                                   loss=loss)

        self.train_fn = K.function(inputs=[self.model.input,
                                           action_onehot_placeholder,
                                           discount_reward_placeholder],
                                   outputs=[],
                                   updates=updates)

    def get_action(self, state):
        """Returns an action at given `state`
        Args:
            state (1-D or 2-D Array): It can be either 1-D array of shape (state_dimension, )
                or 2-D array shape of (n_samples, state_dimension)
        Returns:
            action: an integer action value ranging from 0 to (n_actions - 1)
        """
        state = state.reshape(1,self.input_width*self.input_height)
        action_prob = np.squeeze(self.model.predict(state))
        return np.random.choice(np.arange(self.output_height), p=action_prob)

    def fit(self, S, A, R):
        """Train a network
        Args:
            S (2-D Array): `state` array of shape (n_samples, state_dimension)
            A (1-D Array): `action` array of shape (n_samples,)
                It's simply a list of int that stores which actions the agent chose
            R (1-D Array): `reward` array of shape (n_samples,)
                A reward is given after each action.
        """
        action_onehot = np_utils.to_categorical(A, num_classes=self.output_height)

        self.train_fn([S, action_onehot, R])

    # get the action based on the estimated value
    def choose_action(self, state):

        act_prob = self.get_one_act_prob(state)

        csprob_n = np.cumsum(act_prob)
        act = (csprob_n > np.random.rand()).argmax()

        # print(act_prob, act)

        return act

    def train(self, states, actions, values):

        loss = self._train_fn(states, actions, values)
        return loss

    def get_params(self):

        return self._get_param()

    def get_grad(self, states, actions, values):

        return self._get_grad(states, actions, values)

    def get_one_act_prob(self, state):

        states = np.zeros((1, 1, self.input_height, self.input_width), dtype=theano.config.floatX)
        states[0, :, :] = state
        act_prob = self._get_act_prob(states)[0]

        return act_prob

    def get_act_probs(self, states):  # multiple states, assuming in floatX format
        act_probs = self._get_act_prob(states)
        return act_probs

    #  -------- Supervised Learning --------
    def su_train(self, states, target):
        loss, prob_act = self._su_train_fn(states, target)
        return np.sqrt(loss), prob_act

    def su_test(self, states, target):
        loss, prob_act = self._su_loss(states, target)
        return np.sqrt(loss), prob_act

    #  -------- Save/Load network parameters --------
    def return_net_params(self):
        return lasagne.layers.helper.get_all_param_values(self.l_out)

    def set_net_params(self, net_params):
        lasagne.layers.helper.set_all_param_values(self.l_out, net_params)


# ===================================
# build neural network
# ===================================


def build_pg_network(input_height, input_width, output_length):
    model = Sequential()
    model.add(Dense(20,  input_shape= (input_height*input_width,)))
    print(model.input_shape)
    model.add(Activation('relu'))
    model.add(Dense(output_length))
    model.add(Activation('softmax'))
    return model










