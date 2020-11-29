import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import sys


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='sigmoid')
        self.d2 = Dense(10)

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


class Trainer:

    def __init__(self):

        self.model = MyModel()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    def train_step_SA_gradient(self, x_train, y_train, temperature):
        '''
        In each train step, simulated annealing algorithm is used. A new state is proposed based 
        on the gradient of the cost function 
        :param x_train: The independent variables of the training dataset 
        :param y_train: The label of the training dataset 
        :param temperature: The temperature that adjusts how volatile the distribution is
        :return: None
        '''

        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(x_train, training=True)
            loss = self.loss_object(y_train, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        if tf.random.uniform([], 0, 1) < tf.exp((self.losses[-1] - loss) / temperature):
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.losses.append(loss)
        else:
            self.losses.append(self.losses[-1])

        self.train_loss(loss)
        self.train_accuracy(y_train, predictions)

    @tf.function
    def gradient_descent(self, x_train, y_train):
        '''
        The vanilla gradient descent
        :param x_train:
        :param y_train:
        :return: None
        '''
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(x_train, training=True)
            loss = self.loss_object(y_train, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(y_train, predictions)

    def coolingschedule(self, i, n):
        '''
        Return the appropriate temperature in simulated annealing
        :param i: the number that tracks how many iterations of annealing have been done so far
        :param n: the total number of annealing to be done
        :return: the appropriate temperature
        '''

        return 1 - (i / (n + 1))

    def train_step_SA_random(self, x_train, y_train, temperature):
        '''
        In each train step, simulated annealing algorithm is used. A new state is proposed
        based on a uniform distribution on the weights
        :param x_train: The independent variables of the training dataset
        :param y_train: The label of the training dataset
        :param temperature: The temperature that adjusts how volatile the distribution is
        :return: None
        '''

        orig_variable_values = []
        orig_loss = self.losses[-1]

        # ================================ Proposal Function ========================================
        for variable in self.model.trainable_variables:
            orig_variable_value = tf.keras.backend.get_value(variable)
            orig_variable_values.append(orig_variable_value)
            variable_shape = variable.get_shape()
            addition = tf.random.uniform(shape=variable_shape, minval=-5, maxval=5)

            variable.assign(tf.add(variable, addition))

            new_variable_value = tf.keras.backend.get_value(variable)

            # Test if the new proposal is different from the original value
            # if tf.reduce_all(tf.equal(orig_variable_value, new_variable_value)):
            #   tf.print("they are changed")
            # else:
            #   tf.print("they are not change, {}".format(addition))

        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(x_train, training=True)
            loss = self.loss_object(y_train, predictions)

        # ================================ Evaluate the new proposal =================================
        self.train_loss(loss)
        self.train_accuracy(y_train, predictions)

        # If the proposal is bad, assign the original values to the trainable variables
        if tf.random.uniform([], 0, 1) > tf.exp((orig_loss - loss) / temperature):
            for variable_index in range(len(self.model.trainable_variables)):
                variable = self.model.trainable_variables[variable_index]
                orig_value = orig_variable_values[variable_index]

                variable.assign(orig_value)

            self.losses.append(orig_loss)
            self.acceptance.append(0)
        else:
            self.losses.append(loss)
            self.acceptance.append(1)

    @tf.function
    def test_step(self, x_train, y_train):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(x_train, training=False)
        t_loss = self.loss_object(y_train, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(y_train, predictions)

    # Load data
    def load_data(self):
        '''
        Load the MNIST data
        :return: (x_train, y_train), (x_test, y_test)
        '''

        file_name = "./mnist.npz"
        if os.path.exists(file_name):
            f = np.load(file_name)

            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']

            # Standardize the dataset
            x_train, x_test = x_train / 255.0, x_test / 255.0

            # Add a channels dimension
            x_train = x_train[..., tf.newaxis].astype("float32")
            x_test = x_test[..., tf.newaxis].astype("float32")

            return (x_train, y_train), (x_test, y_test)

    def train(self, epochs=10000):
        '''
        :param epochs: The number of times to go over the entire dataset
        :return: None
        '''

        # ============================== Initialize Log ================================================
        time = datetime.datetime.now()
        dir = "./SA_random_step_{}".format(time)
        os.mkdir(dir)

        old_stdout = sys.stdout
        log_file = open("{}/output.log".format(dir), "w")
        sys.stdout = log_file
        sys.stdout = old_stdout

        # =============================== Load Data ====================================================

        (x_train, y_train), (x_test, y_test) = self.load_data()

        train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).shuffle(10000).batch(32)

        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

        # =============================== Start Training ================================================

        self.acceptance = []
        accuracy = []
        self.losses = [1000000]

        ## Initial step

        # Take the first step outside the loop to initialize the variable, otherwise the trainable 
        # variable are not initialized
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(x_train, training=True)
            loss = self.loss_object(y_train, predictions)

        for epoch in range(epochs):
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            temperature = self.coolingschedule(epoch, epochs)
            x_train = x_train
            y_train = y_train
            self.train_step_SA_random(x_train, y_train, temperature)

            # Run the model on the testing dataset to get accuracy
            for test_x_train, test_y_train in test_ds:
                self.test_step(test_x_train, test_y_train)

            print(
                f'Epoch {epoch + 1}, '
                f'Loss: {self.train_loss.result()}, '
                f'Accuracy: {self.train_accuracy.result() * 100}, '
                f'Test Loss: {self.test_loss.result()}, '
                f'Test Accuracy: {self.test_accuracy.result() * 100}'
            )

            accuracy.append((self.train_accuracy.result(), self.test_accuracy.result()))

        # ============================== Save Training Variables ========================================
        np.save("{}/acceptances".format(dir), self.acceptance)
        np.save("{}/losses".format(dir), self.losses)
        np.save("{}/accuracy".format(dir), accuracy)
        self.model.save("{}/SA_random_step".format(dir))

    def plot(self, values_to_be_plot):
        '''
        Plot the values [loss, accuracy, acceptance]
        :param values_to_be_plot: an array
        :return: None
        '''

        plt.plot(values_to_be_plot)
        plt.show()

    def load_train_record(self, record_dir="./SA_random_step"):
        '''
        :param record_dir: the training record directory
        :return:
        '''

        acceptance_series = np.load("{}/acceptances.npy".format(record_dir))
        accuracy_series = np.load("{}/accuracy.npy".format(record_dir))
        loss_series = np.load("{}/losses.npy".format(record_dir))

        return acceptance_series, accuracy_series, loss_series
