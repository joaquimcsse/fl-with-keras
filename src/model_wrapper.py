import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, initializers
from tensorflow.keras.optimizers import Adam

import random as python_random
import numpy as np

class Model:
    
    def __init__(self,
            output_units=1,
            output_activation='linear',
            alpha=0.0001,
            random_state=42,
            name : str="some_model",
            autocompile : bool=False,
            manual : bool=False,
            validation: str="",
            verbose: bool=True
    ):

        self.output_units = output_units
        self.output_activation = output_activation
        self.alpha = alpha
        self.random_state = random_state
        self.name = name
        self.validation = validation
        self.verbose = verbose

        if not manual:
            self.model = self._criar_modelo()
        if autocompile:
            self.compilar()
    
    def _criar_modelo(self):

        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        python_random.seed(self.random_state)

        l2_reg = regularizers.l2(self.alpha)
        initializer = initializers.Orthogonal()

        modelo = keras.Sequential([
            keras.Input(shape=(3,), batch_size=1),
            layers.Dense(50, kernel_initializer=initializer, activation='tanh',
                         kernel_regularizer=l2_reg),
            layers.Dense(25, kernel_initializer=initializer, activation='tanh',
                         kernel_regularizer=l2_reg),
            layers.Dense(self.output_units, activation=self.output_activation,
                        kernel_regularizer=l2_reg)
        ])
        
        if self.verbose: print(f"model {self.name} was created")
        return modelo
    
    def compilar(self, optimizer=None, loss='mse', metrics=['mae']):
        if optimizer is None:
          optimizer = Adam(learning_rate=0.001, beta_1=0.9,
                          beta_2=0.999, epsilon=1e-08)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        if self.verbose: print(f"model {self.name} was compiled")
    
    def treinar(self, X_train, y_train):
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        python_random.seed(self.random_state)

        history = self.model.fit(
            X_train, y_train,
            epochs=800,
            batch_size=200,
            validation_split=0.2,
            verbose=self.verbose
        )

        return history
    
    def set_modelo(self, modelo):
        self.model = modelo

    def get_modelo(self):
        return self.model

    def set_pesos(self, weights):
        self.model.set_weights(weights)

    def get_pesos(self):
        return self.model.get_weights()

    def salvar_modelo(self, caminho):
      self.model.save(caminho, include_optimizer=True)

    def prever(self, X):
        return self.model.predict(X, verbose=self.verbose)

    def resumo(self):
        return self.model.summary()

    def config(self):
        return self.model.get_config()
