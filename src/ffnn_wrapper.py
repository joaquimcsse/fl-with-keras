import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, initializers
from tensorflow.keras.optimizers import Adam

import random as python_random
import numpy as np

# ffnn_wrapper.py

class Model:

    def __init__(self,
                    output_units=1,
                    output_activation='linear',
                    alpha=0.0001,
                    random_state=42,
                    name : str="some_model",
                    autocompile : bool=False,
                    manual : bool=False,
                    validation: str=""
                    ):

        self.output_units = output_units
        self.output_activation = output_activation
        self.alpha = alpha
        self.random_state = random_state
        self.name = name
        self.validation = validation

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

        print(f"model {self.name} was created")
        return modelo

    def compilar(self, optimizer=None, loss='mse', metrics=['mae']):
        if optimizer is None:
          optimizer = Adam(learning_rate=0.001, beta_1=0.9,
                          beta_2=0.999, epsilon=1e-08)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print(f"model {self.name} was compiled")

    def treinar(self, X_train, y_train):
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        python_random.seed(self.random_state)

        history = self.model.fit(
            X_train, y_train,
            epochs=800,
            batch_size=200,
            validation_split=0.2,
            verbose=0
        )

        return history

    def neuron_update_interpolated(
        self,
        global_weights,
        percentual: float = 0.1,
        alpha: float = 0.5,
        importance: str = "drift",
        keep_head_local: bool = True,
        verbose: bool = True,
    ):

        """
            Percentage based partial update of local parameters using an interpolated whole-neuron transfer.

            Parameters:
                global_weights  : list[np.ndarray] from the global model's get_pesos().
                percentual      : 0.0 < p <= 1.0 - fraction of neurons touched per layer.
                alpha           : 0.0 < a <= 1.0 - interpolation strength toward the global neuron.
                importance      : "drift"  -> update the neurons that diverged most from global
                                  "random" -> uniform sampling
                keep_head_local : if True the output layer is never aggregated (ideal since the output layer
                            is the most client specift part of the network).
        """

        if not (0.0 < percentual <= 1.0):
            raise ValueError("percentual must satisfy 0.0 < percentual <= 1.0")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must satisfy 0.0 < alpha <= 1.0")
        if importance not in ("drift", "random"):
            raise ValueError("importance must be 'drift' or 'random'")

        local = self.model.get_weights()
        if len(global_weights) != len(local):
            raise ValueError("layer number mismatch between global and local models")
        for lw, gw in zip(local, global_weights):
            if lw.shape != gw.shape:
                raise ValueError(f"shape mismatch: local {lw.shape} vs global {gw.shape}")

        updated = [p.copy() for p in local]
        n_pairs = len(updated) // 2
        # last (kernel, bias) pair is the output layer
        last_pair = n_pairs - 1 if keep_head_local else n_pairs

        rng = np.random.default_rng()

        for i in range(last_pair):
            idx_k, idx_b = 2 * i, 2 * i + 1
            kernel_local, bias_local = updated[idx_k], updated[idx_b]
            kernel_global, bias_global = global_weights[idx_k], global_weights[idx_b]

            num_neurons = bias_local.shape[0]
            n_to_update = max(1, int(np.round(num_neurons * percentual)))

            if importance == "drift":
                # per-neuron L2 distance between local and global parameter vectors
                drift = np.linalg.norm(kernel_global - kernel_local, axis=0)
                drift += np.abs(bias_global - bias_local)
                chosen = np.argsort(drift)[-n_to_update:] # largest divergence first
            else:
                chosen = rng.choice(num_neurons, size=n_to_update, replace=False)

            kernel_local[:, chosen] = (
                (1.0 - alpha) * kernel_local[:, chosen] + alpha * kernel_global[:, chosen]
            )
            bias_local[chosen] = (
                (1.0 - alpha) * bias_local[chosen] + alpha * bias_global[chosen]
            )

            # also updates the selected neurons ('chosen') outgoing weights in the next layer
            # to make the downstream consumer expect the updated feature
            nxt_k = idx_k + 2
            if nxt_k < len(updated):
                row_local, row_global = updated[nxt_k], global_weights[nxt_k]
                if row_local.shape[0] == num_neurons:
                    row_local[chosen, :] = (
                        (1.0 - alpha) * row_local[chosen, :] + alpha * row_global[chosen, :]
                    )

        self.model.set_weights(updated)
        if verbose:
            print(f"--- neuron update done ---")

    def neurons_update(self, global_weights, percentual: float=0.1, verbose: bool=True):

        """
        Parametros:
          global_weights : list[np.ndarray] - returned by calling get_pesos() from a
                                              global model.
          percentual    : float [0.0 < percentual < 1] - neurons fraction to be updated.

        Objective:
        Conceptually it replaces a randomly chosen neuron's incoming weights + bias, yet keeps the next layer's outgoing row for that neuron local.
          therefore, no neuron is ever fully transplanted.
        """

        if not (0.0 < percentual <= 1.0):
            raise ValueError("percentual must be a value in 0.0 < percentual <= 1.0")

        if len(global_weights) != len(self.model.get_weights()):
            raise ValueError("layer number mismatch between global and local models")

        np.random.seed(self.random_state)
        updated = [p.copy() for p in self.model.get_weights()]

        for i in range(len(updated) // 2):

            idx_kernel, idx_bias = i*2, i*2 + 1

            kernel_local, bias_local = updated[idx_kernel], updated[idx_bias]
            kernel_global, bias_global = global_weights[idx_kernel], global_weights[idx_bias]

            if kernel_local.shape != kernel_global.shape:
                raise ValueError("kernel shape mismatch")
            if bias_local.shape != bias_global.shape:
                raise ValueError("bias shape mismatch")

            num_neurons = bias_local.shape[0]
            n_to_update   = max(1, int(np.round(num_neurons * percentual)))

            chosen = np.random.choice(
                num_neurons, size=n_to_update, replace=False
            )

            kernel_local[:, chosen] = kernel_global[:, chosen]
            bias_local[chosen] = bias_global[chosen]

        self.model.set_weights(updated)
        if verbose:
            print("--- updates successfull ---")

    def weights_update(self, global_weights, percentual: float=0.1, verbose: bool=True):

        """
        Parameters:
        global_weights : list[np.ndarray]  → output from get_weights() of a global model.

        percentual : the % in float [0.0 < percentual < 1] you want to update
        """


        if not (0.0 < percentual <= 1.0):
            raise ValueError("percentual must be a value in 0.0 < percentual <= 1.0")

        if len(global_weights) != len(self.model.get_weights()):
            raise ValueError("layer number mismatch between global and local models")

        np.random.seed(self.random_state)
        updated = [p.copy() for p in self.model.get_weights()]

        for i in range(len(updated // 2)):

            idx_kernel, idx_bias = i*2, i*2 + 1

            kernel_local, bias_local = updated[idx_kernel], updated[idx_bias]
            kernel_global, bias_global = global_weights[idx_kernel], global_weights[idx_bias]

            if kernel_local.shape != kernel_global.shape:
                raise ValueError("kernel shape mismatch")
            if bias_local.shape != bias_global.shape:
                raise ValueError("bias shape mismatch")

            target_kernel   = max(1, int(np.round(percentual * kernel_local.shape[0])))
            for column_idx in range(kernel_local.shape[1]):
                chosen = np.random.choice(
                kernel_local.shape[0], size=target_kernel, replace=False
                )

                kernel_local[chosen, column_idx] = kernel_global[chosen, column_idx]

            target_bias = max(1, int(np.round(percentual * bias_local.shape[0])))
            chosen_bias = np.random.choice(
                bias_local.shape[0], size=target_bias, replace=False
                )
            bias_local[chosen_bias] = bias_global[chosen_bias]

        self.model.set_weights(updated)
        if verbose:
            print("--- updates successfull ---")

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
        return self.model.predict(X, verbose=0)

    def resumo(self):
        return self.model.summary()

    def config(self):
        return self.model.get_config()
