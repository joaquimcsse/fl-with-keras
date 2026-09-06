import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, initializers
from tensorflow.keras.optimizers import Adam

import random as python_random
import numpy as np
from .model_wrapper import Model

class NeuronUpdater:

    def __init__(self, model): 
        if isinstance(model, keras.Sequential):
            self.model = model
        elif isinstance(model, Model):
            self.model = model.get_modelo()
        else:
            raise ValueError("'model' must be either keras.Sequential or model_wrapper.Model")


    def validated_neuron_update(
        self,
        global_weights,
        percentual: float = 0.2,
        alpha: float=None,
        X_val: list=None,
        y_val: list=None,
        keep_head_local: bool = True,
        layer_decay: float = 0.5,
        rng=None,
        verbose: bool = True
    ):

        """
            Update a fraction of the current network's neurons with a given set of global neurons;
            Both sets must have identical shapes and numbers of layers.
            Goal: to provide a robust percentage based neuron update in a highly heterogenous environment.

            Parameters:

            global_weights : list[np.ndarray]
                Weights returned by get_pesos() of the global model.
            percentual : float, 0.0 < percentual <= 1.0
                Fraction of neurons updated per layer (unchanged contract).
            alpha : float, 0.0 < alpha <= 1.0
                Base blending strength. alpha = 1.0 reproduces hard replacement.
            X_val, y_val : optional
                Small local validation split (e.g. the client's own last cycles).
                If given, the update is only kept when local error does not get worse,
                and alpha is chosen from a tiny candidate list by local validation.
            keep_head_local : bool
                Keep the output layer fully local (personalization head).
            layer_decay : float
                Per-layer damping: effective alpha of layer i is alpha * layer_decay**i,
                so early (feature) layers absorb more global knowledge than late ones.
            rng : np.random.Generator, optional
                Pass a persistent generator so different neurons are drawn each round.
        """

        if not (0.0 < percentual <= 1.0):
            raise ValueError("percentual must satisfy 0.0 < percentual <= 1.0")
        if alpha is None:
            alpha = self.alpha
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must satisfy 0.0 < alpha <= 1.0")

        local_weights = self.model.get_weights()
        if len(global_weights) != len(local_weights):
            raise ValueError("layer number mismatch between global and local models")

        if rng is None:
            rng = np.random.default_rng(self.random_state)

        n_layers = len(local_weights) // 2
        last_layer = n_layers - 1

        def _blend(a_base):
            """Build one candidate weight list for a given base alpha."""
            updated = [w.copy() for w in local_weights]

            for i in range(n_layers):
                if keep_head_local and i == last_layer: # dont update output layer
                    continue

                idx_k, idx_b = 2 * i, 2 * i + 1
                k_loc, b_loc = updated[idx_k], updated[idx_b]
                k_glb, b_glb = global_weights[idx_k], global_weights[idx_b]

                if k_loc.shape != k_glb.shape or b_loc.shape != b_glb.shape:
                    raise ValueError(f"shape mismatch on layer {i}")

                num_neurons = b_loc.shape[0]
                n_to_update = max(1, min(num_neurons,
                                        int(np.round(num_neurons * percentual))))

                # --- heterogeneity-aware selection -------------------------------
                # Neurons that drifted most from the global model are the ones this
                # client over-specialized; they are the useful ones to pull back.
                drift = np.linalg.norm(k_glb - k_loc, axis=0) + np.abs(b_glb - b_loc)
                total = drift.sum()
                if total <= 0 or not np.isfinite(total):
                    chosen = rng.choice(num_neurons, size=n_to_update, replace=False)
                else:
                    # probabilistic, drift-weighted: keeps randomness (no layer gets
                    # frozen into the same neurons every round) but favours drift.
                    p = drift / total
                    chosen = rng.choice(num_neurons, size=n_to_update,
                                        replace=False, p=p)

                # --- damped, per-layer blending ----------------------------------
                a_eff = float(np.clip(a_base * (layer_decay ** i), 0.0, 1.0))
                if a_eff <= 0.0:
                    continue

                k_loc[:, chosen] = (1.0 - a_eff) * k_loc[:, chosen] + a_eff * k_glb[:, chosen]
                b_loc[chosen] = (1.0 - a_eff) * b_loc[chosen] + a_eff * b_glb[chosen]

            return updated

        # update without validation if no validation is available
        if X_val is None or y_val is None:
            self.model.set_weights(_blend(alpha))
            if verbose:
                print(f"--- update applied (alpha={alpha}, percentual={percentual}) ---")
            return {"alpha": alpha, "accepted": True}

        # try to validate before updating
        def _rmse(weights):
            self.model.set_weights(weights)
            avg_rmse = n = 0
            for X, y in zip(X_val, y_val):
                pred = np.asarray(self.model.predict(X, verbose=0)).ravel()
                avg_rmse += float(np.sqrt(np.mean((np.asarray(y).ravel() - pred) ** 2)))
                n += 1
            return avg_rmse / n

        base_rmse = _rmse(local_weights)

        best_w, best_rmse, best_a = local_weights, base_rmse, 0.0
        for a in (alpha * 0.25, alpha * 0.5, alpha):
            cand = _blend(a)
            r = _rmse(cand)
            if np.isfinite(r) and r < best_rmse:
                best_w, best_rmse, best_a = cand, r, a

        self.model.set_weights(best_w)

        accepted = best_a > 0.0
        if verbose:
            if accepted:
                print(f"--- weights updated ---")
            else:
                print(f"--- LOCAL WEIGHTS WERE KEPT SINCE NO ALPHA IMPROVED THE RMSE ---")


    def neuron_update_interpolated(
        self,
        global_weights,
        percentual: float = 0.1,
        alpha: float = 0.5,
        importance: str = "drift",
        keep_head_local: bool = True,
        verbose: bool = True
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


    def neurons_update(self,
        global_weights,
        percentual: float=0.1,
        verbose: bool=True
    ):

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


    def weights_update(self,
        global_weights,
        percentual: float=0.1,
        verbose: bool=True
    ):

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
