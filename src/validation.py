from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import os

from .ffnn_wrapper import Model

def train_models(inputs, dir_save, validation="c3026"):
    cells = ["c3023", "c3024", "c3025", "c3026", "c3027"]
    cells = [cell for cell in cells if cell != validation]
    train_cicles = ['0'+str(i) for i in range(29, 39)]

    models = {cell: Model(name=f"cell-{cell}", autocompile=True, manual=False) for cell in cells}

    for cell in cells:
        for cicle in train_cicles:
            model = models[cell]
            features = inputs[cell][cicle]['X']
            target = inputs[cell][cicle]['y']
            model.treinar(features, target)
            try:
              path = os.path.join(dir_save, f"model_cell_{cell}.keras")
              model.salvar_modelo(path)
            except Exception as e:
                print(f"couldn't save {cell} model to {path}")

    return models


def load_keras_model(
    path: str,
    compile: bool=True, 
    name: str="model-name"
    ):
    """
    Load a Keras model from a .keras file and attach to ffnn_wrapper.Model() 
    instance.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    model = Model(name=name, manual=True)
    model.set_modelo(load_model(compile=compile))
    return model


def train_over_new_cicle(models : dict, inputs : dict, cicle : str) -> dict:
    new = {}
    for cell, model in models.items():
        X_train = inputs[cell][cicle]['X']
        y_train = inputs[cell][cicle]['y']

        model.treinar(X_train, y_train)
        new[cell] = model
    return new


def test_on_validation_cell(models, inputs, validation: str="c3026", cicles:list=None):
    if cicles is None:
        cicles = ["029"]
    cells = ["c3023", "c3024", "c3025", "c3026", "c3027"]
    cells = [cell for cell in cells if cell != validation]

    results = {cell: {"r2":[],
                      "rmse": []} for cell in cells}

    for cell in cells:
        model = models[cell]
        for cicle in cicles:
            X_test = inputs[validation][cicle]['X']
            y_test = inputs[validation][cicle]['y']

            y_pred = model.prever(X_test)

            results[cell]["r2"].append(r2_score(y_test, y_pred))
            results[cell]["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred)))

    return results


def dump_to_excel(results: dict, path, validation="c3026"):
    cells = ["c3023", "c3024", "c3025", "c3026", "c3027"]
    cells = [cell for cell in cells if cell != validation]

    df = pd.DataFrame()
    for cell, d in results.items():
        for metric, values in d.items():
            df[f"cell_{cell}_{metric}"] = pd.Series(values)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel file not found: {path}")
    df.to_excel(path)
    print(f"values written on {path}")
    

def basic_test(modelos, inputs_teste, ciclo: str=None):
    if ciclo is None:
        ciclo = "039"

    r2_values = {}
    rmse_values = {}

    for celula, modelo in modelos.items():

        X_test = inputs_teste[celula][ciclo]['X']
        y_test = inputs_teste[celula][ciclo]['y']

        y_pred = modelo.prever(X_test)

        r2_values[celula] = r2_score(y_test, y_pred)
        rmse_values[celula] = np.sqrt(mean_squared_error(y_test, y_pred))

    return (r2_values, rmse_values)
