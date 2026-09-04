from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import pickle as pkl
import os

def load_df(path : str):
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower().str.strip()
        df.drop(columns='unnamed: 0', inplace=True)
        df.dropna(inplace=True)

        df['capacidade'] = (df['c'].cumsum() / 360)
        capacidade_total = df['capacidade'].iloc[-1]
        df['soc_clean'] = 100 - ((df['capacidade'] / capacidade_total) * 100)
        df.drop(columns=['soc'], inplace=True)

        return df

    except Exception as inst:
        raise inst


def save_scaler(scaler, dir_scaler : str):
    try:
        path = os.path.join(dir_scaler, "validation_round_scalr.pkl")
        with open(path, "wb") as f:
            pkl.dump(scaler, f)
    except Exception as inst:
        raise inst


def make_scaler(dir_dataset, dir_scaler, cell=None, cicle=None):
    if cell is None:
        cell = "c3023"
    if cicle is None:
        cicle = "029"

    path = os.path.join(dir_dataset, f"{cell}-{cicle}.csv")
    df = load_df(path)

    features = df.drop(columns=['soc_clean', 'timestamp', 'capacidade'])
    X_values = features.values.astype('float32')

    scaler = MinMaxScaler()
    scaler.fit(X_values)

    try:
      save_scaler(scaler, dir_scaler)
    except Exception as inst:
        print("exception ocurred in save_scaler():")
        print(type(inst))
        print(inst.args)

    return scaler


def load_scaler(path):
    try:
        with open(path, "rb") as f:
            scaler = pkl.load(f)
        return scaler
    except Exception as e:
        print(f"couldnt load scaler on path {path}")
        raise(e)


def build_data(df, scaler) -> dict:
    features = df.drop(columns=['soc_clean', 'timestamp', 'capacidade'])
    target = df['soc_clean']

    X_values = features.values.astype('float32')
    y_values = target.values.astype('float32')

    X_values = scaler.transform(X_values)

    return {"X": X_values,
            "y": y_values}


def load_all_data(dir_dataset: str, scaler, cells: list = None, cicles: list = None) -> dict:
    if cells is None:
        cells = [f"c30{i}" for i in range(23, 28)]
    if cicles is None:
        cicles = range(29, 44)
        cicles = ["0"+str(cicle) for cicle in cicles]

    data = {}

    for cell in cells:
        data[cell] = {}
        for cicle in cicles:
            path = os.path.join(dir_dataset, f"{cell}-{cicle}.csv")
            try:
                df = load_df(path)
                data[cell][cicle] = build_data(df, scaler)
            except FileNotFoundError as e:
                print(type(e))
                print(f"path [{path}] not found in load_all_data()")
            except Exception as inst:
                print("exception ocurred in load_all_data():")
                print(type(inst))

    return data


def build_packed_data(df, scaler, fit_to_data=False):
    features = df.drop(columns=['soc_clean', 'timestamp', 'capacidade'])
    target = df['soc_clean']

    X_values = features.values.astype('float32')
    y_values = target.values.astype('float32')

    if fit_to_data:
        scaler.fit(X_values)
    X_values = scaler.transform(X_values)

    X_values, y_values = keras.pack_x_y_sample_weight(X_values, y_values, None)

    return (X_values, y_values, scaler)
