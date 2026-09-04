from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
import numpy as np
from .ffnn_wrapper import Model

def find_sample_num(inputs, cells: list=None, cicles: list=None):
    if cells is None:
        cells = ["c3023", "c3024", "c3025", "c3027"]
    if cicles is None:
        cicles = range(29, 44)
        cicles = ["0"+str(cicle) for cicle in cicles]

    samples = {cell: 0 for cell in cells}
    for cell in cells:
        for cicle in cicles:
                samples[cell] += len(inputs[cell][cicle]['X'])
    return samples


def find_rmse_models(models, inputs, validation="c3026", test_cicle : str=None):
    if validation=="c3026":
        cells = ["c3023", "c3024", "c3025", "c3027"]
    if test_cicle is None:
        test_cicle = "029"

    rmse_models = {}
    for cell in cells:
        modelo = models[cell]

        y_test = inputs[validation][test_cicle]['y']
        X_test = inputs[validation][test_cicle]['X']

        y_pred = modelo.prever(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_models[cell] = rmse
    return rmse_models


def aggregate_model_weights(weights_by_cell, cell_proportions):
    """
    Combine model weights using a proportion assigned to each cell.
    """
    reference_weights = next(iter(weights_by_cell.values()))
    aggregated_weights = [
        np.zeros_like(weight_array, dtype=float)
        for weight_array in reference_weights
    ]
    for cell, cell_weights in weights_by_cell.items():
        proportion = cell_proportions[cell]
        for weight_index, weight_array in enumerate(cell_weights):
            aggregated_weights[weight_index] += proportion * weight_array
    return aggregated_weights


def build_global_models(
    local_models,
    inputs,
    validation_cell="c3026",
):
    """
    Build global models using three aggregation strategies:
    1. Simple average of local model weights.
    2. Sample-weighted average based on the number of training samples.
    3. RMSE-weighted average based on local model performance.
    Args:
        local_models: Dictionary containing one trained model per cell.
        inputs: inputs built by .data_processing.build_all_data()
        validation_cell: Cell being used for validation.
    Returns:
        A dictionary containing the three aggregated global models.
    """
    validation_sets = {
        "c3026": ["c3023", "c3024", "c3025", "c3027"],
    }
    if validation_cell not in validation_sets:
        raise ValueError(
            f"Unsupported validation cell: {validation_cell}"
        )
    test_cells = validation_sets[validation_cell]
    global_models = {
        "simple_average": Model(name="simple_average", autocompile=True),
        "sample_weighted": Model(name="sample_weighted", autocompile=True),
        "rmse_weighted": Model(name="rmse_weighted", autocompile=True),
    }
    local_weights_by_cell = {
        cell: local_models[cell].get_pesos()
        for cell in test_cells
    }
    # ------------------------------------------------------------------
    # 1. Simple average of local model weights
    # ------------------------------------------------------------------
    equal_proportion = 1 / len(test_cells)
    simple_proportions = {
        cell: equal_proportion
        for cell in test_cells
    }
    simple_average_weights = aggregate_model_weights(
        local_weights_by_cell,
        simple_proportions,
    )
    global_models["simple_average"].set_pesos(simple_average_weights)
    # ------------------------------------------------------------------
    # 2. Sample-weighted average of local model weights
    # ------------------------------------------------------------------
    samples_by_cell = find_sample_num(inputs) 
    total_samples = sum(samples_by_cell.values())
    if total_samples <= 0:
        raise ValueError("The total number of training samples must be greater than zero.")
    sample_proportions = {
        cell: samples_by_cell[cell] / total_samples
        for cell in test_cells
    }
    sample_weighted_weights = aggregate_model_weights(
        local_weights_by_cell,
        sample_proportions,
    )
    global_models["sample_weighted"].set_pesos(sample_weighted_weights)
    # ------------------------------------------------------------------
    # 3. RMSE-weighted average of local model weights
    # ------------------------------------------------------------------
    rmse_by_cell = find_rmse_models(local_models, inputs)
    if any(rmse <= 0 for rmse in rmse_by_cell.values()):
        raise ValueError("RMSE values must be greater than zero.")
    inverse_rmse_by_cell = {
        cell: 1 / rmse
        for cell, rmse in rmse_by_cell.items()
    }
    total_inverse_rmse = sum(inverse_rmse_by_cell.values())
    rmse_proportions = {
        cell: inverse_rmse_by_cell[cell] / total_inverse_rmse 
        for cell in test_cells
    }
    rmse_weighted_weights = aggregate_model_weights(
        local_weights_by_cell,
        rmse_proportions,
    )
    global_models["rmse_weighted"].set_pesos(rmse_weighted_weights)
    return global_models
