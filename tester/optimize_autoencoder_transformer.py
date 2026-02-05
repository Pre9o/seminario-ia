import os
import optuna
from optuna.pruners import MedianPruner
from dataset import Dataset
from autoencoder_transformer import AutoencoderTransformer
from argparse import ArgumentParser
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import numpy as np


def objective(trial, dataset_path, target_column, categorical_indices, categorical_cardinalities):
    dataset = Dataset(dataset_path, target_column)

    mask_ratio = trial.suggest_float('mask_ratio', 0.3, 0.7)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64])
    epochs = trial.suggest_int('epochs', 50, 300, step=50)

    embed_dim = trial.suggest_categorical('embed_dim', [16, 32, 48, 64, 96, 128])
    num_layers = trial.suggest_int('num_layers', 1, 4)

    num_heads = trial.suggest_int('num_heads', 2, 8)

    if num_heads > embed_dim:
        return float('inf')
    if embed_dim % num_heads != 0:
        return float('inf')

    ff_dim = trial.suggest_categorical('ff_dim', [16, 32, 48, 64, 96, 128, 192, 256])
    dropout = trial.suggest_float('dropout', 0.0, 0.3)

    try:
        autoencoder = AutoencoderTransformer(
            shape=dataset.get_shape(),
            categorical_indices=categorical_indices,
            categorical_cardinalities=categorical_cardinalities,
            mask_ratio=mask_ratio,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            mask_value=-999.0,
        )

        history = autoencoder.train(dataset, epochs=epochs, batch_size=batch_size, verbose=0, plot_path=None)
        val_loss = min(history['val_loss']) if 'val_loss' in history else min(history['loss'])

        if trial.should_prune():
            raise optuna.TrialPruned()

        return float(val_loss)
    except Exception:
        return float('inf')


def optimize_hyperparameters(dataset_path, target_column, categorical_indices, categorical_cardinalities, n_trials, result_dir):
    study = optuna.create_study(
        study_name='autoencoder_transformer_optimization',
        direction='minimize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    study.optimize(
        lambda trial: objective(trial, dataset_path, target_column, categorical_indices, categorical_cardinalities),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    joblib.dump(study, os.path.join(result_dir, 'study.pkl'))

    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig(os.path.join(result_dir, 'optimization_history.svg'), dpi=300, bbox_inches='tight')
    plt.close()

    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig(os.path.join(result_dir, 'param_importance.svg'), dpi=300, bbox_inches='tight')
    plt.close()

    return study


def train_and_evaluate(study, dataset, categorical_indices, categorical_cardinalities, result_dir, n_runs):
    best_params = study.best_params

    n_runs = 20
    all_results = []
    runs_path = os.path.join(result_dir, 'reconstruction_metrics_runs.txt')
    with open(runs_path, 'w'):
        pass

    test_array = dataset.features_test.values if hasattr(dataset.features_test, 'values') else dataset.features_test

    for run_idx in range(n_runs):
        autoencoder = AutoencoderTransformer(
            shape=dataset.get_shape(),
            categorical_indices=categorical_indices,
            categorical_cardinalities=categorical_cardinalities,
            mask_ratio=best_params['mask_ratio'],
            embed_dim=best_params['embed_dim'],
            num_layers=best_params['num_layers'],
            num_heads=best_params['num_heads'],
            ff_dim=best_params['ff_dim'],
            dropout=best_params['dropout'],
            learning_rate=best_params['learning_rate'],
            mask_value=-999.0,
        )

        autoencoder.train(
            dataset,
            epochs=best_params['epochs'],
            batch_size=best_params['batch_size'],
            verbose=1,
            plot_path=os.path.join(result_dir, f'training_curves_run_{run_idx + 1}.svg'),
        )

        autoencoder.save_encoder(os.path.join(result_dir, 'best_encoder.keras'))

        X_test_masked, _ = autoencoder.create_masked_data(test_array)
        y_test_targets = autoencoder.prepare_targets(test_array)

        predictions = autoencoder.model.predict(X_test_masked, verbose=0)

        results = {}

        if autoencoder.continuous_indices:
            continuous_pred = predictions[0]
            continuous_true = y_test_targets[0]
            mse = np.mean((continuous_pred - continuous_true) ** 2)
            mae = np.mean(np.abs(continuous_pred - continuous_true))
            results['continuous_mse'] = float(mse)
            results['continuous_mae'] = float(mae)

        if autoencoder.categorical_indices:
            offset = 1 if autoencoder.continuous_indices else 0
            for i, cat_idx in enumerate(autoencoder.categorical_indices):
                cat_pred = predictions[offset + i]
                cat_true = y_test_targets[offset + i]

                cat_pred_class = np.argmax(cat_pred, axis=1)
                accuracy = np.mean(cat_pred_class == cat_true)

                cat_true_one_hot = np.zeros_like(cat_pred)
                cat_true_one_hot[np.arange(len(cat_true)), cat_true.astype(int)] = 1
                ce_loss = -np.mean(np.sum(cat_true_one_hot * np.log(cat_pred + 1e-10), axis=1))

                results[f'categorical_{cat_idx}_accuracy'] = float(accuracy)
                results[f'categorical_{cat_idx}_ce_loss'] = float(ce_loss)

        all_results.append(results)

        with open(runs_path, 'a') as f:
            f.write(f"run: {run_idx + 1}\n")
            for key in sorted(results.keys()):
                f.write(f"{key}: {results[key]:.6f}\n")
            f.write("\n")

    metric_keys = sorted({k for r in all_results for k in r.keys()})
    mean_results = {}
    std_results = {}
    for k in metric_keys:
        vals = [r[k] for r in all_results if k in r]
        mean_results[k] = float(np.mean(vals)) if vals else float('nan')
        if not vals:
            std_results[k] = float('nan')
        elif len(vals) == 1:
            std_results[k] = 0.0
        else:
            std_results[k] = float(np.std(vals, ddof=1))

    with open(os.path.join(result_dir, 'best_hyperparameters.txt'), 'w') as f:
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")

    with open(os.path.join(result_dir, 'reconstruction_metrics.txt'), 'w') as f:
        f.write("Métricas de Reconstrução no Conjunto de Teste:\n\n")
        f.write(f"n_runs: {n_runs}\n\n")
        for key in metric_keys:
            f.write(f"{key}_mean: {mean_results[key]:.4f}\n")
            f.write(f"{key}_std: {std_results[key]:.4f}\n")


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--dataset_name', type=str, default='age_elderly.csv')
    args.add_argument('--target_column', type=str, default='CKD progression')
    args.add_argument('--n_trials', type=int, default=20)
    args.add_argument('--load_study', action='store_true')
    args.add_argument('--study_path', type=str, default='')
    args.add_argument('--n_runs', type=int, default=20)
    args = args.parse_args()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dataset_base = args.dataset_name.replace('.csv', '')
    result_dir = os.path.join('results', 'autoencoder_transformer', f'{dataset_base}_{timestamp}')
    os.makedirs(result_dir, exist_ok=True)

    dataset_path = f'datasets/dataset_filled_boruta_{args.dataset_name}'
    dataset = Dataset(dataset_path, args.target_column)

    if args.dataset_name == 'age_elderly.csv':
        categorical_indices = [1, 5, 6]
        categorical_cardinalities = {1: 4, 5: 4, 6: 2}
    elif args.dataset_name == 'etiology12.csv':
        categorical_indices = [5, 6]
        categorical_cardinalities = {5: 4, 6: 2}
    else:
        categorical_indices = [2, 6]
        categorical_cardinalities = {2: 4, 6: 2}

    if args.load_study:
        study = joblib.load(args.study_path)
    else:
        study = optimize_hyperparameters(
            dataset_path,
            args.target_column,
            categorical_indices,
            categorical_cardinalities,
            args.n_trials,
            result_dir,
        )

    train_and_evaluate(study, dataset, categorical_indices, categorical_cardinalities, result_dir, args.n_runs)

