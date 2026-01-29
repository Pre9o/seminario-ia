import os
import optuna
from optuna.pruners import MedianPruner
from dataset import Dataset
from autoencoder import Autoencoder
from argparse import ArgumentParser
import numpy as np
from datetime import datetime
import joblib
import matplotlib.pyplot as plt


def objective(trial, dataset_path, target_column):
    dataset = Dataset(dataset_path, target_column)
    
    n_layers = trial.suggest_int('n_layers', 2, 4)
    
    hidden_units = []
    input_size = dataset.get_shape()
    
    for i in range(n_layers):
        if i == 0:
            min_units = max(16, input_size // 2)
            max_units = min(256, input_size * 2)
        else:
            min_units = 8
            max_units = max(16, hidden_units[-1] // 2)
            
            if max_units >= hidden_units[-1]:
                max_units = hidden_units[-1] - 8
        
        if max_units < min_units:
            max_units = min_units
        
        step = 8 if max_units < 64 else 16
        
        units = trial.suggest_int(f'units_layer_{i}', min_units, max_units, step=step)
        hidden_units.append(units)
    
    mask_ratio = trial.suggest_float('mask_ratio', 0.1, 0.5)
    pretrain_lr = trial.suggest_float('pretrain_lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    pretrain_epochs = trial.suggest_int('pretrain_epochs', 50, 300, step=50)
    
    categorical_indices = [4, 5]
    categorical_cardinalities = {4: 4, 5: 2}
    
    try:
        autoencoder = Autoencoder(
            shape=dataset.get_shape(),
            categorical_indices=categorical_indices,
            categorical_cardinalities=categorical_cardinalities,
            mask_ratio=mask_ratio,
            hidden_units=hidden_units,
            learning_rate=pretrain_lr,
            mask_value=-999.0
        )
        
        history = autoencoder.train(dataset, epochs=pretrain_epochs, batch_size=batch_size, verbose=0)
        
        val_loss = min(history['val_loss'])
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return val_loss
    except Exception:
        return float('inf')  


def optimize_hyperparameters(dataset_path, target_column, n_trials, result_dir):
    study = optuna.create_study(
        study_name='autoencoder_optimization',
        direction='minimize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    study.optimize(
        lambda trial: objective(trial, dataset_path, target_column),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    joblib.dump(study, os.path.join(result_dir, 'study.pkl'))
    
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig(os.path.join(result_dir, 'optimization_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig(os.path.join(result_dir, 'param_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return study


def train_and_evaluate(study, dataset, result_dir):
    best_params = study.best_params
    
    n_layers = best_params['n_layers']
    hidden_units = [best_params[f'units_layer_{i}'] for i in range(n_layers)]
    
    categorical_indices = [4, 5]
    categorical_cardinalities = {4: 4, 5: 2}
    
    autoencoder = Autoencoder(
        shape=dataset.get_shape(),
        categorical_indices=categorical_indices,
        categorical_cardinalities=categorical_cardinalities,
        mask_ratio=best_params['mask_ratio'],
        hidden_units=hidden_units,
        learning_rate=best_params['pretrain_lr'],
        mask_value=-999.0
    )
    
    autoencoder.train(
        dataset,
        epochs=best_params['pretrain_epochs'],
        batch_size=best_params['batch_size'],
        verbose=1,
        plot_path=os.path.join(result_dir, 'training_curves.png')
    )
    
    autoencoder.save_encoder(os.path.join(result_dir, 'best_encoder.keras'))
    
    return autoencoder

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('--dataset_name', type=str, default='age_elderly.csv', help='File name of the dataset (e.g., age_adults.csv)')
    args.add_argument('--target_column', type=str, default='CKD progression', help='Name of the target column in the dataset')
    args.add_argument('--n_trials', type=int, default=20, help='Number of trials for optimization')
    args.add_argument('--load_study', action='store_true', help='Load existing study')
    args.add_argument('--study_path', type=str, default='', help='Path to the saved study')
    args = args.parse_args()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dataset_base = args.dataset_name.replace('.csv', '')
    result_dir = os.path.join('results', 'autoencoder', f'{dataset_base}_{timestamp}')
    os.makedirs(result_dir, exist_ok=True)
    
    dataset_path = f'datasets/dataset_filled_boruta_{args.dataset_name}'
    dataset = Dataset(dataset_path, args.target_column)

    print(f"DATASET SHAPE: {dataset.get_shape()}")

    if args.load_study:
        study = joblib.load(args.study_path)
    else:
        study = optimize_hyperparameters(dataset_path, args.target_column, args.n_trials, result_dir)

    train_and_evaluate(study, dataset, result_dir)
