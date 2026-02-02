import os
import optuna
from optuna.pruners import MedianPruner
from dataset import Dataset
from mlp_modified import MLP
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, brier_score_loss, f1_score
import numpy as np
from argparse import ArgumentParser
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

def calculate_ece(y_true, y_pred_proba, n_bins=10, threshold=0.5):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin] == (y_pred_proba[in_bin] >= threshold).astype(int))
            avg_confidence_in_bin = np.mean(y_pred_proba[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    
    return ece


def compute_class_weight(y):
    y = np.asarray(y).astype(int)
    counts = np.bincount(y, minlength=2)
    if counts[0] == 0 or counts[1] == 0:
        return None
    total = counts.sum()
    return {0: total / (2.0 * counts[0]), 1: total / (2.0 * counts[1])}


def find_best_threshold(y_true, y_proba, thresholds=None):
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    best_threshold = 0.5
    best_score = -1.0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        score = f1_score(y_true, y_pred, average='macro', zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(t)

    return best_threshold, float(best_score)

def objective(trial, dataset_path, target_column):
    dataset = Dataset(dataset_path, target_column)
    
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    epochs = trial.suggest_int('epochs', 50, 500, step=50)
    
    try:
        mlp = MLP(shape=dataset.get_shape())
        mlp.compile(learning_rate=learning_rate)
        class_weight = compute_class_weight(dataset.target_train.values)
        mlp.train(dataset, epochs=epochs, batch_size=batch_size, verbose=0, plot_path=None, class_weight=class_weight)

        y_val_true = dataset.target_validation.values
        y_val_proba = mlp.model.predict(dataset.features_validation, verbose=0).flatten()
        _, val_f1 = find_best_threshold(y_val_true, y_val_proba)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return val_f1
    except Exception:
        return 0.0


def optimize_hyperparameters(dataset_path, target_column, n_trials, result_dir):
    study = optuna.create_study(
        study_name='baseline_optimization',
        direction='maximize',
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
    
    mlp = MLP(shape=dataset.get_shape())
    mlp.compile(learning_rate=best_params['learning_rate'])
    class_weight = compute_class_weight(dataset.target_train.values)
    mlp.train(
        dataset,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        verbose=1,
        plot_path=os.path.join(result_dir, 'training_curves.png'),
        class_weight=class_weight,
    )
    
    mlp.model.save(os.path.join(result_dir, 'best_model.keras'))
    
    y_val_true = dataset.target_validation.values
    y_val_proba = mlp.model.predict(dataset.features_validation, verbose=0).flatten()
    best_threshold, best_val_f1 = find_best_threshold(y_val_true, y_val_proba)

    y_pred_proba = mlp.model.predict(dataset.features_test, verbose=0).flatten()
    y_pred_class = (y_pred_proba >= best_threshold).astype(int)
    y_true = dataset.target_test.values
    
    class_report = classification_report(y_true, y_pred_class, digits=4)
    conf_matrix = confusion_matrix(y_true, y_pred_class)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = auc(fpr, tpr)
    brier_score = brier_score_loss(y_true, y_pred_proba)
    ece = calculate_ece(y_true, y_pred_proba, n_bins=10, threshold=best_threshold)
    
    with open(os.path.join(result_dir, 'best_hyperparameters.txt'), 'w') as f:
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"class_weight: {class_weight}\n")
        f.write(f"best_threshold: {best_threshold}\n")
        f.write(f"best_val_f1_macro: {best_val_f1}\n")
    
    with open(os.path.join(result_dir, 'classification_report.txt'), 'w') as f:
        f.write(class_report)
        f.write("\nMatriz de Confusão:\n")
        f.write(np.array2string(conf_matrix))
        f.write(f"\nThreshold: {best_threshold:.4f}\n")
        f.write(f"\nAUC-ROC: {auc_score:.4f}\n")
        f.write(f"Brier Score: {brier_score:.4f}\n")
        f.write(f"ECE: {ece:.4f}\n")
    
    return mlp

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('--dataset_name', type=str, default='age_adults.csv', help='File name of the dataset (e.g., age_adults.csv)')
    args.add_argument('--target_column', type=str, default='CKD progression', help='Name of the target column in the dataset')
    args.add_argument('--n_trials', type=int, default=20, help='Number of trials for optimization')
    args.add_argument('--load_study', action='store_true', help='Load existing study')
    args.add_argument('--study_path', type=str, default='', help='Path to the saved study')
    args = args.parse_args()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dataset_base = args.dataset_name.replace('.csv', '')
    result_dir = os.path.join('results', 'baseline', f'{dataset_base}_{timestamp}')
    os.makedirs(result_dir, exist_ok=True)
    
    dataset_path = f'datasets/dataset_filled_boruta_{args.dataset_name}'
    dataset = Dataset(dataset_path, args.target_column)

    if args.load_study:
        study = joblib.load(args.study_path)
    else:
        study = optimize_hyperparameters(dataset_path, args.target_column, args.n_trials, result_dir)

    train_and_evaluate(study, dataset, result_dir)
