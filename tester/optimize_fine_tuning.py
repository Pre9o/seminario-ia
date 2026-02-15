import os
import optuna
from optuna.pruners import MedianPruner
from dataset import Dataset
from mlp_modified import MLP
from sklearn.metrics import roc_curve, auc, confusion_matrix, brier_score_loss, f1_score, precision_score, recall_score, accuracy_score
import numpy as np
from argparse import ArgumentParser
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow import keras


def sample_hidden_layers(trial, *, max_layers=5, first_layer_choices=None, min_neurons=4):
    if first_layer_choices is None:
        first_layer_choices = [8, 16, 32, 64, 128]

    n_layers = trial.suggest_int('n_layers', 1, max_layers)
    layer_1 = trial.suggest_categorical('layer_1', first_layer_choices)
    layers = [int(layer_1)]

    prev = layers[0]
    for i in range(2, n_layers + 1):
        ratio = trial.suggest_categorical(f'layer_{i}_ratio', ['same', 'half'])
        if prev <= min_neurons or (prev // 2) < min_neurons:
            ratio = 'same'
        nxt = prev if ratio == 'same' else prev // 2
        layers.append(int(nxt))
        prev = nxt

    layers = [n for n in layers if n >= min_neurons]
    return layers

def build_hidden_layers_from_params(best_params, *, min_neurons=4):
    n_layers = best_params.get('n_layers', 1)
    layer_1 = best_params.get('layer_1', 128)

    layers = [layer_1]
    prev = layer_1
    for i in range(2, n_layers + 1):
        ratio = best_params.get(f'layer_{i}_ratio', 'same')
        if prev <= min_neurons:
            ratio = 'same'
        nxt = prev if ratio == 'same' else prev // 2
        if nxt < min_neurons:
            break
        layers.append(int(nxt))
        prev = nxt

    return layers

def calculate_ece(y_true, y_pred_proba, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
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
    # y_true = np.asarray(y_true).astype(int)
    # y_proba = np.asarray(y_proba).astype(float)
    thresholds = np.linspace(0.0, 1.0, 101)

    best_threshold = 0.5
    best_score = -1.0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        score = f1_score(y_true, y_pred, average='macro', zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold, best_score

def objective(trial, source_model_path, dataset_path, target_column, freeze_layers):
    dataset = Dataset(dataset_path, target_column)
    pre_trained_model = keras.models.load_model(source_model_path)

    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True)
    if freeze_layers:
        learning_rate = trial.suggest_float('learning_rate', 1e-9, 1e-2, log=True)
    else:
        learning_rate = trial.suggest_float('learning_rate', 1e-9, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    epochs = trial.suggest_int('epochs', 50, 1000, step=50)
    
    try:
        mlp = MLP(shape=dataset.get_shape(), pre_trained_mlp=pre_trained_model, freeze_layers=freeze_layers)
        mlp.reset_output_layer()
        mlp.compile(learning_rate=learning_rate, weight_decay=weight_decay)
        class_weight = compute_class_weight(dataset.target_train.values)
        mlp.train(dataset, epochs=epochs, batch_size=batch_size, verbose=0, plot_path=None, class_weight=class_weight)

        y_val_true = dataset.target_validation.values
        y_val_proba = mlp.model.predict(dataset.features_validation, verbose=0).flatten()
        _, val_f1 = find_best_threshold(y_val_true, y_val_proba)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return val_f1
    except Exception as e:
        print(f"Trial failed with exception: {e}")
        return 0.0

def optimize_hyperparameters(dataset_path, source_model_path, freeze_layers, target_column, n_trials, result_dir):
    study = optuna.create_study(
        study_name='baseline_optimization',
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    study.optimize(
        lambda trial: objective(trial, source_model_path, dataset_path, target_column, freeze_layers),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    joblib.dump(study, os.path.join(result_dir, 'study.pkl'))
    
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig(os.path.join(result_dir, 'optimization_history.svg'), dpi=300, bbox_inches='tight')
    plt.close()
    
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig(os.path.join(result_dir, 'param_importance.svg'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return study


def train_and_evaluate(study, dataset, result_dir, n_runs, pre_trained_model_path=None, freeze_layers=False):
    best_params = study.best_params

    all_results = []
    roc_tprs = []
    roc_aucs = []

    best_run_f1_macro = -1.0

    mean_fpr = np.linspace(0.0, 1.0, 201)
    runs_path = os.path.join(result_dir, 'classification_metrics_runs.txt')
    with open(runs_path, 'w'):
        pass

    class_weight = compute_class_weight(dataset.target_train.values)
    with open(os.path.join(result_dir, 'best_hyperparameters.txt'), 'w') as f:
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"class_weight: {class_weight}\n")

    for run_idx in range(n_runs):
        pre_trained_model = keras.models.load_model(pre_trained_model_path) if pre_trained_model_path else None
        mlp = MLP(shape=dataset.get_shape(), pre_trained_mlp=pre_trained_model, freeze_layers=freeze_layers)
        mlp.reset_output_layer()
        mlp.compile(learning_rate=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
        
        mlp.train(
            dataset,
            epochs=best_params['epochs'],
            batch_size=best_params['batch_size'],
            verbose=1,
            plot_path=os.path.join(result_dir, f'training_curves_run_{run_idx+1}.svg'),
            class_weight=class_weight,
        )
    
        y_val_true = dataset.target_validation.values
        y_val_proba = mlp.model.predict(dataset.features_validation, verbose=0).flatten()
        best_threshold, best_val_f1 = find_best_threshold(y_val_true, y_val_proba)

        results = {}

        y_pred_proba = mlp.model.predict(dataset.features_test, verbose=0).flatten()
        y_pred_class = (y_pred_proba >= best_threshold).astype(int)
        y_true = dataset.target_test.values

        accuracy = accuracy_score(y_true, y_pred_class)
        precision_macro = precision_score(y_true, y_pred_class, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred_class, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred_class, average='macro', zero_division=0)

        conf_matrix = confusion_matrix(y_true, y_pred_class)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = auc(fpr, tpr)
        brier_score = brier_score_loss(y_true, y_pred_proba)
        ece = calculate_ece(y_true, y_pred_proba, n_bins=10)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        roc_tprs.append(interp_tpr)
        roc_aucs.append(float(auc_score))

        results['best_threshold'] = best_threshold
        results['best_val_f1_macro'] = best_val_f1
        results['accuracy'] = accuracy
        results['precision_macro'] = precision_macro
        results['recall_macro'] = recall_macro
        results['f1_macro'] = f1_macro
        results['auc_roc'] = auc_score
        results['brier_score'] = brier_score
        results['ece'] = ece

        if results['f1_macro'] > best_run_f1_macro:
            best_run_f1_macro = results['f1_macro']
            mlp.model.save(os.path.join(result_dir, 'best_model.keras'))
            

        all_results.append(results)

        with open(runs_path, 'a') as f:
            f.write(f"run: {run_idx + 1}\n")
            for key in sorted(results.keys()):
                f.write(f"{key}: {results[key]:.4f}\n")
            f.write("confusion_matrix:\n")
            f.write(np.array2string(conf_matrix))
            f.write("\n\n")

    if roc_tprs:
        tprs = np.vstack(roc_tprs)
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0, ddof=1) if len(roc_tprs) > 1 else np.zeros_like(mean_tpr)
        auc_mean = float(np.mean(roc_aucs))
        auc_std = float(np.std(roc_aucs, ddof=1)) if len(roc_aucs) > 1 else 0.0

        plt.figure(figsize=(6, 6))
        plt.plot(mean_fpr, mean_tpr, color='C0', lw=2, label=f"ROC média (AUC={auc_mean:.3f}±{auc_std:.3f})")
        plt.fill_between(
            mean_fpr,
            np.clip(mean_tpr - std_tpr, 0, 1),
            np.clip(mean_tpr + std_tpr, 0, 1),
            color='C0',
            alpha=0.2,
            label="±1 desvio padrão",
        )
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1, label='Aleatório')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('Curva ROC média (teste)')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(result_dir, 'roc_mean.svg'), dpi=300, bbox_inches='tight')
        plt.close()

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

    with open(os.path.join(result_dir, 'classification_metrics.txt'), 'w') as f:
        f.write("Métricas de Classificação no Conjunto de Teste (média):\n\n")
        f.write(f"n_runs: {n_runs}\n\n")
        for key in metric_keys:
            f.write(f"{key}_mean: {mean_results[key]:.4f}\n")
            f.write(f"{key}_std: {std_results[key]:.4f}\n")
        

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('--dataset_name', type=str, default='age_adults.csv', help='File name of the dataset (e.g., age_adults.csv)')
    args.add_argument('--target_column', type=str, default='CKD progression', help='Name of the target column in the dataset')
    args.add_argument('--source_model_path', type=str, default='', help='Path to the pre-trained model to fine-tune')
    args.add_argument('--freeze_layers', action='store_true', help='Whether to freeze layers of the pre-trained model')
    args.add_argument('--n_trials', type=int, default=20, help='Number of trials for optimization')
    args.add_argument('--load_study', action='store_true', help='Load existing study')
    args.add_argument('--study_path', type=str, default='', help='Path to the saved study')
    args.add_argument('--n_runs', type=int, default=20, help='Number of runs for training and evaluation')
    args = args.parse_args()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dataset_base = args.dataset_name.replace('.csv', '')
    result_dir = os.path.join('results', 'fine_tuning', f'{dataset_base}_{timestamp}_{"freeze" if args.freeze_layers else "unfreeze"}')
    os.makedirs(result_dir, exist_ok=True)
    
    dataset_folder_name = args.dataset_name.split('_')[0]
    
    dataset_path = f'datasets_processed/{dataset_folder_name}/{args.dataset_name}'
    dataset = Dataset(dataset_path, args.target_column)

    if args.load_study:
        study = joblib.load(args.study_path)
    else:
        study = optimize_hyperparameters(dataset_path, args.source_model_path, args.freeze_layers, args.target_column, args.n_trials, result_dir)

    train_and_evaluate(study, dataset, result_dir, args.n_runs, pre_trained_model_path=args.source_model_path, freeze_layers=args.freeze_layers)
