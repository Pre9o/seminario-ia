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
import random
import tensorflow as tf
from autoencoder import Autoencoder
import shap
import pandas as pd


def sample_hidden_layers(trial, *, prefix='', max_layers=3, first_layer_choices=None, min_neurons=2):
    if first_layer_choices is None:
        first_layer_choices = [4, 8, 16, 32, 64]

    n_layers = trial.suggest_int(f'{prefix}n_layers', 1, max_layers)
    layer_1 = trial.suggest_categorical(f'{prefix}layer_1', first_layer_choices)
    layers = [int(layer_1)]

    prev = layers[0]
    for i in range(2, n_layers + 1):
        ratio = trial.suggest_categorical(f'{prefix}layer_{i}_ratio', ['same', 'half'])
        if prev <= min_neurons or (prev // 2) < min_neurons:
            ratio = 'same'
        nxt = prev if ratio == 'same' else prev // 2
        layers.append(int(nxt))
        prev = nxt

    layers = [n for n in layers if n >= min_neurons]
    return layers


def build_hidden_layers_from_params(best_params, *, prefix='', min_neurons=2):
    n_layers = best_params.get(f'{prefix}n_layers', 1)
    layer_1 = best_params.get(f'{prefix}layer_1', 256)

    layers = [layer_1]
    prev = layer_1
    for i in range(2, n_layers + 1):
        ratio = best_params.get(f'{prefix}layer_{i}_ratio', 'same')
        if prev <= min_neurons:
            ratio = 'same'
        nxt = prev if ratio == 'same' else prev // 2
        if nxt < min_neurons:
            break
        layers.append(int(nxt))
        prev = nxt

    return layers


def sample_dropout_rates(trial, n_layers, *, prefix='', min_rate=0.0, max_rate=0.5, step=0.1):
    return [trial.suggest_float(f'{prefix}dropout_{i}', min_rate, max_rate, step=step) for i in range(1, n_layers + 1)]


def build_dropout_rates_from_params(best_params, n_layers, *, prefix='', default_rate=0.0):
    rates = []
    for i in range(1, n_layers + 1):
        rates.append(best_params.get(f'{prefix}dropout_{i}', default_rate))
    return rates


def compute_class_weight(y):
    y = np.asarray(y).astype(int)
    counts = np.bincount(y, minlength=2)
    if counts[0] == 0 or counts[1] == 0:
        return None
    total = counts.sum()
    return {0: total / (2.0 * counts[0]), 1: total / (2.0 * counts[1])}


def find_best_threshold(y_true, y_proba):
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


def set_global_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        tf.random.set_seed(seed)


def get_categorical_info(dataset_base_folder):
    if dataset_base_folder == 'age':
        return [0, 1], {0: 4, 1: 2}
    elif dataset_base_folder == 'etiology':
        return [0], {0: 2}
    elif dataset_base_folder == 'stage':
        return [0, 1], {0: 4, 1: 2}
    return [], {}


def objective(trial, dataset_source, dataset_target, training_type, dataset_base_folder):
    if training_type == 'baseline':
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        hidden_layers = sample_hidden_layers(trial, max_layers=3, min_neurons=2, first_layer_choices=[4, 8, 16, 32])
        dropout_rates = sample_dropout_rates(trial, len(hidden_layers), min_rate=0.0, max_rate=0.5, step=0.1)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
        epochs = trial.suggest_int('epochs', 50, 1000, step=50)
        
        mlp = MLP(shape=dataset_target.get_shape(), layers=hidden_layers, dropout_rate=dropout_rates)
        mlp.compile(learning_rate=learning_rate, weight_decay=weight_decay)
        class_weight = compute_class_weight(dataset_target.target_train.values)
        mlp.train(dataset_target, epochs=epochs, batch_size=batch_size, verbose=0, plot_path=None, class_weight=class_weight)

        y_val_true = dataset_target.target_validation.values
        y_val_proba = mlp.model.predict(dataset_target.features_validation, verbose=0).flatten()
        _, val_f1 = find_best_threshold(y_val_true, y_val_proba)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return val_f1

    categorical_indices, categorical_cardinalities = get_categorical_info(dataset_base_folder)

    pretrain_hidden_units = sample_hidden_layers(trial, prefix='ae_', max_layers=5, min_neurons=16, first_layer_choices=[8, 16, 32, 64, 128])
    
    pretrain_mask_ratio = trial.suggest_float('mask_ratio', 0.1, 0.5)
    pretrain_lr = trial.suggest_float('pretrain_lr', 1e-4, 1e-2, log=True)
    pretrain_batch_size = trial.suggest_categorical('pretrain_batch_size', [4, 8, 16, 32, 64])
    pretrain_epochs = trial.suggest_int('pretrain_epochs', 50, 300, step=50)
    pretrain_dropout_rate = trial.suggest_float('ae_dropout_rate', 0.1, 0.5)
    pretrain_l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)
    
    autoencoder = Autoencoder(
        shape=dataset_source.get_shape(),
        categorical_indices=categorical_indices,
        categorical_cardinalities=categorical_cardinalities,
        mask_ratio=pretrain_mask_ratio,
        hidden_units=pretrain_hidden_units,
        learning_rate=pretrain_lr,
        mask_value=-999.0,
        dropout_rate=pretrain_dropout_rate,
        l2_reg=pretrain_l2_reg
    )

    autoencoder.train(dataset_source, epochs=pretrain_epochs, batch_size=pretrain_batch_size, verbose=0)

    ft_learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    ft_batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    ft_epochs = trial.suggest_int('epochs', 50, 500, step=50)
    
    mlp = MLP(shape=dataset_target.get_shape(), pretrained_encoder=autoencoder.encoder)
    if training_type == 'unfrozen':
        mlp.unfreeze_encoder(learning_rate=ft_learning_rate)
    else:
        mlp.compile(learning_rate=ft_learning_rate)
    class_weight = compute_class_weight(dataset_target.target_train.values)
    mlp.train(dataset_target, epochs=ft_epochs, batch_size=ft_batch_size, verbose=0, plot_path=None, class_weight=class_weight)

    y_val_true = dataset_target.target_validation.values
    y_val_proba = mlp.model.predict(dataset_target.features_validation, verbose=0).flatten()
    _, val_f1 = find_best_threshold(y_val_true, y_val_proba)
    
    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_f1


def optimize_hyperparameters(dataset_source, dataset_target, training_type, dataset_base_folder, n_trials, result_dir):
    study = optuna.create_study(
        study_name=f'{training_type}_optimization',
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    study.optimize(
        lambda trial: objective(trial, dataset_source, dataset_target, training_type, dataset_base_folder),
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


def calculate_autoencoder_metrics(autoencoder, dataset, categorical_indices):
    test_array = dataset.features_test.values if hasattr(dataset.features_test, 'values') else dataset.features_test
    X_test_masked, _ = autoencoder.create_masked_data(test_array)
    y_test_targets = autoencoder.prepare_targets(test_array)
    predictions = autoencoder.model.predict(X_test_masked, verbose=0)

    results = {}
    total_loss = 0.0

    if autoencoder.continuous_indices:
        continuous_pred = predictions[0]
        continuous_true = y_test_targets[0]
        mse = np.mean((continuous_pred - continuous_true) ** 2)
        mae = np.mean(np.abs(continuous_pred - continuous_true))
        results['continuous_mse'] = float(mse)
        results['continuous_mae'] = float(mae)
        total_loss += mse + mae

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
            total_loss += ce_loss

    results['total_loss'] = total_loss
    return results


def train_and_evaluate(study, dataset_source, dataset_target, training_type, dataset_base_folder, result_dir, n_runs):
    best_params = study.best_params
    
    all_results = []
    all_ae_results = []
    roc_tprs = []
    roc_aucs = []
    best_run_f1_macro = -1.0
    best_ae_loss = float('inf')

    mean_fpr = np.linspace(0.0, 1.0, 201)
    runs_path = os.path.join(result_dir, 'classification_metrics_runs.txt')
    with open(runs_path, 'w'):
        pass

    class_weight = compute_class_weight(dataset_target.target_train.values)
    
    with open(os.path.join(result_dir, 'best_hyperparameters.txt'), 'w') as f:
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"class_weight: {class_weight}\n")

    categorical_indices, categorical_cardinalities = get_categorical_info(dataset_base_folder)

    for run_idx in range(n_runs):
        set_global_seed(run_idx)

        if training_type == 'baseline':
            hidden_layers = build_hidden_layers_from_params(best_params, min_neurons=2)
            dropout_rates = build_dropout_rates_from_params(best_params, len(hidden_layers), default_rate=0.0)
            
            mlp = MLP(shape=dataset_target.get_shape(), layers=hidden_layers, dropout_rate=dropout_rates)
            mlp.compile(learning_rate=best_params['learning_rate'], weight_decay=best_params['weight_decay'])
            
            mlp.train(
                dataset_target,
                epochs=best_params['epochs'],
                batch_size=best_params['batch_size'],
                verbose=1,
                plot_path=os.path.join(result_dir, f'training_curves_run_{run_idx+1}.svg'),
                class_weight=class_weight,
            )
        else:
            ae_hidden_units = build_hidden_layers_from_params(best_params, prefix='ae_', min_neurons=16)
            
            autoencoder = Autoencoder(
                shape=dataset_source.get_shape(),
                categorical_indices=categorical_indices,
                categorical_cardinalities=categorical_cardinalities,
                mask_ratio=best_params['mask_ratio'],
                hidden_units=ae_hidden_units,
                learning_rate=best_params['pretrain_lr'],
                mask_value=-999.0,
                dropout_rate=best_params['ae_dropout_rate'],
                l2_reg=best_params['l2_reg']
            )

            autoencoder.train(
                dataset_source,
                epochs=best_params['pretrain_epochs'],
                batch_size=best_params['pretrain_batch_size'],
                verbose=1,
                plot_path=os.path.join(result_dir, f'ae_training_curves_run_{run_idx+1}.svg')
            )

            ae_metrics = calculate_autoencoder_metrics(autoencoder, dataset_source, categorical_indices)
            all_ae_results.append(ae_metrics)

            if ae_metrics['total_loss'] < best_ae_loss:
                best_ae_loss = ae_metrics['total_loss']
                autoencoder.save_encoder(os.path.join(result_dir, 'best_encoder.keras'))

            mlp = MLP(shape=dataset_target.get_shape(), pretrained_encoder=autoencoder.encoder)
            if training_type == 'unfrozen':
                mlp.unfreeze_encoder(learning_rate=best_params['learning_rate'])
            else:
                mlp.compile(learning_rate=best_params['learning_rate'])
            
            mlp.train(
                dataset_target,
                epochs=best_params['epochs'],
                batch_size=best_params['batch_size'],
                verbose=1,
                plot_path=os.path.join(result_dir, f'training_curves_run_{run_idx+1}.svg'),
                class_weight=class_weight,
            )

        y_val_true = dataset_target.target_validation.values
        y_val_proba = mlp.model.predict(dataset_target.features_validation, verbose=0).flatten()
        best_threshold, best_val_f1 = find_best_threshold(y_val_true, y_val_proba)

        results = {}

        y_pred_proba = mlp.model.predict(dataset_target.features_test, verbose=0).flatten()
        y_pred_class = (y_pred_proba >= best_threshold).astype(int)
        y_true = dataset_target.target_test.values

        accuracy = accuracy_score(y_true, y_pred_class)
        precision_macro = precision_score(y_true, y_pred_class, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred_class, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred_class, average='macro', zero_division=0)

        conf_matrix = confusion_matrix(y_true, y_pred_class)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = auc(fpr, tpr)
        brier_score = brier_score_loss(y_true, y_pred_proba)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        roc_tprs.append(interp_tpr)
        roc_aucs.append(auc_score)

        results['best_threshold'] = best_threshold
        results['best_val_f1_macro'] = best_val_f1
        results['accuracy'] = accuracy
        results['precision_macro'] = precision_macro
        results['recall_macro'] = recall_macro
        results['f1_macro'] = f1_macro
        results['auc_roc'] = auc_score
        results['brier_score'] = brier_score

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

    if all_ae_results:
        ae_metric_keys = sorted({k for r in all_ae_results for k in r.keys()})
        ae_mean_results = {}
        ae_std_results = {}
        
        for k in ae_metric_keys:
            vals = [r[k] for r in all_ae_results if k in r]
            ae_mean_results[k] = float(np.mean(vals)) if vals else float('nan')
            if not vals:
                ae_std_results[k] = float('nan')
            elif len(vals) == 1:
                ae_std_results[k] = 0.0
            else:
                ae_std_results[k] = float(np.std(vals, ddof=1))

        with open(os.path.join(result_dir, 'reconstruction_metrics.txt'), 'w') as f:
            f.write("Métricas de Reconstrução do Autoencoder (média):\n\n")
            f.write(f"n_runs: {n_runs}\n\n")
            for key in ae_metric_keys:
                f.write(f"{key}_mean: {ae_mean_results[key]:.4f}\n")
                f.write(f"{key}_std: {ae_std_results[key]:.4f}\n")

    best_model = tf.keras.models.load_model(os.path.join(result_dir, 'best_model.keras'))
    compute_shap_values(best_model, dataset_target, result_dir)


def compute_shap_values(model, dataset, result_dir):
    X_train = dataset.features_train.values if hasattr(dataset.features_train, 'values') else dataset.features_train
    X_test = dataset.features_test.values if hasattr(dataset.features_test, 'values') else dataset.features_test
    feature_names = dataset.features_train.columns.tolist() if hasattr(dataset.features_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]

    background = X_train[np.random.choice(X_train.shape[0], min(100, X_train.shape[0]), replace=False)]
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.array(shap_values).squeeze()

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    feature_importance.to_csv(os.path.join(result_dir, 'shap_feature_importance.csv'), index=False)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'shap_summary.svg'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'shap_bar.svg'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('--dataset_source_name', type=str, default='age_elderly', help='File name of the dataset (e.g., age_adults)')
    args.add_argument('--dataset_target_name', type=str, default='age_adults', help='File name of the dataset (e.g., age_adults)')
    args.add_argument('--training_type', choices=['frozen', 'unfrozen', 'baseline'], required=True, help='Training type (frozen, unfrozen, baseline)')
    args.add_argument('--target_column', type=str, default='CKD progression', help='Name of the target column in the dataset')
    args.add_argument('--n_trials', type=int, default=20, help='Number of trials for optimization')
    args.add_argument('--load_study', action='store_true', help='Load existing study')
    args.add_argument('--study_path', type=str, default='', help='Path to the saved study')
    args.add_argument('--n_runs', type=int, default=20, help='Number of runs for training and evaluation')
    args = args.parse_args()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dataset_base_folder = args.dataset_target_name.split('_')[0]
    result_dir = os.path.join('results', f'{args.training_type}', f'{args.dataset_target_name}_{timestamp}')
    os.makedirs(result_dir, exist_ok=True)

    dataset_folder_name = args.dataset_target_name.split('_')[0]
    
    dataset_source_path = f'datasets_processed_2/{dataset_folder_name}/{args.dataset_source_name}'
    dataset_source = Dataset(dataset_source_path, args.target_column)

    dataset_target_path = f'datasets_processed_2/{dataset_folder_name}/{args.dataset_target_name}'
    dataset_target = Dataset(dataset_target_path, args.target_column)

    if args.load_study:
        study = joblib.load(args.study_path)
    else:
        study = optimize_hyperparameters(dataset_source, dataset_target, args.training_type, dataset_base_folder, args.n_trials, result_dir)

    train_and_evaluate(study, dataset_source, dataset_target, args.training_type, dataset_base_folder, result_dir, args.n_runs)