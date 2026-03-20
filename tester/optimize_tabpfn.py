import os
import optuna
from optuna.pruners import MedianPruner
from dataset import Dataset
from tabpfn import TabPFNClassifier
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, brier_score_loss,
    f1_score, accuracy_score, precision_score, recall_score,
)
import numpy as np
from argparse import ArgumentParser
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import shap
import pandas as pd


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
            best_threshold = float(t)

    return best_threshold, float(best_score)

def get_categorical_info(dataset_base_folder):
    if dataset_base_folder == 'age':
        return [0, 1], {0: 4, 1: 2}
    elif dataset_base_folder == 'etiology':
        return [0], {0: 2}
    elif dataset_base_folder == 'stage':
        return [0, 1], {0: 4, 1: 2}
    return [], {}



def objective(trial, dataset_target, dataset_base_folder):
    categorical_indices, categorical_cardinalities = get_categorical_info(dataset_base_folder)

    n_estimators = trial.suggest_int('n_estimators', 2, 32)
    softmax_temperature = trial.suggest_float('softmax_temperature', 0.1, 5.0, log=True)
    balance_probabilities = trial.suggest_categorical('balance_probabilities', [True, False])
    average_before_softmax = trial.suggest_categorical('average_before_softmax', [True, False])
    ignore_pretraining_limits = trial.suggest_categorical('ignore_pretraining_limits', [True, False])

    X_train = dataset_target.features_train.values
    y_train = dataset_target.target_train.values

    X_val = dataset_target.features_validation.values
    y_val = dataset_target.target_validation.values

    # class_weight = compute_class_weight(y_train)
    # if class_weight is not None:
    #     sample_weight = np.array([class_weight[int(c)] for c in y_train])
    # else:
    #     sample_weight = None

    try:
        clf = TabPFNClassifier(
            n_estimators=n_estimators,
            categorical_features_indices=categorical_indices,
            softmax_temperature=softmax_temperature,
            balance_probabilities=balance_probabilities,
            average_before_softmax=average_before_softmax,
            ignore_pretraining_limits=ignore_pretraining_limits,
        )

        clf.fit(X_train, y_train,)

        y_val_proba = clf.predict_proba(X_val)[:, 1]
        _, val_f1 = find_best_threshold(y_val, y_val_proba)
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0

    return val_f1


def optimize_hyperparameters(dataset_target, n_trials, result_dir):
    sampler = optuna.samplers.TPESampler()

    study = optuna.create_study(
        study_name='tabpfn_optimization',
        direction='maximize',
        sampler=sampler,
    )

    study.optimize(
        lambda trial: objective(trial, dataset_target, args.dataset_target_name.split('_')[0]),
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


def train_and_evaluate(study, dataset_target, result_dir, n_runs, dataset_base_folder):
    categorical_indices, categorical_cardinalities = get_categorical_info(dataset_base_folder)

    best_params = study.best_params

    all_results = []
    roc_tprs = []
    roc_aucs = []
    best_run_f1_macro = -1.0
    best_clf = None

    mean_fpr = np.linspace(0.0, 1.0, 201)
    runs_path = os.path.join(result_dir, 'classification_metrics_runs.txt')
    with open(runs_path, 'w'):
        pass

    X_train = dataset_target.features_train.values
    y_train = dataset_target.target_train.values

    # class_weight = compute_class_weight(y_train)
    # if class_weight is not None:
    #     sample_weight = np.array([class_weight[int(c)] for c in y_train])
    # else:
    #     sample_weight = None

    with open(os.path.join(result_dir, 'best_hyperparameters.txt'), 'w') as f:
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
        # f.write(f"class_weight: {class_weight}\n")

    for run_idx in range(n_runs):
        clf = TabPFNClassifier(
            n_estimators=best_params['n_estimators'],
            categorical_features_indices=categorical_indices,
            softmax_temperature=best_params['softmax_temperature'],
            balance_probabilities=best_params['balance_probabilities'],
            average_before_softmax=best_params['average_before_softmax'],
            ignore_pretraining_limits=best_params['ignore_pretraining_limits'],
            random_state=run_idx
        )

        clf.fit(X_train, y_train)

        y_val_true = dataset_target.target_validation.values
        y_val_proba = clf.predict_proba(dataset_target.features_validation.values)[:, 1]
        best_threshold, best_val_f1 = find_best_threshold(y_val_true, y_val_proba)

        results = {}

        y_pred_proba = clf.predict_proba(dataset_target.features_test.values)[:, 1]
        y_pred_class = (y_pred_proba >= best_threshold).astype(int)
        y_true = dataset_target.target_test.values

        accuracy = accuracy_score(y_true, y_pred_class)
        precision_macro = precision_score(y_true, y_pred_class, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred_class, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred_class, average='macro', zero_division=0)

        conf_matrix = confusion_matrix(y_true, y_pred_class)
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = auc(fpr, tpr)
        brier = brier_score_loss(y_true, y_pred_proba)

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
        results['brier_score'] = brier

        if results['f1_macro'] > best_run_f1_macro:
            best_run_f1_macro = results['f1_macro']
            best_clf = clf
            joblib.dump(clf, os.path.join(result_dir, 'best_model.pkl'))

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

    if best_clf is not None:
        compute_shap_values(best_clf, dataset_target, result_dir)


def compute_shap_values(model, dataset, result_dir):
    X_train = dataset.features_train.values if hasattr(dataset.features_train, 'values') else dataset.features_train
    X_test = dataset.features_test.values if hasattr(dataset.features_test, 'values') else dataset.features_test
    feature_names = (
        dataset.features_train.columns.tolist()
        if hasattr(dataset.features_train, 'columns')
        else [f'feature_{i}' for i in range(X_train.shape[1])]
    )

    rng = np.random.default_rng(0)

    background_size = min(100, X_train.shape[0])
    test_size = min(500, X_test.shape[0])
    background = X_train[rng.choice(X_train.shape[0], background_size, replace=False)]
    X_test_sample = X_test[rng.choice(X_test.shape[0], test_size, replace=False)]

    explainer_name = 'KernelExplainer'

    def _predict_proba(x):
        return model.predict_proba(x)[:, 1]

    try:
        explainer = shap.KernelExplainer(_predict_proba, background)
        shap_values = explainer.shap_values(X_test_sample)
    except Exception:
        explainer_name = 'PermutationExplainer'
        masker = shap.maskers.Independent(background)
        explanation = shap.PermutationExplainer(_predict_proba, masker)(X_test_sample)
        shap_values = explanation.values

    with open(os.path.join(result_dir, 'shap_explainer.txt'), 'w') as f:
        f.write(f"explainer: {explainer_name}\n")
        f.write(f"background_size: {background_size}\n")
        f.write(f"test_sample_size: {test_size}\n")

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.array(shap_values).squeeze()

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
    }).sort_values('mean_abs_shap', ascending=False)
    feature_importance.to_csv(os.path.join(result_dir, 'shap_feature_importance.csv'), index=False)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'shap_summary.svg'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'shap_bar.svg'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('--dataset_target_name', type=str, default='age_adults', help='File name of the target dataset (e.g., age_adults)')
    args.add_argument('--target_column', type=str, default='CKD progression', help='Name of the target column in the dataset')
    args.add_argument('--n_trials', type=int, default=20, help='Number of trials for optimization')
    args.add_argument('--load_study', action='store_true', help='Load existing study')
    args.add_argument('--study_path', type=str, default='', help='Path to the saved study')
    args.add_argument('--n_runs', type=int, default=20, help='Number of runs for training and evaluation')
    args = args.parse_args()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_dir = os.path.join('results', 'tabpfn', f'{args.dataset_target_name}_{timestamp}')
    os.makedirs(result_dir, exist_ok=True)

    dataset_folder_name = args.dataset_target_name.split('_')[0]

    dataset_target_path = f'datasets_processed_2/{dataset_folder_name}/{args.dataset_target_name}'
    dataset_target = Dataset(dataset_target_path, args.target_column)

    if args.load_study:
        study = joblib.load(args.study_path)
    else:
        study = optimize_hyperparameters(dataset_target, args.n_trials, result_dir)

    train_and_evaluate(study, dataset_target, result_dir, args.n_runs, dataset_folder_name)
