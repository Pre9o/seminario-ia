import os
from old.exemplo import plot
import optuna
from optuna.pruners import MedianPruner
from dataset import Dataset
from mlp_modified import MLP
from sklearn.metrics import f1_score, recall_score, roc_curve, auc, classification_report, confusion_matrix
from tensorflow import keras
import numpy as np
from argparse import ArgumentParser
import joblib
import matplotlib.pyplot as plt
import time
from datetime import datetime

def objective(trial, dataset_path, encoder_path, target_column='CKD progression'):
    dataset = Dataset(dataset_path, target_column)
    
    pretrained_encoder = keras.models.load_model(encoder_path)
    
    finetune_lr = trial.suggest_float('finetune_lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    finetune_epochs = trial.suggest_int('finetune_epochs', 50, 300, step=50)
    
    try:
        mlp = MLP(
            shape=dataset.get_shape(),
            pretrained_encoder=pretrained_encoder,
            freeze_encoder=True
        )
        
        mlp.compile(learning_rate=finetune_lr)
        
        mlp.train(
            dataset,
            epochs=finetune_epochs,
            batch_size=batch_size,
            verbose=0,
            plot_path=None
        )
        
        y_pred = mlp.model.predict(dataset.features_validation, verbose=0)
        y_pred_class = (y_pred > 0.5).astype(int).flatten()
        y_true = dataset.target_validation.values

        print(f"Distribuição das predições: {np.unique(y_pred_class, return_counts=True)}")
        print(f"Distribuição dos targets: {np.unique(y_true, return_counts=True)}")
        recall = recall_score(y_true, y_pred_class)
        print(f"Recall: {recall:.4f}")
        
        f1 = f1_score(y_true, y_pred_class)
        
        trial.report(f1, step=0)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return f1
    
    except Exception as e:
        print(f"ERRO no trial {trial.number}: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def optimize_hyperparameters(
    dataset_name,
    encoder_path='best_pretrained_encoder.keras',
    target_column='CKD progression',
    n_trials=50,
    study_name='frozen_optimization',
    timestamp=None,
    directories=None
):
    dataset_path = f'datasets/dataset_filled_boruta_{dataset_name}'
    

    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    study.optimize(
        lambda trial: objective(trial, dataset_path, encoder_path, target_column),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print("OTIMIZAÇÃO CONCLUÍDA")
    print(f"Número de trials concluídos: {len(study.trials)}\n")
    print(f"Melhor F1-score: {study.best_value:.4f}")
    
    print("Melhores hiperparâmetros:\n")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    study_filename = f'{study_name}_{timestamp}.pkl'
    joblib.dump(study, os.path.join(directories['study_dir'], study_filename))
    print(f"Estudo salvo em: {os.path.join(directories['study_dir'], study_filename)}\n")
    
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig(os.path.join(directories['plot_dir'], f'optuna_frozen_history_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig(os.path.join(directories['plot_dir'], f'optuna_frozen_importance_{timestamp}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return study


def train_with_best_params(
    study,
    dataset,
    encoder_path='best_pretrained_encoder.keras',
    timestamp=None,
    plot_path=None
):
    plot_path = plot_path + f'/training_curves_frozen_{timestamp}.png'
    best_params = study.best_params
    
    pretrained_encoder = keras.models.load_model(encoder_path)
    
    mlp = MLP(
        shape=dataset.get_shape(),
        pretrained_encoder=pretrained_encoder,
        freeze_encoder=True
    )
    
    mlp.compile(learning_rate=best_params['finetune_lr'])
    
    mlp.train(
        dataset,
        epochs=best_params['finetune_epochs'],
        batch_size=best_params['batch_size'],
        verbose=1,
        plot_path=plot_path
    )
    
    return mlp

def save_results(dataset_name='age_adults.csv', best_model=None, dataset=None, timestamp=None, directories=None):
    if best_model is None or dataset is None:
        print("Modelo ou dataset não fornecido para salvar os resultados.")
        return
    
    model_path = os.path.join(directories['model_dir'], f'best_frozen_model_{dataset_name.replace(".csv", "")}_{timestamp}.keras')

    best_model.model.save(model_path)

    y_pred = best_model.model.predict(dataset.features_test)
    y_pred_class = (y_pred > 0.5).astype(int).flatten()
    y_true = dataset.target_test.values

    classification_report_path = os.path.join(directories['model_dir'], f'classification_report_{dataset_name.replace(".csv", "")}_{timestamp}.txt')

    class_report = classification_report(y_true, y_pred_class, digits=4)
    conf_matrix = confusion_matrix(y_true, y_pred_class)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)

    with open(classification_report_path, 'w') as f:
        f.write(class_report)
        f.write("\nMatriz de Confusão:\n")
        f.write(np.array2string(conf_matrix))
        f.write(f"\nAUC-ROC: {auc_score:.4f}\n")

    print("RELATÓRIO DE CLASSIFICAÇÃO DO MELHOR MODELO OTIMIZADO:\n")
    print(class_report)

    print("MATRIZ DE CONFUSÃO DO MELHOR MODELO OTIMIZADO:\n")
    print(conf_matrix)

    print("CURVA ROC AUC DO MELHOR MODELO OTIMIZADO:\n")
    print(f"AUC-ROC: {auc_score:.4f}")

def set_directories(dataset_name):
    directories = {'results_dir': 'results',
                   'frozen_dir': os.path.join('results', 'frozen'),
                   'model_dir': os.path.join('results', 'frozen', dataset_name.replace('.csv', '')),
                   'plot_dir': os.path.join('results', 'frozen', dataset_name.replace('.csv', ''), 'plots'),
                   'study_dir': os.path.join('results', 'frozen', dataset_name.replace('.csv', ''), 'studies')}
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)

    return directories

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('--dataset_name', type=str, default='age_adults.csv')
    args.add_argument('--target_column', type=str, default='CKD progression')
    args.add_argument('--encoder_path', type=str, default='best_models_27_01_2026/best_pretrained_encoder.keras')
    args.add_argument('--n_trials', type=int, default=20)
    args.add_argument('--load_study', action='store_true')
    args.add_argument('--study_name', type=str, default=None)
    args = args.parse_args()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dataset = Dataset(f'datasets/dataset_filled_boruta_{args.dataset_name}', args.target_column)
    directories = set_directories(args.dataset_name)
    study_path = os.path.join(directories['study_dir'], args.study_name) if args.study_name else None

    if args.load_study:
        study = joblib.load(study_path)
        print(f"Estudo carregado de: {study_path}")
    else:
        study = optimize_hyperparameters(
            dataset_name=args.dataset_name,
            encoder_path=args.encoder_path,
            target_column=args.target_column,
            n_trials=args.n_trials,
            study_name='frozen_optimization',
            timestamp=timestamp,
            directories=directories
        )

    
    best_model = train_with_best_params(
        study,
        dataset,
        encoder_path=args.encoder_path,
        timestamp=timestamp,
        plot_path=directories['plot_dir']
    )

    save_results(dataset_name=args.dataset_name, best_model=best_model, dataset=dataset, timestamp=timestamp, directories=directories)

