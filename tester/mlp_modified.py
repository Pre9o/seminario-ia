import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report   

HIDDEN_LAYERS=[
    128,
    128,
    128,
    128,
    128,
]
DROPOUTS=[
    0.2,
    0.2,
    0.2,
    0.2,
    0.2,
]
LEARNING_RATE=0.001

# HIDDEN_LAYERS=[
#     260,
#     1008,
#     1008,
#     1008,
#     1008,
#     1008,
#     954,
# ]
# DROPOUTS=[
#     0.1901307611036021,
#     0.1901307611036021,
#     0.1901307611036021,
#     0.1901307611036021,
#     0.1901307611036021,
#     0.1901307611036021,
#     0.1901307611036021,
# ]
# LEARNING_RATE=0.0002716544410603358
WEIGHT_DECAY=0.000
ACTIVATION='relu'
LOSS='binary_crossentropy'

class MacroF1Callback(keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.val_f1_macro = []

    def find_best_threshold(self, y_true, y_proba):
        thresholds = np.linspace(0.0, 1.0, 101)
        best_threshold = 0.5
        best_score = -1.0

        for threshold in thresholds:
            y_pred = (y_proba > threshold).astype(int).flatten()
            score = f1_score(y_true, y_pred, average='macro', zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold
    
    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred_proba = self.model.predict(X_val, verbose=0)
        best_threshold = self.find_best_threshold(y_val, y_pred_proba)
        y_pred = (y_pred_proba > best_threshold).astype(int).flatten()
        y_true = y_val.values if hasattr(y_val, 'values') else y_val
        
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        self.val_f1_macro.append(f1_macro)
        logs['val_f1_macro'] = f1_macro
        

class RestoreBestF1(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_f1 = -np.inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_f1 = logs.get("val_f1_macro")

        if current_f1 is not None and current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)

class MLP:
    def __init__(self, shape, layers=HIDDEN_LAYERS, activation=ACTIVATION, dropout_rate=0.2, pretrained_encoder=None, pre_trained_mlp=None, freeze_layers=False):
        self.model = keras.models.Sequential()
        self.shape = shape
        
        if pretrained_encoder is not None:
            print("Usando encoder pré-treinado!")
            self.model.add(pretrained_encoder)
            
            for layer in pretrained_encoder.layers:
                layer.trainable = False
                for sublayer in getattr(layer, "layers", []):
                    sublayer.trainable = False
        elif pre_trained_mlp is not None:
            print("Usando MLP pré-treinado!")
            self.model = pre_trained_mlp
            if freeze_layers:
                for layer in self.model.layers:
                    if layer.name != 'camada_saida_sigmoid':
                        layer.trainable = False
                        if hasattr(layer, "layers"):
                            for sublayer in layer.layers:
                                sublayer.trainable = False
            else:
                for layer in self.model.layers:
                    layer.trainable = True
                    if hasattr(layer, "layers"):
                        for sublayer in layer.layers:
                            sublayer.trainable = True
        else:
            self.model.add(keras.layers.InputLayer(input_shape=(self.shape,), name='input_layer'))
            for idx, neurons in enumerate(layers):
                current_dropout = dropout_rate[idx] if isinstance(dropout_rate, (list, tuple, np.ndarray)) else dropout_rate
                self.model.add(
                    keras.layers.Dense(
                        neurons,
                        activation=activation,
                        name=f'camada_oculta_{idx+1}_{activation}'
                    )
                )
                self.model.add(keras.layers.Dropout(current_dropout, name=f'dropout_{idx+1}'))

        if pre_trained_mlp is None:
            self.model.add(keras.layers.Dense(1, activation='sigmoid', name='camada_saida_sigmoid'))

        self.history = None

        # self.model.summary()

    def reset_output_layer(self, units=1, activation='sigmoid', layer_name='camada_saida_sigmoid'):
        if self.model is None:
            return None

        if isinstance(self.model, keras.Sequential):
            if len(self.model.layers) == 0:
                return self.model
            self.model.pop()
            self.model.add(keras.layers.Dense(units, activation=activation, name=layer_name))
            return self.model

        if not getattr(self.model, 'inputs', None) or len(getattr(self.model, 'layers', [])) < 2:
            return self.model
        
        penultimate = self.model.layers[-2].output
        out = keras.layers.Dense(units, activation=activation, name=layer_name)(penultimate)
        return keras.Model(inputs=self.model.inputs, outputs=out, name=self.model.name)

    
    def unfreeze_encoder(self, learning_rate=0.0001, weight_decay=0.0001):
        for layer in self.model.layers:
            if layer.name == 'encoder':
                layer.trainable = True
                for sublayer in getattr(layer, "layers", []):
                    sublayer.trainable = True

        # self.model.summary()
        self.compile(learning_rate=learning_rate, weight_decay=weight_decay)
    

    def compile(self, loss=LOSS, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
        self.model.compile(
            loss=loss,
            optimizer=keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=weight_decay
            ),
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Recall(name='recall', thresholds=0.5),
                keras.metrics.Precision(name='precision', thresholds=0.5),
                keras.metrics.F1Score(name='f1-score', threshold=0.5, dtype=tf.float32)]       
        )

    def summary(self):
        self.model.summary()

    def train(
        self,
        dataset,
        epochs=5000,
        batch_size=32,
        verbose=0,
        plot_path=None,
        class_weight=None
    ):
        early_stopping = EarlyStopping(
            monitor='val_auc',
            patience=50,
            restore_best_weights=False,
            mode='max',
            verbose=1
        )

        macro_f1_callback = MacroF1Callback(
            validation_data=(dataset.features_validation, dataset.target_validation)
        )

        restore_best_f1 = RestoreBestF1()

        y_train = dataset.target_train.astype('float32')
        y_val = dataset.target_validation.astype('float32')

        history = self.model.fit(
            dataset.features_train,
            y_train,
            epochs=epochs, 
            validation_data=(dataset.features_validation, y_val),
            callbacks=[macro_f1_callback, early_stopping, restore_best_f1],
            batch_size=batch_size,
            verbose=verbose,
            class_weight=class_weight
        )

        self.history = history.history 
        self.history['val_f1_macro'] = macro_f1_callback.val_f1_macro

        self.plot_training_curves(plot_path)
        actual_epochs = len(history.history['loss'])
        print(f"\nTreinamento interrompido na época: {actual_epochs}")

        # Extrai valores escalares (alguns podem vir como arrays)
        def extract_value(v):
            return v.item() if hasattr(v, 'item') else (v[0] if isinstance(v, (list, np.ndarray)) else v)
        
        train_acc = extract_value(history.history['accuracy'][-1])
        val_acc = extract_value(history.history['val_accuracy'][-1])
        train_loss = extract_value(history.history['loss'][-1])
        val_loss = extract_value(history.history['val_loss'][-1])
        train_precision = extract_value(history.history['precision'][-1])
        val_precision = extract_value(history.history['val_precision'][-1])
        train_recall = extract_value(history.history['recall'][-1])
        val_recall = extract_value(history.history['val_recall'][-1])
        train_f1_score = extract_value(history.history['f1-score'][-1])
        val_f1_score = extract_value(history.history['val_f1-score'][-1])
        train_f1_macro = extract_value(history.history['val_f1_macro'][-1])
        val_f1_macro = extract_value(history.history['val_f1_macro'][-1])
        train_auc = extract_value(history.history['auc'][-1])
        val_auc = extract_value(history.history['val_auc'][-1])
        

        if train_auc - val_auc > 0.05:
            print("\nPossível overfitting detectado: a acurácia de treino está significativamente maior que a de validação.")
            print(f"Acurácia final - Treino: {train_acc:.2%} | Validação: {val_acc:.2%}")
            print(f"Perda final    - Treino: {train_loss:.4f} | Validação: {val_loss:.4f}")
            print(f"Precisão final    - Treino: {train_precision:.4f} | Validação: {val_precision:.4f}")
            print(f"Recall final    - Treino: {train_recall:.4f} | Validação: {val_recall:.4f}")
            print(f"F1 Score final    - Treino: {train_f1_score:.4f} | Validação: {val_f1_score:.4f}")
            print(f"AUC final    - Treino: {train_auc:.4f} | Validação: {val_auc:.4f}")
        elif val_loss > train_loss * 1.2:
            print("\nPossível overfitting detectado: a perda de validação está significativamente maior que a de treino.")
            print(f"Acurácia final - Treino: {train_acc:.2%} | Validação: {val_acc:.2%}")
            print(f"Perda final    - Treino: {train_loss:.4f} | Validação: {val_loss:.4f}")
            print(f"Precisão final    - Treino: {train_precision:.4f} | Validação: {val_precision:.4f}")
            print(f"Recall final    - Treino: {train_recall:.4f} | Validação: {val_recall:.4f}")
            print(f"F1 Score final    - Treino: {train_f1_score:.4f} | Validação: {val_f1_score:.4f}")
            print(f"AUC final    - Treino: {train_auc:.4f} | Validação: {val_auc:.4f}")
        else:
            print("\nNão há sinais claros de overfitting.")
            print(f"Acurácia final - Treino: {train_acc:.2%} | Validação: {val_acc:.2%}")
            print(f"Perda final    - Treino: {train_loss:.4f} | Validação: {val_loss:.4f}")
            print(f"Precisão final    - Treino: {train_precision:.4f} | Validação: {val_precision:.4f}")
            print(f"Recall final    - Treino: {train_recall:.4f} | Validação: {val_recall:.4f}")
            print(f"F1 Score final    - Treino: {train_f1_score:.4f} | Validação: {val_f1_score:.4f}")
            print(f"AUC final    - Treino: {train_auc:.4f} | Validação: {val_auc:.4f}")
        
        return history.history

    def plot_training_curves(self, plot_path='training_curves.svg', figsize=(12, 5)):
        """
        Plota e salva as curvas de treinamento (loss e accuracy).
        
        Args:
            plot_path (str): Nome do arquivo para salvar o gráfico
            figsize (tuple): Tamanho da figura (largura, altura)
        """
        if self.history is None:
            print("Erro: Nenhum histórico de treinamento disponível. Execute o método train() primeiro.")
            return
        
        if plot_path is None:
            return
        
        # import os
        # os.makedirs('plot', exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Gráfico de Loss
        ax1.plot(self.history['loss'], label='Treino', linewidth=2)
        ax1.plot(self.history['val_loss'], label='Validação', linewidth=2)
        ax1.set_title('Perda durante o Treinamento', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Época', fontsize=12)
        ax1.set_ylabel('Perda', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de Accuracy
        # ax2.plot(self.history['accuracy'], label='Treino', linewidth=2)
        # # ax2.plot(self.history['val_accuracy'], label='Validação', linewidth=2)
        # ax2.set_title('Acurácia durante o Treinamento', fontsize=14, fontweight='bold')
        # ax2.set_xlabel('Época', fontsize=12)
        # ax2.set_ylabel('Acurácia', fontsize=12)
        # ax2.legend(fontsize=10)
        # ax2.grid(True, alpha=0.3)

        ax2.plot(self.history['auc'], label='Treino', linewidth=2)
        ax2.plot(self.history['val_auc'], label='Validação', linewidth=2)
        ax2.set_title('AUC durante o Treinamento', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Época', fontsize=12)
        ax2.set_ylabel('AUC', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # ax4.plot(self.history['recall'], label='Treino', linewidth=2)
        # # ax4.plot(self.history['val_recall'], label='Validação', linewidth=2)
        # ax4.set_title('Recall durante o Treinamento', fontsize=14, fontweight='bold')
        # ax4.set_xlabel('Época', fontsize=12)
        # ax4.set_ylabel('Recall', fontsize=12)
        # ax4.legend(fontsize=10)
        # ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nGráfico salvo como: {plot_path}")

   
    # @tf.function(reduce_retracing=True)
    # def _predict_step(self, inputs):
    #     return self.model(inputs, training=False)

    # def predict(self, dataset):
    #     # --- Métricas detalhadas ---
    #     raw_predictions = self._predict_step(dataset.features_test)
    #     target_predicted = (raw_predictions > 0.5).numpy().astype(int)

    #     res = {
    #         "accuracy": accuracy_score(dataset.target_test, target_predicted),
    #         "precision": precision_score(dataset.target_test, target_predicted, zero_division=0),
    #         "recall": recall_score(dataset.target_test, target_predicted, zero_division=0),
    #         "f1_score": f1_score(dataset.target_test, target_predicted, zero_division=0),
    #         "auc_roc": tf.keras.metrics.AUC()(dataset.target_test, raw_predictions).numpy()
    #     }
    #     # print("\nMétricas detalhadas no conjunto de validação:")
    #     # for key, value in res.items():
    #     #     print(f"{key} : {value:.2%}")

    #     # Guarantee all metric values are native Python floats so the result is JSON-serializable
    #     for k, v in list(res.items()):
    #         try:
    #             res[k] = float(v)
    #         except Exception:
    #             # leave value as-is if it cannot be converted
    #             pass

    #     return res