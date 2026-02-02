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
    
    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred_proba = self.model.predict(X_val, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_true = y_val.values if hasattr(y_val, 'values') else y_val
        
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        self.val_f1_macro.append(f1_macro)
        logs['val_f1_macro'] = f1_macro
        
        if epoch % 10 == 0 or epoch < 5:
            print(f" - val_f1_macro: {f1_macro:.4f}", end='')

class MLP:
    def __init__(self, shape, layers=HIDDEN_LAYERS, activation=ACTIVATION, pretrained_encoder=None):
        self.model = keras.models.Sequential()
        self.shape = shape
        
        if pretrained_encoder is not None:
            print("Usando encoder pré-treinado!")
            self.model.add(pretrained_encoder)
            
            for layer in pretrained_encoder.layers:
                layer.trainable = False
        else:
            self.model.add(keras.layers.InputLayer(input_shape=(self.shape,), name='input_layer'))
            for idx, neurons in enumerate(layers):
                self.model.add(
                    keras.layers.Dense(
                        neurons,
                        activation=activation,
                        kernel_regularizer=regularizers.l2(0.01),
                        name=f'camada_oculta_{idx+1}_{activation}'
                    )
                )
                self.model.add(keras.layers.Dropout(DROPOUTS[idx], name=f'dropout_{idx+1}'))

        self.model.add(keras.layers.Dense(1, activation='sigmoid', name='camada_saida_sigmoid'))
        self.compile()
        self.history = None
    
    def unfreeze_encoder(self, learning_rate=0.0001):
        for layer in self.model.layers:
            if layer.name == 'encoder':
                layer.trainable = True
                if hasattr(layer, "layers"):
                    for sublayer in layer.layers:
                        sublayer.trainable = True
        self.compile(learning_rate=learning_rate)
    

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
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        macro_f1_callback = MacroF1Callback(
            validation_data=(dataset.features_validation, dataset.target_validation)
        )

        y_train = dataset.target_train.astype('float32')
        y_val = dataset.target_validation.astype('float32')

        history = self.model.fit(
            dataset.features_train,
            y_train,
            epochs=epochs, 
            validation_data=(dataset.features_validation, y_val),
            callbacks=[macro_f1_callback, early_stopping],
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

    def plot_training_curves(self, plot_path='training_curves.png', figsize=(12, 5)):
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