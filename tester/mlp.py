import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

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

class MLP:
    def __init__(self, shape, layers=HIDDEN_LAYERS, activation=ACTIVATION):
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.InputLayer(shape=(shape,)))
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
        self.history = None  # Armazena o histórico de treinamento

    def compile(self, loss=LOSS, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
        self.model.compile(
            loss=loss,      # Ideal para classificação binária
            optimizer=keras.optimizers.AdamW(
                learning_rate=learning_rate, # You can tune the learning rate
                weight_decay=weight_decay   # A common starting point for weight decay
            ), # Otimizador popular e eficiente
            metrics=[
                keras.metrics.Accuracy(name='accuracy'),
                keras.metrics.AUC(name='auc'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.F1Score(name='f1_score')]             # Métrica para acompanhar o desempenho
        )

    def summary(self):
        self.model.summary()

    def train(
        self,
        dataset,
        epochs=5000,
        batch_size=32,
        verbose=0,
        name=None
    ):
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )


        history = self.model.fit(
            dataset.features_train,
            dataset.target_train,
            epochs=epochs, # Número de vezes que o modelo verá todo o dataset
            validation_data=(dataset.features_validation, dataset.target_validation),
            callbacks=[early_stopping],
            batch_size=batch_size,
            verbose=verbose # Usamos verbose=0 para não poluir a saída, mas você pode usar 1 para ver o progresso
        )

        self.history = history.history  # Armazena o histórico
        self.plot_training_curves(name)
        actual_epochs = len(history.history['loss'])
        print(f"\nTreinamento interrompido na época: {actual_epochs}")

        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        train_precision = history.history['precision'][-1]
        val_precision = history.history['val_precision'][-1]
        train_recall = history.history['recall'][-1]
        val_recall = history.history['val_recall'][-1]
        train_f1_score = history.history['f1_score'][-1]
        val_f1_score = history.history['val_f1_score'][-1]
        train_auc = history.history['auc'][-1]
        val_auc = history.history['val_auc'][-1]
        

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

    def plot_training_curves(self, filename='training_curves.svg', figsize=(12, 5)):
        """
        Plota e salva as curvas de treinamento (loss e accuracy).
        
        Args:
            filename (str): Nome do arquivo para salvar o gráfico
            figsize (tuple): Tamanho da figura (largura, altura)
        """
        if self.history is None:
            print("Erro: Nenhum histórico de treinamento disponível. Execute o método train() primeiro.")
            return
        
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
        plt.savefig('plot/' + filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nGráfico salvo como: {filename}")

   
    @tf.function(reduce_retracing=True)
    def _predict_step(self, inputs):
        return self.model(inputs, training=False)

    def predict(self, dataset):
        # --- Métricas detalhadas ---
        raw_predictions = self._predict_step(dataset.features_test)
        target_predicted = (raw_predictions > 0.5).numpy().astype(int)

        res = {
            "accuracy": accuracy_score(dataset.target_test, target_predicted),
            "precision": precision_score(dataset.target_test, target_predicted, zero_division=0),
            "recall": recall_score(dataset.target_test, target_predicted, zero_division=0),
            "f1_score": f1_score(dataset.target_test, target_predicted, zero_division=0),
            "auc_roc": tf.keras.metrics.AUC()(dataset.target_test, raw_predictions).numpy()
        }
        # print("\nMétricas detalhadas no conjunto de validação:")
        # for key, value in res.items():
        #     print(f"{key} : {value:.2%}")

        # Guarantee all metric values are native Python floats so the result is JSON-serializable
        for k, v in list(res.items()):
            try:
                res[k] = float(v)
            except Exception:
                # leave value as-is if it cannot be converted
                pass

        return res