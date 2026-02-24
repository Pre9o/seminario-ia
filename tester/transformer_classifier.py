import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

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


class TransformerClassifier:
    def __init__(
        self,
        shape,
        pretrained_encoder=None,
        encoder_path=None,
        hidden_units=16,
        dropout=0.2,
        learning_rate=1e-3,
        weight_decay=0.0,
        loss='binary_crossentropy',
        threshold=0.5,
    ):
        self.shape = shape
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss = loss
        self.threshold = threshold

        if pretrained_encoder is None and encoder_path is not None:
            pretrained_encoder = keras.models.load_model(encoder_path)

        self.encoder = pretrained_encoder
        if self.encoder is None:
            raise ValueError('pretrained_encoder or encoder_path is required')

        self.model = self._build_model()
        self.history = None
        # self.compile()

    def _build_model(self):
        inputs = keras.Input(shape=(self.shape,), name='input')
        x = self.encoder(inputs)
        if len(x.shape) == 3:
            x = keras.layers.GlobalAveragePooling1D()(x)
        elif len(x.shape) == 2:
            x = x
        else:
            x = keras.layers.Flatten()(x)

        x = keras.layers.Dense(int(self.hidden_units), activation='relu', name=f'head_dense_1')(x)
        x = keras.layers.Dropout(self.dropout, name=f'head_dropout_1')(x)

        outputs = keras.layers.Dense(1, activation='sigmoid', name='output')(x)
        return keras.Model(inputs=inputs, outputs=outputs, name='transformer_classifier')

    def freeze_encoder(self):
        self.encoder.trainable = False
        if hasattr(self.encoder, 'layers'):
            for layer in self.encoder.layers:
                layer.trainable = False

    def unfreeze_encoder(self):
        self.encoder.trainable = True
        if hasattr(self.encoder, 'layers'):
            for layer in self.encoder.layers:
                layer.trainable = True

    def compile(self, learning_rate=None, weight_decay=None):
        self.model.compile(
            loss=self.loss,
            optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Recall(name='recall', thresholds=self.threshold),
                keras.metrics.Precision(name='precision', thresholds=self.threshold),
                keras.metrics.F1Score(name='f1-score', threshold=self.threshold, dtype=tf.float32),
            ],
        )

    def train(self, dataset, epochs=200, batch_size=32, verbose=1, plot_path=None, class_weight=None):
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_auc',
            patience=50,
            mode='max',
            restore_best_weights=False,
            verbose=1,
        )

        macro_f1_callback = MacroF1Callback(
            validation_data=(dataset.features_validation, dataset.target_validation),
        )

        restore_f1_callback = RestoreBestF1()

        y_train = dataset.target_train.astype('float32')
        y_val = dataset.target_validation.astype('float32')

        history = self.model.fit(
            dataset.features_train,
            y_train,
            epochs=epochs,
            validation_data=(dataset.features_validation, y_val),
            callbacks=[macro_f1_callback, early_stopping, restore_f1_callback],
            batch_size=batch_size,
            verbose=verbose,
            class_weight=class_weight,
        )

        self.history = history.history
        self.history['val_f1_macro'] = macro_f1_callback.val_f1_macro

        if plot_path is not None:
            self.plot_training_curves(plot_path)

        return self.history

    def plot_training_curves(self, plot_path='training_curves.png'):
        if not self.history:
            return

        h = self.history
        epochs = range(1, len(h.get('loss', [])) + 1)
        if len(list(epochs)) == 0:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(list(epochs), h['loss'], label='Treino', linewidth=2)
        if 'val_loss' in h:
            ax1.plot(list(epochs), h['val_loss'], label='Validação', linewidth=2)
        ax1.set_title('Perda durante o Treinamento')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Perda')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        if 'auc' in h:
            ax2.plot(list(epochs), h['auc'], label='Treino', linewidth=2)
        if 'val_auc' in h:
            ax2.plot(list(epochs), h['val_auc'], label='Validação', linewidth=2)
        ax2.set_title('AUC durante o Treinamento')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('AUC')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

