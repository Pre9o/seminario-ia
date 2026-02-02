import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def _register_serializable(package):
    def decorator(cls):
        if hasattr(keras, 'saving') and hasattr(keras.saving, 'register_keras_serializable'):
            cls = keras.saving.register_keras_serializable(package=package)(cls)
        else:
            cls = keras.utils.register_keras_serializable(package=package)(cls)
        return cls
    return decorator


@_register_serializable(package='seminario_ia')
class FeatureSlice(keras.layers.Layer):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = int(index)

    def call(self, inputs):
        return inputs[:, self.index:self.index + 1]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'index': self.index})
        return cfg


@_register_serializable(package='seminario_ia')
class TokenSlice(keras.layers.Layer):
    def __init__(self, index, keepdims=True, **kwargs):
        super().__init__(**kwargs)
        self.index = int(index)
        self.keepdims = bool(keepdims)

    def call(self, inputs):
        if self.keepdims:
            return inputs[:, self.index:self.index + 1, :]
        return inputs[:, self.index, :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'index': self.index, 'keepdims': self.keepdims})
        return cfg


@_register_serializable(package='seminario_ia')
class CategoricalMap(keras.layers.Layer):
    def __init__(self, min_value, n_classes, **kwargs):
        super().__init__(**kwargs)
        self.min_value = int(min_value)
        self.n_classes = int(n_classes)

    def call(self, inputs):
        t = tf.cast(tf.round(inputs), tf.int32)
        is_mask = t < 0
        t = t - self.min_value
        t = tf.clip_by_value(t, 0, self.n_classes - 1)
        t = t + 1
        t = tf.where(is_mask, tf.zeros_like(t), t)
        return t

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'min_value': self.min_value, 'n_classes': self.n_classes})
        return cfg


class AutoencoderTransformer:
    def __init__(
        self,
        shape,
        categorical_indices=None,
        categorical_cardinalities=None,
        mask_ratio=0.3,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        ff_dim=128,
        dropout=0.1,
        learning_rate=0.001,
        mask_value=-999.0,
    ):
        self.shape = shape
        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.mask_value = mask_value
        self.categorical_indices = categorical_indices if categorical_indices else []
        self.categorical_cardinalities = categorical_cardinalities if categorical_cardinalities else {}
        self.continuous_indices = [i for i in range(self.shape) if i not in self.categorical_indices]

        self.categorical_mins = None
        self.encoder = None
        self.model = None
        self.history = None

    def _infer_categorical_mins(self, data_array):
        mins = {}
        for idx in self.categorical_indices:
            col = data_array[:, idx]
            col = col[np.isfinite(col)]
            col = col[col >= 0]
            if col.size == 0:
                mins[idx] = 0
            else:
                mins[idx] = int(np.min(col))
        return mins

    def _build_transformer_block(self, x):
        attn_out = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            dropout=self.dropout,
        )(x, x)
        x = keras.layers.Add()([x, attn_out])
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

        ff = keras.layers.Dense(self.ff_dim, activation='gelu')(x)
        ff = keras.layers.Dropout(self.dropout)(ff)
        ff = keras.layers.Dense(self.embed_dim)(ff)
        x = keras.layers.Add()([x, ff])
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
        return x

    def _build_encoder(self):
        inputs = keras.Input(shape=(self.shape,), name='input')

        tokens = []
        for i in range(self.shape):
            xi = FeatureSlice(i, name=f'feature_{i}')(inputs)
            if i in self.categorical_indices:
                min_val = int(self.categorical_mins.get(i, 0))
                n_classes = int(self.categorical_cardinalities.get(i, 2))
                emb_in_dim = n_classes + 4

                xi_cat = CategoricalMap(min_value=min_val, n_classes=n_classes, name=f'cat_map_{i}')(xi)
                tok = keras.layers.Embedding(emb_in_dim, self.embed_dim, name=f'cat_emb_{i}')(xi_cat)
            else:
                tok = keras.layers.Dense(self.embed_dim, name=f'cont_emb_{i}')(xi)
                tok = keras.layers.Reshape((1, self.embed_dim))(tok)
            tokens.append(tok)

        x = keras.layers.Concatenate(axis=1, name='token_concat')(tokens)
        x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = keras.layers.Dropout(self.dropout)(x)

        for _ in range(self.num_layers):
            x = self._build_transformer_block(x)

        return keras.Model(inputs=inputs, outputs=x, name='encoder')

    def _build_autoencoder_model(self):
        inputs = keras.Input(shape=(self.shape,), name='input')
        encoded_tokens = self.encoder(inputs)

        outputs = []

        if self.continuous_indices:
            cont_tokens = []
            for idx in self.continuous_indices:
                ti = TokenSlice(idx, keepdims=True, name=f'cont_token_{idx}')(encoded_tokens)
                cont_tokens.append(ti)
            cont_seq = keras.layers.Concatenate(axis=1)(cont_tokens) if len(cont_tokens) > 1 else cont_tokens[0]
            cont_out = keras.layers.Dense(1, activation='linear')(cont_seq)
            cont_out = keras.layers.Reshape((len(self.continuous_indices),), name='continuous_reconstruction')(cont_out)
            outputs.append(cont_out)

        if self.categorical_indices:
            for cat_idx in self.categorical_indices:
                n_classes = int(self.categorical_cardinalities.get(cat_idx, 2))
                ti = TokenSlice(cat_idx, keepdims=False, name=f'cat_token_{cat_idx}')(encoded_tokens)
                cat_out = keras.layers.Dense(n_classes, activation='softmax', name=f'categorical_{cat_idx}_reconstruction')(ti)
                outputs.append(cat_out)

        return keras.Model(inputs=inputs, outputs=outputs, name='transformer_hybrid_reconstruction')

    def _compile_model(self):
        losses = []
        metrics_list = []

        if self.continuous_indices:
            losses.append('mse')
            metrics_list.append(['mae'])

        if self.categorical_indices:
            for _ in self.categorical_indices:
                losses.append('sparse_categorical_crossentropy')
                metrics_list.append(['accuracy'])

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=losses,
            metrics=metrics_list,
        )

    def _ensure_built(self, data_array):
        if self.categorical_mins is None:
            self.categorical_mins = self._infer_categorical_mins(data_array)
        if self.encoder is None:
            self.encoder = self._build_encoder()
        if self.model is None:
            self.model = self._build_autoencoder_model()
            self._compile_model()

    def create_masked_data(self, data):
        data_array = data.values if hasattr(data, 'values') else data
        masked_data = data_array.copy().astype(float)
        mask = np.zeros_like(masked_data, dtype=bool)

        n_samples, n_features = masked_data.shape
        n_mask = int(self.mask_ratio * n_features)

        for i in range(n_samples):
            if n_mask <= 0:
                continue
            mask_indices = np.random.choice(n_features, n_mask, replace=False)
            mask[i, mask_indices] = True
            for idx in mask_indices:
                if idx in self.categorical_indices:
                    masked_data[i, idx] = -1
                else:
                    masked_data[i, idx] = self.mask_value

        return masked_data, mask

    def prepare_targets(self, data):
        data_array = data.values if hasattr(data, 'values') else data
        targets = []

        if self.continuous_indices:
            continuous_targets = data_array[:, self.continuous_indices].astype(float)
            targets.append(continuous_targets)

        if self.categorical_indices:
            for cat_idx in self.categorical_indices:
                min_val = int(self.categorical_mins.get(cat_idx, 0))
                cat = data_array[:, cat_idx]
                cat = np.round(cat).astype(int) - min_val
                n_classes = int(self.categorical_cardinalities.get(cat_idx, 2))
                cat = np.clip(cat, 0, n_classes - 1)
                targets.append(cat)

        return targets

    def train(self, dataset, epochs=100, batch_size=32, verbose=1, plot_path=None):
        train_array = dataset.features_train.values if hasattr(dataset.features_train, 'values') else dataset.features_train
        val_array = dataset.features_validation.values if hasattr(dataset.features_validation, 'values') else dataset.features_validation

        X_pretrain = np.vstack([train_array, val_array])
        self._ensure_built(X_pretrain)

        X_masked, _ = self.create_masked_data(X_pretrain)
        y_targets = self.prepare_targets(X_pretrain)

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0,
        )

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=0,
        )

        history = self.model.fit(
            X_masked,
            y_targets,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose,
        )

        self.history = history.history

        if plot_path is not None:
            self.plot_training_curves(plot_path=plot_path)

        return history.history

    def plot_training_curves(self, plot_path='training_curves.png'):
        if self.history is None:
            return

        history = self.history
        epochs = range(1, len(history.get('loss', [])) + 1)
        if len(epochs) == 0:
            return

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(list(epochs), history['loss'], label='Train Loss')
        if 'val_loss' in history:
            plt.plot(list(epochs), history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        keys = [k for k in history.keys() if k != 'loss' and not k.startswith('val_')]
        for k in keys:
            plt.plot(list(epochs), history[k], label=k)
            vk = f'val_{k}'
            if vk in history:
                plt.plot(list(epochs), history[vk], label=vk)
        plt.title('Training and Validation Metrics')
        plt.xlabel('Epochs')
        plt.ylabel('Metric Value')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def get_encoder(self):
        return self.encoder

    def save_encoder(self, filepath='best_encoder.keras'):
        if self.encoder is None:
            raise ValueError('Encoder not built')
        self.encoder.save(filepath)

    def load_encoder(self, filepath='best_encoder.keras'):
        self.encoder = keras.models.load_model(filepath, compile=False)
        self.model = None

