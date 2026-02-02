import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Autoencoder:
    def __init__(
        self, 
        shape, 
        categorical_indices=None,
        categorical_cardinalities=None,
        mask_ratio=0.3, 
        hidden_units=[128, 128, 128, 128, 128], 
        activation='relu', 
        learning_rate=0.001, 
        mask_value=-999.0
    ):
       
        self.shape = shape
        self.mask_ratio = mask_ratio
        self.hidden_units = hidden_units
        self.activation = activation
        self.learning_rate = learning_rate
        self.mask_value = mask_value
        
        self.categorical_indices = categorical_indices if categorical_indices else []
        self.categorical_cardinalities = categorical_cardinalities if categorical_cardinalities else {}
        self.continuous_indices = [i for i in range(shape) if i not in self.categorical_indices]
        
        
        self.encoder = self._build_encoder()
        self.model = self._build_hybrid_model()
        self._compile_model()
        self.history = None

    def _build_encoder(self):
        encoder = keras.Sequential(name='encoder')
        encoder.add(keras.layers.InputLayer(input_shape=(self.shape,)))
        
        for idx, units in enumerate(self.hidden_units):
            encoder.add(keras.layers.Dense(
                units, 
                activation=self.activation, 
                kernel_regularizer=keras.regularizers.l2(0.01),
                name=f'encoder_dense_{idx+1}'
            ))
            encoder.add(keras.layers.Dropout(0.3, name=f'encoder_dropout_{idx+1}'))
        
        return encoder
    
    def _build_hybrid_model(self):
        input_layer = keras.Input(shape=(self.shape,), name='input')
        
        encoded = self.encoder(input_layer)
        
        continuous_outputs = []
        if self.continuous_indices:
            x_cont = encoded
            for idx, units in enumerate(reversed(self.hidden_units[:-1])):
                x_cont = keras.layers.Dense(
                    units, 
                    activation=self.activation,
                    name=f'decoder_continuous_{idx+1}'
                )(x_cont)
            
            continuous_out = keras.layers.Dense(
                len(self.continuous_indices), 
                activation='linear',
                name='continuous_reconstruction'
            )(x_cont)
            continuous_outputs.append(continuous_out)
        
        categorical_outputs = []
        if self.categorical_indices:
            for cat_idx in self.categorical_indices:
                x_cat = encoded
                for idx, units in enumerate(reversed(self.hidden_units[:-1])):
                    x_cat = keras.layers.Dense(
                        units,
                        activation=self.activation,
                        name=f'decoder_cat_{cat_idx}_layer_{idx+1}'
                    )(x_cat)
                
                # softmax para categoricas
                n_classes = self.categorical_cardinalities.get(cat_idx, 2)
                cat_out = keras.layers.Dense(
                    n_classes,
                    activation='softmax',
                    name=f'categorical_{cat_idx}_reconstruction'
                )(x_cat)
                categorical_outputs.append(cat_out)
        
        all_outputs = continuous_outputs + categorical_outputs
        
        model = keras.Model(inputs=input_layer, outputs=all_outputs, name='hybrid_reconstruction')
        
        return model
    
    def _compile_model(self):
        losses = []
        loss_weights = []
        metrics_list = []
        
        if self.continuous_indices:
            losses.append('mse')
            loss_weights.append(1.0)
            metrics_list.append(['mae']) 
        
        if self.categorical_indices:
            for _ in self.categorical_indices:
                losses.append('sparse_categorical_crossentropy')
                loss_weights.append(1.0)
                metrics_list.append(['accuracy'])  
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics_list 
        )
        
    
    def create_masked_data(self, data):
        if hasattr(data, 'values'):
            data_array = data.values
        else:
            data_array = data
            
        masked_data = data_array.copy().astype(float)
        mask = np.zeros_like(masked_data, dtype=bool)
        
        n_samples, n_features = masked_data.shape
        n_mask = int(self.mask_ratio * n_features)

        for i in range(n_samples):
            mask_indices = np.random.choice(n_features, n_mask, replace=False)
            mask[i, mask_indices] = True
            
            # máscara diferente para categóricas e contínuas
            for idx in mask_indices:
                if idx in self.categorical_indices:
                    masked_data[i, idx] = -1  
                else:
                    masked_data[i, idx] = self.mask_value 

        return masked_data, mask
    
    def prepare_targets(self, data):
        if hasattr(data, 'values'):
            data_array = data.values
        else:
            data_array = data
        
        targets = []
        
        if self.continuous_indices:
            continuous_targets = data_array[:, self.continuous_indices]
            targets.append(continuous_targets)
        
        if self.categorical_indices:
            for cat_idx in self.categorical_indices:
                cat_targets = data_array[:, cat_idx].astype(int)
                
                cat_targets = cat_targets - cat_targets.min()
                
                targets.append(cat_targets)
        
        return targets
    

    def train(self, dataset, epochs=100, batch_size=32, verbose=1, plot_path=None):
        train_array = dataset.features_train.values if hasattr(dataset.features_train, 'values') else dataset.features_train
        val_array = dataset.features_validation.values if hasattr(dataset.features_validation, 'values') else dataset.features_validation
        
        X_pretrain = np.vstack([train_array, val_array])
        
        X_masked, mask = self.create_masked_data(X_pretrain)
        
        y_targets = self.prepare_targets(X_pretrain)
        
        if self.continuous_indices:
            print(f"  - Continuous: shape {y_targets[0].shape}")
        if self.categorical_indices:
            for i, cat_idx in enumerate(self.categorical_indices):
                offset = 1 if self.continuous_indices else 0
                print(f"  - Categorical {cat_idx}: shape {y_targets[offset + i].shape}, unique classes {np.unique(y_targets[offset + i])}")
        
        # callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
        
        history = self.model.fit(
            X_masked,
            y_targets,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )
        
        self.history = history.history
        
        X_val_masked, _ = self.create_masked_data(val_array)
        y_val_targets = self.prepare_targets(val_array)
        
        val_loss = self.model.evaluate(X_val_masked, y_val_targets, verbose=0)
        
        print(f"Pré-treino concluído\n")
        print(f"Épocas treinadas: {len(history.history['loss'])}")
        print(f"Loss final (validação): {val_loss[0]:.4f}")
        
        if len(val_loss) > 1:
            idx = 1
            if self.continuous_indices:
                print(f"  - continuous MSE: {val_loss[idx]:.4f}")
                idx += 1
            if self.categorical_indices:
                for cat_idx in self.categorical_indices:
                    if idx < len(val_loss):
                        print(f"  - categorical {cat_idx} CE: {val_loss[idx]:.4f}")
                        idx += 1
        
        if plot_path is not None:
            self.plot_training_curves(plot_path=plot_path)
        return history.history
    
    def plot_training_curves(self, plot_path='training_curves.png'):
        if self.history is None:
            print("Nenhum histórico de treinamento disponível para plotar.")
            return
        
        history = self.history
        epochs = range(1, len(history['loss']) + 1)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['loss'], label='Train Loss')
        plt.plot(epochs, history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        
        metric_idx = 1
        if self.continuous_indices:
            plt.plot(epochs, history['continuous_reconstruction_mae'], label='Train Continuous MAE')
            plt.plot(epochs, history['val_continuous_reconstruction_mae'], label='Val Continuous MAE')
            metric_idx += 1
        
        if self.categorical_indices:
            for cat_idx in self.categorical_indices:
                plt.plot(epochs, history[f'categorical_{cat_idx}_reconstruction_accuracy'], label=f'Train Cat {cat_idx} Acc')
                plt.plot(epochs, history[f'val_categorical_{cat_idx}_reconstruction_accuracy'], label=f'Val Cat {cat_idx} Acc')
                metric_idx += 1
        
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
    
    def save_encoder(self, filepath='pretrained_encoder.keras'):
        self.encoder.save(filepath)
        print(f"Encoder salvo em: {filepath}")
    
    def load_encoder(self, filepath='pretrained_encoder.keras'):
        self.encoder = keras.models.load_model(filepath)
        print(f"Encoder carregado de: {filepath}")