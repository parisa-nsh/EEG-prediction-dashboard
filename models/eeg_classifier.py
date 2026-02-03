import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class EEGClassifier:
    def __init__(self):
        self.models = {
            'cnn': None,
            'deep_nn': None,
            'shallow_nn': None,
            'random_forest': None,
            'svm': None
        }
        self.scalers = {}
        self.current_model = 'cnn'
        self.emotion_classes = ['happy', 'sad', 'anxious', 'neutral']
        
        # Default parameters
        self.parameters = {
            'cnn': {
                'filters': [32, 64, 128],
                'kernel_size': 3,
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 50
            },
            'deep_nn': {
                'layers': [128, 64, 32],
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            },
            'shallow_nn': {
                'hidden_size': 64,
                'dropout_rate': 0.1,
                'learning_rate': 0.01,
                'batch_size': 32,
                'epochs': 50
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            },
            'svm': {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale'
            }
        }
        
        self.model_info = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all models with default parameters"""
        for model_type in self.models.keys():
            self._build_model(model_type)
    
    def _build_model(self, model_type):
        """Build a specific model type"""
        if model_type == 'cnn':
            self._build_cnn()
        elif model_type == 'deep_nn':
            self._build_deep_nn()
        elif model_type == 'shallow_nn':
            self._build_shallow_nn()
        elif model_type == 'random_forest':
            self._build_random_forest()
        elif model_type == 'svm':
            self._build_svm()
    
    def _build_cnn(self):
        """Build CNN model for EEG classification"""
        params = self.parameters['cnn']
        
        model = keras.Sequential([
            layers.Input(shape=(64, 64, 1)),  # Assuming 64x64 input
            
            # Convolutional layers
            layers.Conv2D(params['filters'][0], params['kernel_size'], activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(params['dropout_rate']),
            
            layers.Conv2D(params['filters'][1], params['kernel_size'], activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(params['dropout_rate']),
            
            layers.Conv2D(params['filters'][2], params['kernel_size'], activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(params['dropout_rate']),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(params['dropout_rate']),
            layers.Dense(64, activation='relu'),
            layers.Dropout(params['dropout_rate']),
            layers.Dense(len(self.emotion_classes), activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['cnn'] = model
        self.model_info['cnn'] = {
            'type': 'CNN',
            'parameters': params,
            'architecture': 'Conv2D -> BatchNorm -> MaxPool -> Dropout -> Dense'
        }
    
    def _build_deep_nn(self):
        """Build deep neural network"""
        params = self.parameters['deep_nn']
        
        model = keras.Sequential([
            layers.Input(shape=(4096,)),  # Flattened EEG features
            
            # Hidden layers
            layers.Dense(params['layers'][0], activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(params['dropout_rate']),
            
            layers.Dense(params['layers'][1], activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(params['dropout_rate']),
            
            layers.Dense(params['layers'][2], activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(params['dropout_rate']),
            
            layers.Dense(len(self.emotion_classes), activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['deep_nn'] = model
        self.model_info['deep_nn'] = {
            'type': 'Deep Neural Network',
            'parameters': params,
            'architecture': f'Dense({params["layers"][0]}) -> Dense({params["layers"][1]}) -> Dense({params["layers"][2]})'
        }
    
    def _build_shallow_nn(self):
        """Build shallow neural network"""
        params = self.parameters['shallow_nn']
        
        model = keras.Sequential([
            layers.Input(shape=(4096,)),
            layers.Dense(params['hidden_size'], activation='relu'),
            layers.Dropout(params['dropout_rate']),
            layers.Dense(len(self.emotion_classes), activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['shallow_nn'] = model
        self.model_info['shallow_nn'] = {
            'type': 'Shallow Neural Network',
            'parameters': params,
            'architecture': f'Dense({params["hidden_size"]}) -> Dense({len(self.emotion_classes)})'
        }
    
    def _build_random_forest(self):
        """Build Random Forest classifier"""
        params = self.parameters['random_forest']
        
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            random_state=42
        )
        
        self.models['random_forest'] = model
        self.scalers['random_forest'] = StandardScaler()
        self.model_info['random_forest'] = {
            'type': 'Random Forest',
            'parameters': params,
            'architecture': 'Ensemble of Decision Trees'
        }
    
    def _build_svm(self):
        """Build SVM classifier"""
        params = self.parameters['svm']
        
        model = SVC(
            C=params['C'],
            kernel=params['kernel'],
            gamma=params['gamma'],
            probability=True,
            random_state=42
        )
        
        self.models['svm'] = model
        self.scalers['svm'] = StandardScaler()
        self.model_info['svm'] = {
            'type': 'Support Vector Machine',
            'parameters': params,
            'architecture': f'SVM with {params["kernel"]} kernel'
        }
    
    def update_parameters(self, new_params):
        """Update model parameters and rebuild models if necessary"""
        for model_type, params in new_params.items():
            if model_type in self.parameters:
                self.parameters[model_type].update(params)
                self._build_model(model_type)
    
    def set_current_model(self, model_type):
        """Set the current model to use for prediction"""
        if model_type in self.models:
            self.current_model = model_type
    
    def predict(self, data):
        """Predict emotions from EEG data"""
        model = self.models[self.current_model]
        
        if self.current_model in ['cnn', 'deep_nn', 'shallow_nn']:
            # For neural networks
            if self.current_model == 'cnn':
                # Reshape for CNN (assuming 64x64 input)
                if len(data.shape) == 2:
                    data = data.reshape(-1, 64, 64, 1)
                elif len(data.shape) == 3:
                    data = data.reshape(-1, 64, 64, 1)
            else:
                # Flatten for dense networks
                if len(data.shape) > 2:
                    data = data.reshape(data.shape[0], -1)
            
            probabilities = model.predict(data)
            predictions = np.argmax(probabilities, axis=1)
            
        else:
            # For traditional ML models
            scaler = self.scalers[self.current_model]
            data_scaled = scaler.fit_transform(data.reshape(data.shape[0], -1))
            
            probabilities = model.predict_proba(data_scaled)
            predictions = model.predict(data_scaled)
        
        return predictions, probabilities
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the current model"""
        model = self.models[self.current_model]
        params = self.parameters[self.current_model]
        
        if self.current_model in ['cnn', 'deep_nn', 'shallow_nn']:
            # For neural networks
            if self.current_model == 'cnn':
                X_train = X_train.reshape(-1, 64, 64, 1)
                if X_val is not None:
                    X_val = X_val.reshape(-1, 64, 64, 1)
            else:
                X_train = X_train.reshape(X_train.shape[0], -1)
                if X_val is not None:
                    X_val = X_val.reshape(X_val.shape[0], -1)
            
            # Convert labels to categorical
            y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=len(self.emotion_classes))
            if y_val is not None:
                y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=len(self.emotion_classes))
            
            history = model.fit(
                X_train, y_train_cat,
                validation_data=(X_val, y_val_cat) if X_val is not None else None,
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                verbose=1
            )
            
            return history
            
        else:
            # For traditional ML models
            scaler = self.scalers[self.current_model]
            X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
            
            model.fit(X_train_scaled, y_train)
            return None
    
    def get_available_models(self):
        """Get list of available model types"""
        return list(self.models.keys())
    
    def get_default_parameters(self):
        """Get default parameters for all models"""
        return self.parameters
    
    def get_current_parameters(self):
        """Get current parameters for the active model"""
        return {
            'current_model': self.current_model,
            'parameters': self.parameters[self.current_model]
        }
    
    def get_model_info(self):
        """Get information about the current model"""
        return self.model_info.get(self.current_model, {})
    
    def save_model(self, filepath):
        """Save the current model"""
        model = self.models[self.current_model]
        
        if self.current_model in ['cnn', 'deep_nn', 'shallow_nn']:
            model.save(filepath)
        else:
            joblib.dump(model, filepath)
            scaler_path = filepath.replace('.pkl', '_scaler.pkl')
            joblib.dump(self.scalers[self.current_model], scaler_path)
    
    def load_model(self, filepath, model_type):
        """Load a saved model"""
        if model_type in ['cnn', 'deep_nn', 'shallow_nn']:
            self.models[model_type] = keras.models.load_model(filepath)
        else:
            self.models[model_type] = joblib.load(filepath)
            scaler_path = filepath.replace('.pkl', '_scaler.pkl')
            if os.path.exists(scaler_path):
                self.scalers[model_type] = joblib.load(scaler_path) 