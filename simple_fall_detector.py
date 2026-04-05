import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from scipy import stats
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

class SimpleFallDetector:
    def __init__(self, sampling_rate=50, window_size_seconds=2.5):
        """
        Simple Fall Detection System - handles any file size automatically
        
        Args:
            sampling_rate (int): Sensor sampling rate in Hz
            window_size_seconds (float): Window size in seconds
        """
        self.sampling_rate = sampling_rate
        self.window_size_seconds = window_size_seconds
        self.window_size_samples = int(sampling_rate * window_size_seconds)
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = []
        
    def load_and_train(self, data_path="./"):
        """Load Excel files and train the model"""
        print("Loading and training fall detection model...")
        
        # Load all Excel files
        excel_files = glob.glob(os.path.join(data_path, "mpu6050_data*.xlsx"))
        if not excel_files:
            print("No Excel files found!")
            return False
            
        all_data = []
        for file in excel_files:
            try:
                df = pd.read_excel(file)
                all_data.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                
        if not all_data:
            return False
            
        # Combine and preprocess data
        combined_data = pd.concat(all_data, ignore_index=True)
        processed_data = self._preprocess_training_data(combined_data)
        
        # Extract features and train
        X, y = self._extract_training_features(processed_data)
        self._train_models(X, y)
        
        print("Training completed successfully!")
        return True
    
    def predict_single_window(self, sensor_data):
        """
        Predict fall on exactly one window — no sub-windowing.
        Used for real-time streaming.
        """
        if not self.models:
            return {'prediction': 0, 'probability': 0.0,
                    'confidence': 'low', 'message': 'No model', 'windows_processed': 0}

        if len(sensor_data) < self.window_size_samples:
            return {'prediction': 0, 'probability': 0.0,
                    'confidence': 'low',
                    'message': f'Need {self.window_size_samples} samples, got {len(sensor_data)}',
                    'windows_processed': 0}

        # Use exactly the last window_size_samples rows
        window = sensor_data.iloc[-self.window_size_samples:].copy()
        window = window.reset_index(drop=True)

        # Convert ADC to physical units ONLY — skip filtfilt to avoid edge artifacts
        window = self._convert_only(window)

        try:
            features        = self._extract_window_features(window)
            features_scaled = self.scaler.transform([features])
            best_model      = self._get_best_model()
            prob            = float(best_model.predict_proba(features_scaled)[0][1])

            # Threshold 0.65 — reduces false positives on real sensor data
            prediction = 1 if prob > 0.65 else 0
            confidence = ('high'   if abs(prob - 0.5) > 0.35 else
                          'medium' if abs(prob - 0.5) > 0.15 else 'low')

            return {
                'prediction'      : prediction,
                'probability'     : round(prob, 4),
                'confidence'      : confidence,
                'windows_processed': 1,
                'message'         : 'Success',
                'label'           : 'FALL DETECTED' if prediction == 1 else 'NO FALL'
            }
        except Exception as e:
            return {'prediction': 0, 'probability': 0.0,
                    'confidence': 'low', 'message': str(e), 'windows_processed': 0}

    def _convert_only(self, data):
        """Convert ADC to physical units without filtering (avoids edge artifacts)."""
        data = data.copy()
        if data[['ax', 'ay', 'az']].abs().max().max() > 1000:
            acc_scale  = 16.0 / 32768.0 * 9.81
            gyro_scale = 2000.0 / 32768.0 * (3.14159 / 180.0)
            for col in ['ax', 'ay', 'az']:
                if col in data.columns:
                    data[col] = data[col] * acc_scale
            for col in ['gx', 'gy', 'gz']:
                if col in data.columns:
                    data[col] = data[col] * gyro_scale
        return data

    def predict_file(self, sensor_data):
        """
        Predict fall for any size sensor data file
        
        Args:
            sensor_data: DataFrame with columns [ax, ay, az, gx, gy, gz]
        
        Returns:
            Dictionary with prediction results
        """
        if not self.models:
            return {
                'error': 'Model not trained. Please train the model first.',
                'prediction': 0,
                'probability': 0.0,
                'confidence': 'low'
            }
        
        # Check minimum data requirement
        if len(sensor_data) < self.window_size_samples:
            return {
                'prediction': 0,
                'probability': 0.0,
                'confidence': 'low',
                'message': f'Need at least {self.window_size_samples} samples, got {len(sensor_data)}',
                'windows_processed': 0
            }
        
        # Preprocess data
        processed_data = self._preprocess_prediction_data(sensor_data)
        
        # Create windows and predict
        window_probabilities = []
        step_size = max(1, self.window_size_samples // 2)  # 50% overlap
        
        for i in range(0, len(processed_data) - self.window_size_samples + 1, step_size):
            window = processed_data.iloc[i:i + self.window_size_samples].copy()
            
            try:
                # Extract features
                features = self._extract_window_features(window)
                features_scaled = self.scaler.transform([features])
                
                # Get best model
                best_model = self._get_best_model()
                prob = best_model.predict_proba(features_scaled)[0][1]
                window_probabilities.append(prob)
                
            except Exception as e:
                print(f"Warning: Error processing window {i}: {e}")
                continue
        
        if not window_probabilities:
            return {
                'prediction': 0,
                'probability': 0.0,
                'confidence': 'low',
                'message': 'No windows could be processed',
                'windows_processed': 0
            }
        
        # Combine results
        overall_probability = self._combine_probabilities(window_probabilities)
        overall_prediction = 1 if overall_probability > 0.5 else 0
        confidence = self._calculate_confidence(window_probabilities)
        
        return {
            'prediction': overall_prediction,
            'probability': overall_probability,
            'confidence': confidence,
            'windows_processed': len(window_probabilities),
            'message': 'Success'
        }
    
    def _preprocess_training_data(self, data):
        """Preprocess training data"""
        # Standardize column names
        if len(data.columns) == 8:
            data.columns = ['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'target']
        elif len(data.columns) == 7:
            data.columns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'target']
        
        # Remove duplicates and NaN
        data = data.drop_duplicates().dropna()
        
        # Convert to physical units and filter
        return self._convert_and_filter(data)
    
    def _preprocess_prediction_data(self, data):
        """Preprocess prediction data"""
        data = data.copy()
        return self._convert_and_filter(data)
    
    def _convert_and_filter(self, data):
        """Convert ADC values to physical units and apply filtering"""
        # Check if data needs conversion (large values indicate raw ADC)
        if data[['ax', 'ay', 'az']].abs().max().max() > 1000:
            # Convert to physical units
            acc_scale = 16.0 / 32768.0 * 9.81  # m/s²
            gyro_scale = 2000.0 / 32768.0 * (3.14159 / 180.0)  # rad/s
            
            for col in ['ax', 'ay', 'az']:
                if col in data.columns:
                    data[col] = data[col] * acc_scale
            
            for col in ['gx', 'gy', 'gz']:
                if col in data.columns:
                    data[col] = data[col] * gyro_scale
        
        # Apply low-pass filter
        def apply_filter(signal):
            nyquist = 0.5 * self.sampling_rate
            normal_cutoff = 20 / nyquist
            b, a = butter(4, normal_cutoff, btype='low', analog=False)
            return filtfilt(b, a, signal)
        
        for col in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']:
            if col in data.columns:
                data[col] = apply_filter(data[col].values)
        
        return data
    
    def _extract_training_features(self, data):
        """Extract features for training"""
        features = []
        labels = []
        
        step_size = self.window_size_samples // 2
        
        for i in range(0, len(data) - self.window_size_samples + 1, step_size):
            window = data.iloc[i:i + self.window_size_samples]
            
            if len(window) < self.window_size_samples:
                continue
                
            window_features = self._extract_window_features(window)
            features.append(window_features)
            labels.append(window['target'].mode().iloc[0])
        
        return pd.DataFrame(features), np.array(labels)
    
    def _extract_window_features(self, window):
        """Extract comprehensive features from a window"""
        features = []
        
        # Get sensor data
        ax, ay, az = window['ax'].values, window['ay'].values, window['az'].values
        gx, gy, gz = window['gx'].values, window['gy'].values, window['gz'].values
        
        # Calculate magnitudes
        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        
        # Extract features for each signal
        for signal in [ax, ay, az, gx, gy, gz, acc_mag, gyro_mag]:
            features.extend([
                np.mean(signal),
                np.std(signal),
                np.min(signal),
                np.max(signal),
                np.max(signal) - np.min(signal),
                stats.skew(signal),
                stats.kurtosis(signal),
                np.percentile(signal, 25),
                np.percentile(signal, 75),
                np.median(signal)
            ])
        
        # Time domain features
        features.extend([
            np.max(acc_mag),
            np.argmax(acc_mag),
            np.sum(acc_mag > 2*9.81),
            np.sum(acc_mag > 3*9.81),
        ])
        
        # Differential features
        acc_diff = np.diff(acc_mag)
        features.extend([
            np.mean(np.abs(acc_diff)),
            np.std(acc_diff),
            np.max(np.abs(acc_diff)),
        ])
        
        # Frequency domain features
        acc_fft = np.fft.fft(acc_mag)
        acc_psd = np.abs(acc_fft)**2
        freqs = np.fft.fftfreq(len(acc_mag), 1/self.sampling_rate)
        
        dominant_freq_idx = np.argmax(acc_psd[1:len(acc_psd)//2]) + 1
        features.extend([
            freqs[dominant_freq_idx],
            np.sum(acc_psd[1:6]),
            np.sum(acc_psd[6:11]),
            np.sum(acc_psd[11:21]),
        ])
        
        # Correlation features
        features.extend([
            np.corrcoef(ax, ay)[0,1] if np.std(ax) > 0 and np.std(ay) > 0 else 0,
            np.corrcoef(ax, az)[0,1] if np.std(ax) > 0 and np.std(az) > 0 else 0,
            np.corrcoef(ay, az)[0,1] if np.std(ay) > 0 and np.std(az) > 0 else 0,
        ])
        
        return features
    
    def _train_models(self, X, y):
        """Train multiple models"""
        print("Training models...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
            'SVM': SVC(C=1, gamma='scale', probability=True, random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        
        # Train and evaluate models
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.models[name] = {
                'model': model,
                'accuracy': accuracy
            }
            
            print(f"{name}: {accuracy:.4f} accuracy")
        
        self.X_test = X_test_scaled
        self.y_test = y_test
    
    def _get_best_model(self):
        """Get the best performing model"""
        best_name = max(self.models.keys(), key=lambda x: self.models[x]['accuracy'])
        return self.models[best_name]['model']
    
    def _combine_probabilities(self, probabilities):
        """Combine window probabilities — simple average is most reliable"""
        if not probabilities:
            return 0.0
        return float(np.mean(probabilities))
    
    def _calculate_confidence(self, probabilities):
        """Calculate confidence level"""
        if not probabilities:
            return 'low'
        
        prob_std = np.std(probabilities)
        avg_prob = np.mean(probabilities)
        decisiveness = abs(avg_prob - 0.5) * 2
        consistency = max(0, 1 - prob_std * 2)
        confidence_score = (consistency + decisiveness) / 2
        
        if confidence_score > 0.75:
            return 'high'
        elif confidence_score > 0.5:
            return 'medium'
        else:
            return 'low'
    
    def get_model_info(self):
        """Get information about trained models"""
        if not self.models:
            return "No models trained"
        
        info = "Trained Models:\n"
        for name, model_info in self.models.items():
            info += f"- {name}: {model_info['accuracy']:.4f} accuracy\n"
        
        best_name = max(self.models.keys(), key=lambda x: self.models[x]['accuracy'])
        info += f"\nBest Model: {best_name}"
        
        return info

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = SimpleFallDetector()
    
    # Train the model
    if detector.load_and_train():
        print("\nModel trained successfully!")
        print(detector.get_model_info())
        
        # Example prediction (you would load your CSV file here)
        # sample_data = pd.read_csv('your_sensor_data.csv')
        # result = detector.predict_file(sample_data)
        # print(f"Prediction: {result}")
    else:
        print("Training failed!")