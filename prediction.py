# prediction.py
import joblib
import pandas as pd
import numpy as np

class MLModelPredictor:
    def __init__(self, model_version="latest", model_dir="models/"):
        """Load all models and preprocessors"""
        self.model_dir = model_dir
        self.model_version = model_version
        self.load_models()
    
    def load_models(self):
        """Load all saved models and preprocessors"""
        try:
            self.churn_model = joblib.load(f'{self.model_dir}churn_model_{self.model_version}.pkl')
            self.clv_model = joblib.load(f'{self.model_dir}clv_model_{self.model_version}.pkl')
            self.clustering_model = joblib.load(f'{self.model_dir}clustering_model_{self.model_version}.pkl')
            self.pca_model = joblib.load(f'{self.model_dir}pca_model_{self.model_version}.pkl')
            self.imputer = joblib.load(f'{self.model_dir}imputer_{self.model_version}.pkl')
            self.scaler = joblib.load(f'{self.model_dir}scaler_{self.model_version}.pkl')
            print(f"✅ Models loaded successfully (version: {self.model_version})")
        except FileNotFoundError as e:
            print(f"❌ Error loading models: {e}")
            print("Make sure model files exist and version is correct")
    
    
    def preprocess_input(self, customer_data):
        """Preprocess single customer input using the same pipeline as training"""
        # Convert to DataFrame if not already
        if not isinstance(customer_data, pd.DataFrame):
            customer_data = pd.DataFrame([customer_data])
        
        # Ensure the columns are in the same order as training
        # This is the order from the training pipeline
        feature_order = ['recency', 'frequency', 'monetary', 'avg_order_value', 
                        'pct_card', 'avg_installments', 'avg_review', 'n_categories']
        
        # Reorder columns to match training data
        customer_data = customer_data[feature_order]
        
        # Apply the same imputer and scaler used during training
        customer_data_imputed = self.imputer.transform(customer_data)
        customer_data_scaled = self.scaler.transform(customer_data_imputed)
        
        return customer_data_scaled
    
    def predict_churn(self, customer_data):
        """Predict churn probability using hybrid ML + business logic model"""
        # Preprocess the input data
        processed_data = self.preprocess_input(customer_data)
        
        # Get recency value for business logic
        recency = customer_data.get('recency', 0)
        
        # Use ML model for non-recency features (exclude recency at index 0)
        non_recency_features = [1, 2, 3, 4, 5, 6, 7]  # Exclude recency (index 0)
        non_recency_data = processed_data[:, non_recency_features]
        ml_prob = self.churn_model.predict_proba(non_recency_data)[0][1]
        
        # Apply business logic for recency
        if recency <= 30:
            recency_multiplier = 0.1  # Very low churn
        elif recency <= 60:
            recency_multiplier = 0.3  # Low churn
        elif recency <= 90:
            recency_multiplier = 0.6  # Medium churn
        elif recency <= 120:
            recency_multiplier = 0.8  # High churn
        else:
            recency_multiplier = 0.95  # Very high churn
        
        # Combine ML prediction with recency logic
        churn_prob = ml_prob * recency_multiplier
        churn_prob = min(0.99, max(0.01, churn_prob))  # Ensure reasonable bounds
        
        return {
            'churn_probability': churn_prob,
            'will_churn': bool(churn_prob > 0.5),
            'risk_level': 'High' if churn_prob > 0.7 else 'Medium' if churn_prob > 0.3 else 'Low'
        }
    
    def predict_clv(self, customer_data):
        """Predict Customer Lifetime Value using trained ML model"""
        # Preprocess the input data
        processed_data = self.preprocess_input(customer_data)
        
        # Use the trained CLV model to predict value
        clv_prediction = self.clv_model.predict(processed_data)[0]
        
        # Ensure reasonable bounds
        clv_prediction = max(50, min(5000, clv_prediction))
        
        return {
            'predicted_clv': clv_prediction,
            'clv_category': 'High Value' if clv_prediction > 500 else 'Medium Value' if clv_prediction > 200 else 'Low Value'
        }
    
    def predict_segment(self, customer_data):
        """Predict customer segment using trained ML model"""
        processed_data = self.preprocess_input(customer_data)
        
        # Use the trained clustering model with PCA transformation
        pca_transformed = self.pca_model.transform(processed_data)
        segment = self.clustering_model.predict(pca_transformed)[0]
        
        # Map segment IDs to meaningful names
        segment_names = {0: 'Champions', 1: 'Loyal Customers', 2: 'Potential Loyalists', 3: 'At Risk'}
        segment_name = segment_names.get(segment, 'Unknown')
        
        return {
            'segment_id': segment,
            'segment_name': segment_name
        }
    
    def predict_all(self, customer_data):
        """Make all predictions for a customer"""
        return {
            'churn': self.predict_churn(customer_data),
            'clv': self.predict_clv(customer_data),
            'segment': self.predict_segment(customer_data)
        }