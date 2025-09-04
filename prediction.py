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
        """Predict churn probability with realistic business logic"""
        # Get original values for business logic
        recency = customer_data.get('recency', 0)
        frequency = customer_data.get('frequency', 0)
        monetary = customer_data.get('monetary', 0)
        avg_order_value = customer_data.get('avg_order_value', 0)
        pct_card = customer_data.get('pct_card', 0)
        avg_installments = customer_data.get('avg_installments', 0)
        avg_review = customer_data.get('avg_review', 0)
        n_categories = customer_data.get('n_categories', 0)
        
        # Calculate churn probability based on business rules
        churn_prob = 0.0
        
        # Recency factor (most important)
        if recency == 0:
            churn_prob = 0.05  # Just made a purchase - very low risk
        elif recency <= 7:
            churn_prob = 0.10  # Recent purchase - low risk
        elif recency <= 30:
            churn_prob = 0.25  # Somewhat recent - low-medium risk
        elif recency <= 60:
            churn_prob = 0.45  # Getting stale - medium risk
        elif recency <= 90:
            churn_prob = 0.65  # Stale - high risk
        else:
            churn_prob = 0.85  # Very stale - very high risk
        
        # Frequency adjustment
        if frequency >= 10:
            churn_prob *= 0.6  # High frequency reduces risk
        elif frequency >= 5:
            churn_prob *= 0.8  # Medium frequency slightly reduces risk
        elif frequency <= 1:
            churn_prob *= 1.3  # Low frequency increases risk
        
        # Monetary value adjustment
        if monetary >= 1000:
            churn_prob *= 0.7  # High value reduces risk
        elif monetary >= 500:
            churn_prob *= 0.85  # Medium value slightly reduces risk
        elif monetary <= 50:
            churn_prob *= 1.2  # Low value increases risk
        
        # Review score adjustment
        if avg_review >= 4.5:
            churn_prob *= 0.8  # High satisfaction reduces risk
        elif avg_review <= 2.0:
            churn_prob *= 1.3  # Low satisfaction increases risk
        
        # Credit card usage adjustment
        if pct_card >= 0.8:
            churn_prob *= 0.9  # Heavy credit card users are more loyal
        elif pct_card <= 0.2:
            churn_prob *= 1.1  # Low credit card usage slightly increases risk
        
        # Categories adjustment
        if n_categories >= 5:
            churn_prob *= 0.9  # Diverse purchases reduce risk
        elif n_categories <= 1:
            churn_prob *= 1.1  # Limited categories increase risk
        
        # Ensure reasonable bounds
        churn_prob = max(0.01, min(0.99, churn_prob))
        
        return {
            'churn_probability': churn_prob,
            'will_churn': bool(churn_prob > 0.5),
            'risk_level': 'High' if churn_prob > 0.7 else 'Medium' if churn_prob > 0.3 else 'Low'
        }
    
    def predict_clv(self, customer_data):
        """Predict Customer Lifetime Value with realistic business logic"""
        # Get original values for business logic
        recency = customer_data.get('recency', 0)
        frequency = customer_data.get('frequency', 0)
        monetary = customer_data.get('monetary', 0)
        avg_order_value = customer_data.get('avg_order_value', 0)
        avg_review = customer_data.get('avg_review', 0)
        
        # Calculate CLV based on business rules
        # Base CLV starts with current monetary value
        base_clv = monetary
        
        # Adjust based on frequency (more orders = higher potential)
        if frequency >= 10:
            clv_multiplier = 2.5  # High frequency customers
        elif frequency >= 5:
            clv_multiplier = 2.0  # Medium frequency customers
        elif frequency >= 3:
            clv_multiplier = 1.5  # Low-medium frequency customers
        else:
            clv_multiplier = 1.0  # Low frequency customers
        
        # Adjust based on recency
        if recency == 0:
            recency_multiplier = 1.3  # Just made a purchase - high potential
        elif recency <= 30:
            recency_multiplier = 1.1  # Recent activity - good potential
        elif recency <= 60:
            recency_multiplier = 0.9  # Somewhat stale - reduced potential
        elif recency <= 90:
            recency_multiplier = 0.7  # Stale - low potential
        else:
            recency_multiplier = 0.5  # Very stale - very low potential
        
        # Adjust based on review score
        if avg_review >= 4.5:
            review_multiplier = 1.2  # High satisfaction
        elif avg_review >= 3.5:
            review_multiplier = 1.0  # Average satisfaction
        else:
            review_multiplier = 0.8  # Low satisfaction
        
        # Calculate final CLV
        clv_prediction = base_clv * clv_multiplier * recency_multiplier * review_multiplier
        
        # Ensure reasonable bounds
        clv_prediction = max(50, min(5000, clv_prediction))
        
        return {
            'predicted_clv': clv_prediction,
            'clv_category': 'High Value' if clv_prediction > 500 else 'Medium Value' if clv_prediction > 200 else 'Low Value'
        }
    
    def predict_segment(self, customer_data):
        """Predict customer segment with business logic"""
        processed_data = self.preprocess_input(customer_data)
        
        # Get original values for business logic
        recency = customer_data.get('recency', 0)
        frequency = customer_data.get('frequency', 0)
        monetary = customer_data.get('monetary', 0)
        
        # Business logic for segmentation
        if recency == 0 and frequency >= 5 and monetary >= 500:
            segment_name = 'Champions'
            segment_id = 0
        elif recency <= 30 and frequency >= 3 and monetary >= 200:
            segment_name = 'Loyal Customers'
            segment_id = 1
        elif recency <= 60 and frequency >= 2:
            segment_name = 'Potential Loyalists'
            segment_id = 2
        elif recency > 90 or frequency <= 1:
            segment_name = 'At Risk'
            segment_id = 3
        else:
            # Fallback to model prediction
            try:
                pca_transformed = self.pca_model.transform(processed_data)
                segment = self.clustering_model.predict(pca_transformed)[0]
                segment_names = {0: 'Champions', 1: 'Loyal Customers', 2: 'Potential Loyalists', 3: 'At Risk'}
                segment_name = segment_names.get(segment, 'Unknown')
                segment_id = segment
            except:
                segment_name = 'Potential Loyalists'
                segment_id = 2
        
        return {
            'segment_id': segment_id,
            'segment_name': segment_name
        }
    
    def predict_all(self, customer_data):
        """Make all predictions for a customer"""
        return {
            'churn': self.predict_churn(customer_data),
            'clv': self.predict_clv(customer_data),
            'segment': self.predict_segment(customer_data)
        }