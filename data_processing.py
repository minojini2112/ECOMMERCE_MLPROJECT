# data_processing.py
import pandas as pd
import kagglehub
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class EcommerceDataProcessor:
    def __init__(self):
        self.base_path = "/kaggle/input/brazilian-ecommerce/"
        self.imputer = None
        self.scaler = None
        
    def download_data(self):
        """Download the dataset from Kaggle"""
        path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
        print("Path to dataset files:", path)
        self.base_path = path + "/"
        return path
    
    def load_data(self):
        """Load all datasets"""
        print("Loading datasets...")
        self.customers = pd.read_csv(self.base_path + "olist_customers_dataset.csv")
        self.orders = pd.read_csv(self.base_path + "olist_orders_dataset.csv")
        self.order_items = pd.read_csv(self.base_path + "olist_order_items_dataset.csv")
        self.payments = pd.read_csv(self.base_path + "olist_order_payments_dataset.csv")
        self.reviews = pd.read_csv(self.base_path + "olist_order_reviews_dataset.csv")
        self.products = pd.read_csv(self.base_path + "olist_products_dataset.csv")
        self.product_trans = pd.read_csv(self.base_path + "product_category_name_translation.csv")
        self.geolocation = pd.read_csv(self.base_path + "olist_geolocation_dataset.csv")
        self.sellers = pd.read_csv(self.base_path + "olist_sellers_dataset.csv")
        print("✅ Data loaded successfully!")
    
    def process_dates(self):
        """Convert date columns to datetime"""
        print("Processing date columns...")
        for df, col in [
            (self.orders, "order_purchase_timestamp"),
            (self.orders, "order_approved_at"),
            (self.orders, "order_delivered_customer_date"),
            (self.reviews, "review_creation_date"),
        ]:
            df[col] = pd.to_datetime(df[col])
    
    def merge_data(self):
        """Merge core tables into order-level DataFrame"""
        print("Merging datasets...")
        
        # Merge products + translation to get English category
        prod_en = self.products.merge(
            self.product_trans,
            on="product_category_name",
            how="left"
        )[["product_id", "product_category_name_english"]]
        
        # Build the full order-item level DataFrame
        self.orders_items = (
            self.orders
            .merge(self.customers, on="customer_id", how="left")
            .merge(self.order_items, on="order_id", how="left")
            .merge(self.payments, on="order_id", how="left")
            .merge(self.reviews[["order_id", "review_score"]], on="order_id", how="left")
            .merge(prod_en, on="product_id", how="left")
        )
        
        print(f"✅ Merged data shape: {self.orders_items.shape}")
        return self.orders_items
    
    def create_customer_features(self):
        """Aggregate to customer-level features"""
        print("Creating customer features...")
        
        # Define cutoff date for "today"
        cutoff = self.orders["order_purchase_timestamp"].max()
        
        # Group and compute features
        self.cust_feats = self.orders_items.groupby("customer_id").agg(
            recency=("order_purchase_timestamp", lambda x: (cutoff - x.max()).days),
            frequency=("order_id", "nunique"),
            monetary=("payment_value", "sum"),
            avg_order_value=("payment_value", "mean"),
            pct_card=("payment_type", lambda x: (x=="credit_card").mean()),
            avg_installments=("payment_installments", "mean"),
            avg_review=("review_score", "mean"),
            n_categories=("product_category_name_english", "nunique"),
        ).reset_index()
        
        # Label churn & set CLV target
        self.cust_feats["churned"] = (self.cust_feats["recency"] > 90).astype(int)
        self.cust_feats["CLV"] = self.cust_feats["monetary"]
        
        print(f"✅ Customer features created: {self.cust_feats.shape}")
        return self.cust_feats
    
    def split_and_scale_data(self):
        """Split data and scale features"""
        print("Splitting and scaling data...")
        
        # Define feature matrix X and targets
        X = self.cust_feats.drop(["customer_id", "churned", "CLV"], axis=1)
        y_churn = self.cust_feats["churned"]
        y_clv = self.cust_feats["CLV"]
        
        # Train/test split
        X_train, X_test, yc_train, yc_test, yclv_train, yclv_test = train_test_split(
            X, y_churn, y_clv,
            test_size=0.2,
            random_state=42
        )
        
        # Impute missing values
        self.imputer = SimpleImputer(strategy="median")
        X_train_imp = self.imputer.fit_transform(X_train)
        X_test_imp = self.imputer.transform(X_test)
        
        # Scale data
        self.scaler = StandardScaler().fit(X_train_imp)
        X_train_s = self.scaler.transform(X_train_imp)
        X_test_s = self.scaler.transform(X_test_imp)
        
        print("✅ Data split and scaled successfully!")
        
        return {
            'X_train_s': X_train_s,
            'X_test_s': X_test_s,
            'yc_train': yc_train,
            'yc_test': yc_test,
            'yclv_train': yclv_train,
            'yclv_test': yclv_test,
            'feature_names': X.columns.tolist()
        }
    
    def process_all(self):
        """Run the complete data processing pipeline"""
        self.download_data()
        self.load_data()
        self.process_dates()
        self.merge_data()
        self.create_customer_features()
        return self.split_and_scale_data()