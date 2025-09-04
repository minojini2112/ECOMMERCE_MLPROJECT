# main.py
"""
Main training pipeline for the e-commerce ML project
Run this file to train all models and save them
"""

from data_processing import EcommerceDataProcessor
from model_training import EcommerceMLTrainer

def main():
    print("🚀 Starting E-commerce ML Training Pipeline")
    print("=" * 50)
    
    # Step 1: Data Processing
    print("\n📊 Step 1: Data Processing")
    processor = EcommerceDataProcessor()
    data_dict = processor.process_all()
    
    # Step 2: Model Training
    print("\n🤖 Step 2: Model Training")
    trainer = EcommerceMLTrainer(data_dict, processor.cust_feats.drop(["customer_id", "churned", "CLV"], axis=1).columns.tolist())
    
    # Also save the preprocessors
    import joblib
    import os
    os.makedirs("models/", exist_ok=True)
    
    model_version = trainer.model_version
    joblib.dump(processor.scaler, f'models/scaler_{model_version}.pkl')
    joblib.dump(processor.imputer, f'models/imputer_{model_version}.pkl')
    
    # Train all models
    model_version, best_models, best_ridge = trainer.train_all_models()
    
    print(f"\n✅ Training completed successfully!")
    print(f"📁 Models saved with version: {model_version}")
    print(f"🎯 You can now run: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()