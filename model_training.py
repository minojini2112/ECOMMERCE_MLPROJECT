# model_training.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, auc, precision_recall_curve, average_precision_score,
                           silhouette_score)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class EcommerceMLTrainer:
    def __init__(self, data_dict, feature_names):
        self.data = data_dict
        self.feature_names = feature_names
        self.best_models = {}
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def train_basic_models(self):
        """Train basic regression and classification models"""
        print("Training basic models...")
        
        # CLV Regression - Linear Regression
        self.reg = LinearRegression().fit(self.data['X_train_s'], self.data['yclv_train'])
        rmse = np.sqrt(((self.reg.predict(self.data['X_test_s']) - self.data['yclv_test'])**2).mean())
        print(f"Linear Regression RMSE: {rmse}")
        
        # Churn Classification - Logistic Regression
        self.clf = LogisticRegression().fit(self.data['X_train_s'], self.data['yc_train'])
        y_pred = self.clf.predict(self.data['X_test_s'])
        
        print("Classification Report:")
        print(classification_report(self.data['yc_test'], y_pred))
        print("ROC AUC Score:", roc_auc_score(self.data['yc_test'], 
                                             self.clf.predict_proba(self.data['X_test_s'])[:,1]))
    
    def perform_clustering(self):
        """Perform customer segmentation using K-means"""
        print("Performing customer segmentation...")
        
        # Prepare data for clustering (use same preprocessing as training)
        X_cluster = np.vstack([self.data['X_train_s'], self.data['X_test_s']])
        
        # PCA for visualization
        self.pca = PCA(n_components=2)
        self.pca_2d = self.pca.fit_transform(X_cluster)
        
        # K-means clustering
        self.km_final = KMeans(n_clusters=4, random_state=42).fit(self.pca_2d)
        
        # Plot clusters
        plt.figure(figsize=(8,6))
        plt.scatter(self.pca_2d[:,0], self.pca_2d[:,1], c=self.km_final.labels_, cmap="tab10", s=10)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA of Customer Features + K-Means Clusters")
        plt.show()
        
        print("✅ Clustering completed!")
    
    def hyperparameter_tuning_regression(self):
        """Hyperparameter tuning for CLV regression"""
        print("Hyperparameter tuning for CLV regression...")
        
        ridge = Ridge()
        param_grid = {
            'alpha': [0.01, 0.1, 1, 10, 100],
            'fit_intercept': [True, False]
        }
        
        gs_ridge = GridSearchCV(
            estimator=ridge,
            param_grid=param_grid,
            scoring='neg_root_mean_squared_error',
            cv=5,
            n_jobs=-1
        )
        
        gs_ridge.fit(self.data['X_train_s'], self.data['yclv_train'])
        
        print("Best α:", gs_ridge.best_params_['alpha'])
        print("Best CV RMSE:", -gs_ridge.best_score_)
        
        self.best_ridge = gs_ridge.best_estimator_
        rmse_test = np.sqrt(((self.best_ridge.predict(self.data['X_test_s']) - self.data['yclv_test']) ** 2).mean())
        print("Test RMSE:", rmse_test)
        
        return self.best_ridge
    
    def hyperparameter_tuning_classification(self):
        """Hyperparameter tuning for churn classification"""
        print("Hyperparameter tuning for churn classification...")
        
        # Define parameter grids
        lr_params = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['liblinear']
        }
        
        dt_params = {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10]
        }
        
        svc_params = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
        
        models = [
            ('LogisticRegression', LogisticRegression(), lr_params),
            ('DecisionTree', DecisionTreeClassifier(), dt_params),
            ('SVC', SVC(probability=True), svc_params)
        ]
        
        for name, est, params in models:
            gs = GridSearchCV(
                estimator=est,
                param_grid=params,
                scoring='roc_auc',
                cv=5,
                n_jobs=-1
            )
            gs.fit(self.data['X_train_s'], self.data['yc_train'])
            self.best_models[name] = gs.best_estimator_
            print(f"{name} best params:", gs.best_params_)
            print(f"{name} best CV AUC:", gs.best_score_)
        
        # Evaluate best models
        from sklearn.metrics import roc_auc_score, classification_report
        
        for name, model in self.best_models.items():
            ypred = model.predict(self.data['X_test_s'])
            yprob = model.predict_proba(self.data['X_test_s'])[:,1]
            print(f"\n{name} Test AUC:", roc_auc_score(self.data['yc_test'], yprob))
            print(classification_report(self.data['yc_test'], ypred))
    
    def evaluate_clustering(self):
        """Evaluate optimal number of clusters"""
        print("Evaluating clustering...")
        
        sil_scores = []
        ks = range(2, 11)
        for k in ks:
            km = KMeans(n_clusters=k, random_state=42).fit(self.pca_2d)
            sil = silhouette_score(self.pca_2d, km.labels_)
            sil_scores.append(sil)
        
        plt.figure(figsize=(8, 5))
        plt.plot(ks, sil_scores, marker='o')
        plt.xlabel("n_clusters")
        plt.ylabel("Silhouette Score")
        plt.title("K-Means Silhouette Analysis")
        plt.show()
        
        optimal_k = ks[np.argmax(sil_scores)]
        print(f"Optimal number of clusters: {optimal_k}")
    
    def create_evaluation_plots(self):
        """Create comprehensive evaluation plots"""
        print("Creating evaluation plots...")
        
        # Get best classifier
        best_clf = self.best_models['DecisionTree']  # or whichever performed best
        
        # ROC Curve and other plots
        y_prob = best_clf.predict_proba(self.data['X_test_s'])[:,1]
        fpr, tpr, _ = roc_curve(self.data['yc_test'], y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(15, 5))
        
        # ROC Curve
        plt.subplot(1, 3, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Churn Prediction')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(self.data['yc_test'], y_prob)
        avg_precision = average_precision_score(self.data['yc_test'], y_prob)
        
        plt.subplot(1, 3, 2)
        plt.plot(recall, precision, color='purple', lw=2,
                 label=f'Average Precision = {avg_precision:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Feature Importance
        if hasattr(best_clf, 'feature_importances_'):
            importances = best_clf.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.subplot(1, 3, 3)
            plt.bar(range(len(importances)), importances[indices])
            plt.title('Feature Importance')
            plt.xticks(range(len(importances)), [self.feature_names[i] for i in indices], rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Residual plots for CLV
        y_pred_clv = self.best_ridge.predict(self.data['X_test_s'])
        residuals = self.data['yclv_test'] - y_pred_clv
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_pred_clv, residuals, alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predicted CLV')
        plt.ylabel('Residuals')
        plt.title('Residual Plot - CLV Prediction')
        
        plt.subplot(1, 2, 2)
        plt.scatter(self.data['yclv_test'], y_pred_clv, alpha=0.6)
        plt.plot([self.data['yclv_test'].min(), self.data['yclv_test'].max()], 
                [self.data['yclv_test'].min(), self.data['yclv_test'].max()], 'red', lw=2)
        plt.xlabel('Actual CLV')
        plt.ylabel('Predicted CLV')
        plt.title('Predicted vs Actual CLV')
        
        plt.tight_layout()
        plt.show()
    
    def save_models(self, save_dir="models/"):
        """Save all trained models"""
        print(f"Saving models with version: {self.model_version}")
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save best models
        best_clf = self.best_models['DecisionTree']  # or whichever performed best
        
        joblib.dump(best_clf, f'{save_dir}churn_model_{self.model_version}.pkl')
        joblib.dump(self.best_ridge, f'{save_dir}clv_model_{self.model_version}.pkl')
        joblib.dump(self.km_final, f'{save_dir}clustering_model_{self.model_version}.pkl')
        joblib.dump(self.pca, f'{save_dir}pca_model_{self.model_version}.pkl')
        
        # Save metadata
        y_pred_clv = self.best_ridge.predict(self.data['X_test_s'])
        y_prob = best_clf.predict_proba(self.data['X_test_s'])[:,1]
        roc_auc = roc_auc_score(self.data['yc_test'], y_prob)
        
        model_metadata = {
            'churn_model': {
                'type': 'DecisionTree',
                'accuracy': roc_auc,
                'features': self.feature_names,
                'version': self.model_version
            },
            'clv_model': {
                'type': 'Ridge',
                'rmse': np.sqrt(((y_pred_clv - self.data['yclv_test']) ** 2).mean()),
                'features': self.feature_names,
                'version': self.model_version
            }
        }
        
        with open(f'{save_dir}model_metadata_{self.model_version}.json', 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        print(f"✅ Models saved successfully!")
        return self.model_version
    
    def train_all_models(self):
        """Run the complete training pipeline"""
        self.train_basic_models()
        self.perform_clustering()
        self.hyperparameter_tuning_regression()
        self.hyperparameter_tuning_classification()
        self.evaluate_clustering()
        self.create_evaluation_plots()
        model_version = self.save_models()
        
        return model_version, self.best_models, self.best_ridge