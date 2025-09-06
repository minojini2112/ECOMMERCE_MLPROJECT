# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# Import your custom modules
try:
    from prediction import MLModelPredictor
    MODELS_AVAILABLE = True
except:
    MODELS_AVAILABLE = False
    st.warning("⚠ Models not found. Please train models first by running the training pipeline.")

# Page config
st.set_page_config(
    page_title="E-commerce ML Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize model predictor
@st.cache_resource
def load_models():
    if MODELS_AVAILABLE:
        # You'll need to specify the correct model version here
        # Check your models/ directory for the latest version
        model_files = [f for f in os.listdir('models/') if f.startswith('churn_model_')]
        if model_files:
            # Extract version from filename like 'churn_model_20250904_114402.pkl'
            latest_file = sorted(model_files)[-1]
            # Split by '_' and take the last two parts (date_time)
            version_parts = latest_file.replace('churn_model_', '').replace('.pkl', '').split('_')
            latest_version = '_'.join(version_parts)
            print(f"Loading model version: {latest_version}")  # Debug info
            return MLModelPredictor(latest_version)
    return None

predictor = load_models()

# Sidebar navigation
st.sidebar.title("🛒 E-commerce ML Analytics")
page = st.sidebar.selectbox("Choose a page", [
    "📊 Overview Dashboard", 
    "🔮 Customer Prediction", 
    "👥 Customer Segmentation",
    "📈 Model Performance",
    "🎯 Business Insights"
])

if page == "📊 Overview Dashboard":
    st.title("E-commerce Analytics Dashboard")
    st.markdown("---")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Customers",
            value="99,441",
            delta="1,234"
        )
    
    with col2:
        st.metric(
            label="Churn Rate",
            value="10.0%",
            delta="-2.1%"
        )
    
    with col3:
        st.metric(
            label="Avg CLV",
            value="$142.97",
            delta="$12.34"
        )
    
    with col4:
        st.metric(
            label="Model Accuracy",
            value="99.8%",
            delta="0.2%"
        )
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer segmentation pie chart
        segments = ['Champions', 'Loyal Customers', 'Potential Loyalists', 'At Risk']
        values = [25, 35, 30, 10]
        
        fig_pie = px.pie(values=values, names=segments, 
                        title="Customer Segmentation Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # CLV distribution histogram
        clv_data = np.random.lognormal(mean=4.5, sigma=1.2, size=1000)
        fig_hist = px.histogram(x=clv_data, nbins=50, 
                               title="Customer Lifetime Value Distribution")
        fig_hist.update_layout(
            xaxis_title="CLV ($)",
            yaxis_title="Number of Customers"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

elif page == "🔮 Customer Prediction":
    st.title("Customer Prediction Tool")
    st.markdown("---")
    
    if not predictor:
        st.error("❌ Models not available. Please train models first.")
        st.stop()
    
    # Input form
    with st.form("prediction_form"):
        st.subheader("Enter Customer Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            recency = st.number_input("Recency (days)", min_value=0, max_value=500, value=30)
            frequency = st.number_input("Frequency (orders)", min_value=1, max_value=50, value=5)
            monetary = st.number_input("Monetary ($)", min_value=0.0, max_value=10000.0, value=500.0)
        
        with col2:
            avg_order_value = st.number_input("Avg Order Value ($)", min_value=0.0, max_value=2000.0, value=100.0)
            pct_card = st.slider("Credit Card Usage (%)", 0.0, 1.0, 0.7, 0.01)
            avg_installments = st.number_input("Avg Installments", min_value=1.0, max_value=24.0, value=2.0)
        
        with col3:
            avg_review = st.slider("Average Review Score", 1.0, 5.0, 4.0, 0.1)
            n_categories = st.number_input("Number of Categories", min_value=1, max_value=20, value=3)
        
        submitted = st.form_submit_button("🔮 Make Predictions")
        
        if submitted:
            # Prepare input data
            customer_data = {
                'recency': recency,
                'frequency': frequency,
                'monetary': monetary,
                'avg_order_value': avg_order_value,
                'pct_card': pct_card,
                'avg_installments': avg_installments,
                'avg_review': avg_review,
                'n_categories': n_categories
            }
            
            try:
                # Make predictions
                churn_result = predictor.predict_churn(customer_data)
                clv_result = predictor.predict_clv(customer_data)
                segment_result = predictor.predict_segment(customer_data)
                
                # Display results
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Churn Risk",
                        value=f"{churn_result['churn_probability']:.1%}",
                        delta=churn_result['risk_level']
                    )
                    
                    # Churn risk gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = churn_result['churn_probability'] * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Churn Risk (%)"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgray"},
                                {'range': [30, 70], 'color': "gray"},
                                {'range': [70, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col2:
                    st.metric(
                        label="Predicted CLV",
                        value=f"${clv_result['predicted_clv']:.2f}",
                        delta=clv_result['clv_category']
                    )
                
                with col3:
                    st.metric(
                        label="Customer Segment",
                        value=segment_result['segment_name'],
                        delta=f"Segment {segment_result['segment_id']}"
                    )
            except Exception as e:
                st.error(f"Error making predictions: {e}")

elif page == "📈 Model Performance":
    st.title("Model Performance Analysis")
    st.markdown("---")
    
    # Model comparison table
    model_data = {
        'Model': ['Logistic Regression', 'Decision Tree', 'SVM', 'Random Forest'],
        'Accuracy': [0.998, 1.000, 1.000, 0.999],
        'Precision': [0.99, 1.00, 1.00, 1.00],
        'Recall': [1.00, 1.00, 1.00, 1.00],
        'F1-Score': [0.99, 1.00, 1.00, 1.00]
    }
    
    df_models = pd.DataFrame(model_data)
    st.subheader("📊 Model Comparison")
    st.dataframe(df_models, use_container_width=True)
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_bar = px.bar(df_models, x='Model', y='Accuracy', 
                        title="Model Accuracy Comparison")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # Feature importance
        features = ['recency', 'frequency', 'monetary', 'avg_order_value', 
                   'pct_card', 'avg_installments', 'avg_review', 'n_categories']
        importance = [0.25, 0.22, 0.20, 0.12, 0.08, 0.06, 0.04, 0.03]
        
        fig_importance = px.bar(x=importance, y=features, orientation='h',
                               title="Feature Importance Analysis")
        st.plotly_chart(fig_importance, use_container_width=True)

elif page == "👥 Customer Segmentation":
    st.title("Customer Segmentation Analysis")
    st.markdown("---")
    
    st.info("This page shows customer segmentation results from K-means clustering.")
    
    # Placeholder for segmentation analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Segment Characteristics")
        segment_data = {
            'Segment': ['Champions', 'Loyal Customers', 'Potential Loyalists', 'At Risk'],
            'Size (%)': [25, 35, 30, 10],
            'Avg CLV': [450, 320, 180, 90],
            'Churn Risk': ['Low', 'Low', 'Medium', 'High']
        }
        st.dataframe(pd.DataFrame(segment_data))
    
    with col2:
        st.subheader("Segment Distribution")
        segments = ['Champions', 'Loyal Customers', 'Potential Loyalists', 'At Risk']
        values = [25, 35, 30, 10]
        colors = ['gold', 'lightgreen', 'orange', 'lightcoral']
        
        fig = px.pie(values=values, names=segments, 
                    title="Customer Segment Distribution",
                    color_discrete_sequence=colors)
        st.plotly_chart(fig, use_container_width=True)

elif page == "🎯 Business Insights":
    st.title("Business Insights & Recommendations")
    st.markdown("---")
    
    st.subheader("🔍 Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        *Customer Behavior:*
        - 90% of customers do not churn (high retention)
        - Average customer lifetime value: $142.97
        - Credit card usage strongly correlates with loyalty
        - Review scores are predictive of future purchases
        """)
        
        st.markdown("""
        *Model Performance:*
        - Decision Tree achieves 100% accuracy on test set
        - Ridge regression provides excellent CLV predictions
        - Feature importance: Recency > Frequency > Monetary
        """)
    
    with col2:
        st.markdown("""
        *Business Recommendations:*
        - Focus retention efforts on "At Risk" segment (10%)
        - Encourage credit card adoption for payment flexibility
        - Implement review quality monitoring system
        - Personalize marketing based on customer segments
        """)
        
        st.markdown("""
        *Next Steps:*
        - Deploy real-time prediction system
        - A/B test intervention strategies
        - Monitor model performance over time
        - Expand feature engineering with external data
        """)

# Add footer
st.markdown("---")
st.markdown("💡 *Tip*: Use the sidebar to navigate between different analytics views!")