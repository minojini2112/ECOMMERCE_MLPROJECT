# E-commerce ML Analytics Dashboard

A comprehensive machine learning dashboard for e-commerce customer analytics, featuring churn prediction, customer lifetime value (CLV) estimation, and customer segmentation.

## 🚀 Features

- **Customer Churn Prediction**: Predict customer churn risk based on recency, frequency, and monetary value
- **CLV Estimation**: Calculate customer lifetime value with business logic
- **Customer Segmentation**: Classify customers into Champions, Loyal Customers, Potential Loyalists, and At Risk segments
- **Interactive Dashboard**: Beautiful Streamlit interface with real-time predictions
- **Model Performance Analytics**: View model accuracy and feature importance

## 📊 Dashboard Pages

1. **Overview Dashboard**: Key metrics and visualizations
2. **Customer Prediction**: Interactive prediction tool
3. **Customer Segmentation**: Segmentation analysis
4. **Model Performance**: Model comparison and metrics
5. **Business Insights**: Recommendations and findings

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the training pipeline:
```bash
python main.py
```

4. Launch the dashboard:
```bash
streamlit run streamlit_app.py
```

## 📈 Usage

1. **Train Models**: Run `python main.py` to train ML models on e-commerce data
2. **Launch Dashboard**: Run `streamlit run streamlit_app.py` to start the web interface
3. **Make Predictions**: Use the Customer Prediction page to input customer data and get predictions

## 🔧 Model Details

- **Churn Prediction**: Rule-based system with business logic
- **CLV Estimation**: Multi-factor calculation based on customer behavior
- **Segmentation**: K-means clustering with PCA dimensionality reduction

## 📝 Requirements

- Python 3.8+
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Plotly
- Matplotlib
- Seaborn

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.
