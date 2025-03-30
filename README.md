# Credit Card Fraud Detection System

A comprehensive fraud detection system for financial transactions with predictive modeling and an interactive Streamlit dashboard.

## Features

- **Data Preprocessing**: Automatic feature engineering for financial transaction data
- **Predictive Modeling**: Multiple model options (Random Forest, Logistic Regression, XGBoost)
- **Risk Scoring**: Composite risk score calculation incorporating various risk factors
- **Interactive Dashboard**: Multi-tab interface for exploring fraud patterns
- **Advanced Visualizations**: Visual analysis of transaction patterns and model performance
- **Scenario Analysis**: Cost-benefit analysis for different fraud prevention thresholds

## Dashboard Components

1. **Overview**: High-level risk metrics and summary statistics
2. **Transaction Analysis**: Detailed view of transaction patterns and high-risk transactions
3. **Pattern Recognition**: Advanced analysis using Benford's Law and PCA visualization
4. **Account Analysis**: Account-level risk metrics (if account data is available)
5. **Predictive Modeling**: Model performance metrics and feature importance
6. **Scenario Analysis**: Cost-benefit calculations for different threshold settings

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Ensure your credit card transaction dataset is named `data.csv` in the root directory.
2. Run the Streamlit application:
```bash
streamlit run app.py
```

3. Open your browser at `http://localhost:8501` to view the dashboard.

## Dataset Requirements

The application expects a credit card transaction dataset with the following columns:
- Transaction amount (will be renamed to 'amount' if named differently)
- Class/fraud label (will be renamed to 'is_fraud' if named differently)
- Time information (optional, will be renamed to 'time' if named differently)
- Other transaction features (V1, V2, etc. for PCA visualization)

For example datasets, you can use:
- [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle

## Customization

You can modify the following aspects of the application:
- Risk threshold in the sidebar
- Model type selection
- Cost parameters for scenario analysis
- Data filters based on transaction amount

## Project Structure

```
fraud_detection/
├── preprocessing/
│   ├── __init__.py
│   └── feature_engineering.py  # Data preprocessing and feature creation
├── models/
│   ├── __init__.py
│   └── model_utils.py          # Model training and evaluation functions
├── visualization/
│   ├── __init__.py
│   └── plot_utils.py           # Visualization utility functions
├── app.py                      # Main Streamlit application
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Future Improvements

- Network analysis for detecting fraud rings
- Time series analysis for detecting seasonal patterns
- API integration for real-time fraud detection
- User authentication and role-based access control
- Export/reporting functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details. 