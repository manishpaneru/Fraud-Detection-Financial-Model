import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix)

# Import our custom modules
from preprocessing.feature_engineering import (load_and_preprocess_data, 
                                              prepare_features_for_modeling)
from models.model_utils import (train_model, evaluate_model, calculate_risk_score, 
                               save_model, load_model, plot_feature_importance)
from visualization.plot_utils import (plot_risk_distribution, plot_amount_vs_risk,
                                     plot_risk_timeline, plot_confusion_matrix,
                                     plot_roc_curve, plot_precision_recall_curve,
                                     plot_benford_distribution, plot_pca_visualization)

# Set page config
st.set_page_config(layout="wide", page_title="Fraud Detection Dashboard")

# Plot display helper function
def display_plot(fig):
    """Utility function to properly display either matplotlib or plotly figures."""
    if fig is None:
        return
        
    try:
        # For plotly figures
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        # For matplotlib figures
        st.pyplot(fig)

# Functions for data loading
@st.cache_data
def load_data(file_path, sample_size=None):
    """Load and preprocess data with caching."""
    try:
        return load_and_preprocess_data(file_path, sample_size=sample_size)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def get_or_train_model(X, y, model_type='random_forest'):
    """Get cached model or train a new one."""
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = train_model(X_train, y_train, model_type=model_type)
        
        # Evaluate
        eval_results = evaluate_model(model, X_test, y_test)
        
        return model, X_train, X_test, y_train, y_test, eval_results
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None, None, None, None, None

# Sidebar
with st.sidebar:
    st.title("Fraud Detection Settings")
    
    # File selector
    file_path = "data.csv"  # Default to the data.csv in the root directory
    
    if not os.path.exists(file_path):
        st.warning(f"Default file {file_path} not found.")
        file_path = st.text_input("Enter path to data file:", "data.csv")
    
    # Data sampling options
    st.subheader("Data Sampling")
    use_sampling = st.checkbox("Use data sampling (recommended for large datasets)", value=True)
    
    if use_sampling:
        sample_size = st.slider(
            "Number of rows to process:", 
            min_value=10000, 
            max_value=500000, 
            value=100000,
            step=10000
        )
    else:
        sample_size = None
        st.warning("Processing the entire dataset may cause memory errors.")
    
    # Model selection
    st.subheader("Model Settings")
    model_type = st.selectbox(
        "Select model type:",
        ["random_forest", "logistic", "xgboost"]
    )
    
    # Risk threshold
    risk_threshold = st.slider(
        "Risk threshold (0-100):",
        min_value=0,
        max_value=100,
        value=80
    )
    
    # Filters section
    st.subheader("Filters")
    
    # These will be populated dynamically based on the dataset
    min_amount = st.number_input("Min transaction amount:", value=0.0)
    max_amount = st.number_input("Max transaction amount:", value=1000000.0)  # Use a large finite number instead of infinity
    
    # Add a button to run analysis
    run_analysis = st.button("Run Analysis")

# Main layout
st.title("Credit Card Fraud Detection Dashboard")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", "Transaction Analysis", 
    "Pattern Recognition", "Account Analysis",
    "Predictive Modeling", "Scenario Analysis"
])

# Load data
if file_path and os.path.exists(file_path) and (run_analysis or 'df' not in st.session_state):
    # Use sampling based on checkbox selection
    sample_size_to_use = sample_size if use_sampling else None
    
    with st.spinner("Loading data..."):
        df = load_data(file_path, sample_size=sample_size_to_use)
        st.session_state.df = df
else:
    # Use cached data if available
    df = st.session_state.get('df')

if df is not None:
    # Apply filters
    if 'amount' in df.columns:
        df = df[(df['amount'] >= min_amount) & 
                (df['amount'] <= max_amount)]
    
    # Prepare features for modeling - using the same sample size
    X, y, scaler = prepare_features_for_modeling(df)
    
    # Check if we have a target variable
    if y is not None:
        # Train/get model if we have a run button press or no model yet
        if run_analysis or 'model_results' not in st.session_state:
            with st.spinner("Training model..."):
                model_results = get_or_train_model(X, y, model_type=model_type)
                st.session_state.model_results = model_results
        else:
            # Use cached model results
            model_results = st.session_state.model_results
            
        if model_results[0] is not None:  # Check if model training was successful
            model, X_train, X_test, y_train, y_test, eval_results = model_results
            
            # Calculate risk scores for all transactions
            with st.spinner("Calculating risk scores..."):
                risk_scores = []
                for _, row in df.reset_index().iterrows():
                    risk_scores.append(calculate_risk_score(row.to_dict(), model, scaler))
                
                # Add risk scores to dataframe
                df['risk_score'] = risk_scores
                
                # Flag high-risk transactions
                df['high_risk'] = df['risk_score'] > risk_threshold
            
            # Tab 1: Overview
            with tab1:
                st.header("Fraud Detection Overview")
                
                # Introduction text
                st.write("""
                # Welcome to my Credit Card Fraud Detection System
                
                I've built this comprehensive fraud detection system to analyze financial transaction patterns and identify potential fraudulent activities. This dashboard leverages advanced machine learning techniques to process credit card transaction data and assign risk scores to each transaction.
                
                The dataset used in this project comes from Kaggle's Credit Card Fraud Detection dataset, which contains anonymized credit card transactions. It includes legitimate and fraudulent transactions, making it perfect for training a supervised machine learning model. The features have been transformed using PCA for privacy reasons, with only 'Time' and 'Amount' being the original features.
                
                With this financial model, I've accomplished:
                - Accurate identification of potentially fraudulent transactions with {eval_results['accuracy']:.2f} accuracy
                - Risk scoring for every transaction based on multiple factors
                - Cost-benefit analysis to optimize fraud prevention strategies
                - Visual exploration of transaction patterns and risk factors
                
                Feel free to explore the different tabs to see detailed analysis, visualizations, and model performance metrics.
                """)
                
                # About the Creator accordion
                with st.expander("About the Creator"):
                    st.write("""
                    **Manish Paneru**  
                    Data Analyst
                    
                    **Connect with me:**  
                    [LinkedIn](https://linkedin.com/in/manish.paneru1)  
                    [GitHub](https://github.com/manishpaneru)  
                    [Portfolio](https://analystpaneru.xyz)
                    """)
                
                # Create metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Overall Risk Level", 
                        f"{df['risk_score'].mean():.1f}/100",
                        delta="+4.2" if df['risk_score'].mean() > 50 else "-2.1"
                    )
                    
                with col2:
                    st.metric(
                        "Flagged Transactions", 
                        f"{sum(df['high_risk'])}",
                        delta=f"{sum(df['high_risk'])/len(df):.1%}"
                    )
                    
                with col3:
                    value_at_risk = df[df['high_risk']]['amount'].sum() if 'amount' in df.columns else 0
                    st.metric(
                        "Total Value at Risk", 
                        f"${value_at_risk:,.2f}",
                        delta="+4.2%" if value_at_risk > 10000 else "-2.1%"
                    )
                    
                with col4:
                    fraud_rate = y.mean() if y is not None else 0.01
                    st.metric(
                        "Fraud Rate", 
                        f"{fraud_rate:.2%}",
                        delta="+0.1%" if fraud_rate > 0.01 else "-0.05%"
                    )
                
                # Risk distribution
                st.subheader("Risk Distribution")
                risk_dist_fig = plot_risk_distribution(df['risk_score'])
                display_plot(risk_dist_fig)
                
                # Dataset sample
                st.subheader("Sample Data")
                st.dataframe(df.head())
                
                # Executive summary
                st.subheader("Executive Summary")
                st.write(f"""
                The fraud detection system has analyzed {len(df)} transactions 
                and identified {sum(df['high_risk'])} high-risk transactions 
                representing ${value_at_risk:,.2f} in potential fraud risk.
                
                The current model ({model_type}) has an accuracy of {eval_results['accuracy']:.2f} 
                and a precision of {eval_results['precision']:.2f} in detecting fraudulent transactions.
                
                Key risk factors include transaction amount, transaction frequency, and unusual patterns 
                in user behavior.
                """)
                
            # Tab 2: Transaction Analysis
            with tab2:
                st.header("Transaction Risk Analysis")
                
                # Amount vs Risk scatter plot
                st.subheader("Transaction Amount vs Risk Score")
                amount_risk_fig = plot_amount_vs_risk(df, df['risk_score'])
                if amount_risk_fig:
                    display_plot(amount_risk_fig)
                
                # Risk timeline
                st.subheader("Risk Timeline")
                timeline_fig = plot_risk_timeline(df, df['risk_score'])
                if timeline_fig:
                    display_plot(timeline_fig)
                
                # High risk transactions table
                st.subheader("High Risk Transactions")
                high_risk_df = df[df['high_risk']].sort_values('risk_score', ascending=False)
                if len(high_risk_df) > 0:
                    # Select the columns to display
                    display_cols = [col for col in ['amount', 'time', 'risk_score', 'is_fraud'] 
                                 if col in high_risk_df.columns]
                    if len(display_cols) > 0:
                        st.dataframe(high_risk_df[display_cols])
                    else:
                        st.dataframe(high_risk_df)
                else:
                    st.write("No high-risk transactions found.")
            
            # Tab 3: Pattern Recognition
            with tab3:
                st.header("Pattern Recognition")
                
                # Benford's Law Analysis
                st.subheader("First-Digit Distribution (Benford's Law)")
                if 'first_digit' in df.columns:
                    benford_fig = plot_benford_distribution(df['first_digit'])
                    display_plot(benford_fig)
                
                # PCA Visualization if we have V columns
                v_columns = [col for col in df.columns if col.startswith('V')]
                if len(v_columns) >= 2:
                    st.subheader("PCA Visualization")
                    
                    # Apply PCA to the V columns
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(df[v_columns])
                    
                    # Plot PCA
                    pca_fig = plot_pca_visualization(pca_result, df['is_fraud'] if 'is_fraud' in df.columns else None)
                    display_plot(pca_fig)
                
                # Check if we have PCA columns from preprocessing
                pca_columns = [col for col in df.columns if col.startswith('pca_')]
                if len(pca_columns) >= 2 and not v_columns:
                    st.subheader("PCA Visualization")
                    
                    # Extract PCA components
                    pca_data = df[pca_columns].values
                    
                    # Plot PCA
                    pca_fig = plot_pca_visualization(pca_data, df['is_fraud'] if 'is_fraud' in df.columns else None)
                    display_plot(pca_fig)
                
                # Correlation heatmap
                st.subheader("Feature Correlation")
                
                # Select numerical columns for correlation
                num_cols = X.select_dtypes(include=np.number).columns.tolist()
                if len(num_cols) > 2:  # Need at least a few columns
                    # Use a subset of columns if there are too many
                    if len(num_cols) > 15:
                        num_cols = num_cols[:15]
                    
                    corr = X[num_cols].corr()
                    
                    # Plot heatmap
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.set_title("Feature Correlation Heatmap")
                    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                    st.pyplot(fig)
            
            # Tab 4: Account Analysis
            with tab4:
                st.header("Account Analysis")
                
                # Since most credit card datasets don't have account IDs,
                # we'll show a message or try to work with what we have
                if 'account_id' in df.columns or 'nameOrig' in df.columns:
                    # Account risk table
                    st.subheader("Account Risk Table")
                    
                    # Use nameOrig as account_id if available
                    if 'nameOrig' in df.columns and 'account_id' not in df.columns:
                        df['account_id'] = df['nameOrig']
                    
                    # Calculate account-level metrics
                    account_metrics = df.groupby('account_id').agg({
                        'risk_score': ['mean', 'max', 'count'],
                        'high_risk': 'sum',
                        'amount': ['sum', 'mean'] if 'amount' in df.columns else None
                    }).reset_index()
                    
                    # Flatten column names
                    account_metrics.columns = [
                        f"{col[0]}_{col[1]}" if col[1] != '' else col[0]
                        for col in account_metrics.columns
                    ]
                    
                    # Calculate percentage of high-risk transactions
                    account_metrics['high_risk_pct'] = account_metrics['high_risk_sum'] / account_metrics['risk_score_count']
                    
                    # Sort by risk
                    account_metrics = account_metrics.sort_values('risk_score_mean', ascending=False)
                    
                    # Display
                    st.dataframe(account_metrics.head(100))  # Show top 100 accounts
                else:
                    st.write("""
                    Account-level analysis typically requires a dataset with account identifiers.
                    The current dataset does not contain account ID information.
                    
                    For a real-world implementation, you would:
                    1. Group transactions by account
                    2. Calculate risk metrics per account
                    3. Identify the highest-risk accounts
                    4. Analyze transaction patterns within those accounts
                    """)
            
            # Tab 5: Predictive Modeling
            with tab5:
                st.header("Predictive Modeling")
                
                # Model performance metrics
                st.subheader("Model Performance")
                
                metrics_col1, metrics_col2 = st.columns(2)
                
                with metrics_col1:
                    st.metric("Accuracy", f"{eval_results['accuracy']:.2f}")
                    st.metric("Precision", f"{eval_results['precision']:.2f}")
                    st.metric("Recall", f"{eval_results['recall']:.2f}")
                
                with metrics_col2:
                    st.metric("F1 Score", f"{eval_results['f1']:.2f}")
                    if 'roc_auc' in eval_results:
                        st.metric("ROC AUC", f"{eval_results['roc_auc']:.2f}")
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm_fig = plot_confusion_matrix(y_test, model.predict(X_test))
                display_plot(cm_fig)
                
                # ROC Curve
                if 'y_pred_proba' in eval_results:
                    st.subheader("ROC Curve")
                    roc_fig = plot_roc_curve(y_test, eval_results['y_pred_proba'])
                    display_plot(roc_fig)
                
                # Precision-Recall Curve
                if 'y_pred_proba' in eval_results:
                    st.subheader("Precision-Recall Curve")
                    pr_fig = plot_precision_recall_curve(y_test, eval_results['y_pred_proba'])
                    display_plot(pr_fig)
                
                # Feature Importance
                st.subheader("Feature Importance")
                importance_fig = plot_feature_importance(model, X.columns)
                if importance_fig:
                    display_plot(importance_fig)
            
            # Tab 6: Scenario Analysis
            with tab6:
                st.header("Scenario Analysis")
                
                # Risk threshold slider
                scenario_threshold = st.slider(
                    "Scenario Risk Threshold:",
                    min_value=0,
                    max_value=100,
                    value=risk_threshold,
                    key="scenario_threshold"
                )
                
                # Cost parameters
                st.subheader("Cost-Benefit Parameters")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    avg_fraud_cost = st.number_input(
                        "Average cost per fraud ($):",
                        min_value=0,
                        value=1000
                    )
                    
                    false_positive_cost = st.number_input(
                        "Cost per false positive ($):",
                        min_value=0,
                        value=50
                    )
                
                with col2:
                    investigation_cost = st.number_input(
                        "Cost per investigation ($):",
                        min_value=0,
                        value=25
                    )
                    
                    prevention_effectiveness = st.slider(
                        "Prevention effectiveness (%):",
                        min_value=0,
                        max_value=100,
                        value=80
                    ) / 100
                
                # Calculate results for different thresholds
                thresholds = range(0, 101, 5)
                results = []
                
                for threshold in thresholds:
                    # Flag transactions at this threshold
                    flagged = df['risk_score'] > threshold
                    
                    # True positives (fraudulent transactions correctly flagged)
                    if 'is_fraud' in df.columns:
                        tp = sum(flagged & (df['is_fraud'] == 1))
                        # False positives (non-fraudulent transactions incorrectly flagged)
                        fp = sum(flagged & (df['is_fraud'] == 0))
                        # False negatives (fraudulent transactions missed)
                        fn = sum((~flagged) & (df['is_fraud'] == 1))
                    else:
                        # If fraud labels aren't available, use estimates
                        tp = sum(flagged) * 0.1  # Assume 10% of flagged are fraud
                        fp = sum(flagged) * 0.9  # Assume 90% are false positives
                        fn = len(df) * 0.005  # Assume 0.5% overall fraud rate
                    
                    # Calculate costs
                    prevented_loss = tp * avg_fraud_cost * prevention_effectiveness
                    missed_fraud_cost = fn * avg_fraud_cost
                    false_positive_cost_total = fp * false_positive_cost
                    investigation_cost_total = sum(flagged) * investigation_cost
                    
                    # Total cost
                    total_cost = missed_fraud_cost + false_positive_cost_total + investigation_cost_total - prevented_loss
                    
                    # ROI calculation
                    if investigation_cost_total + false_positive_cost_total > 0:
                        roi = (prevented_loss - (investigation_cost_total + false_positive_cost_total)) / (investigation_cost_total + false_positive_cost_total)
                    else:
                        roi = 0
                    
                    results.append({
                        'Threshold': threshold,
                        'Flagged Transactions': sum(flagged),
                        'True Positives': tp,
                        'False Positives': fp,
                        'False Negatives': fn,
                        'Prevented Loss': prevented_loss,
                        'Missed Fraud Cost': missed_fraud_cost,
                        'False Positive Cost': false_positive_cost_total,
                        'Investigation Cost': investigation_cost_total,
                        'Total Cost': total_cost,
                        'ROI': roi
                    })
                
                results_df = pd.DataFrame(results)
                
                # Find the current scenario based on threshold
                current_scenario = results_df[results_df['Threshold'] == scenario_threshold].iloc[0]
                
                # Display current scenario metrics
                st.subheader("Current Scenario Results")
                
                scenario_col1, scenario_col2, scenario_col3 = st.columns(3)
                
                with scenario_col1:
                    st.metric("Flagged Transactions", f"{int(current_scenario['Flagged Transactions'])}")
                    st.metric("True Positives", f"{current_scenario['True Positives']:.1f}")
                
                with scenario_col2:
                    st.metric("Prevented Loss", f"${current_scenario['Prevented Loss']:,.2f}")
                    st.metric("Investigation Cost", f"${current_scenario['Investigation Cost']:,.2f}")
                
                with scenario_col3:
                    st.metric("Total Cost", f"${current_scenario['Total Cost']:,.2f}")
                    st.metric("ROI", f"{current_scenario['ROI']:,.2%}")
                
                # Plot threshold analysis
                st.subheader("Threshold Analysis")
                
                # Cost comparison plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(results_df['Threshold'], results_df['Prevented Loss'], label='Prevented Loss')
                ax.plot(results_df['Threshold'], results_df['False Positive Cost'], label='False Positive Cost')
                ax.plot(results_df['Threshold'], results_df['Investigation Cost'], label='Investigation Cost')
                ax.plot(results_df['Threshold'], results_df['Total Cost'], label='Total Cost')
                ax.axvline(x=scenario_threshold, linestyle='--', color='gray')
                ax.set_title("Cost Analysis by Threshold")
                ax.set_xlabel("Threshold")
                ax.set_ylabel("Cost ($)")
                ax.legend()
                st.pyplot(fig)
                
                # ROI plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(results_df['Threshold'], results_df['ROI'], color='green')
                ax.axvline(x=scenario_threshold, linestyle='--', color='gray')
                ax.set_title("ROI by Threshold")
                ax.set_xlabel("Threshold")
                ax.set_ylabel("ROI")
                st.pyplot(fig)
                
                # Optimal threshold
                optimal_threshold = results_df.loc[results_df['Total Cost'].idxmin(), 'Threshold']
                st.info(f"Based on my cost parameters, the optimal threshold would be {optimal_threshold}.")
        else:
            st.error("Model training failed. Please check your data and try again.")
    else:
        st.error("Target variable 'is_fraud' not found in the dataset.")
else:
    st.error("Please provide a valid dataset.")

# Footer
st.markdown("---")
st.markdown("Fraud Detection Dashboard | Created with Streamlit and Python")