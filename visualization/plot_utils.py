import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

# Try importing plotly, but provide fallbacks if not available
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Installing matplotlib and seaborn for fallback visualizations.")

def plot_risk_distribution(risk_scores):
    """Plot histogram of risk scores."""
    if PLOTLY_AVAILABLE:
        fig = px.histogram(
            risk_scores, 
            nbins=20,
            title="Transaction Risk Distribution",
            labels={"value": "Risk Score", "count": "Number of Transactions"},
            color_discrete_sequence=["#3A86FF"]
        )
        fig.add_vline(x=80, line_dash="dash", line_color="red", 
                     annotation_text="High Risk Threshold")
        return fig
    else:
        # Matplotlib fallback - return Figure object instead of plt
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(risk_scores, bins=20, color='#3A86FF')
        ax.axvline(x=80, linestyle='--', color='red', label='High Risk Threshold')
        ax.set_title('Transaction Risk Distribution')
        ax.set_xlabel('Risk Score')
        ax.set_ylabel('Number of Transactions')
        ax.legend()
        return fig

def plot_amount_vs_risk(df, risk_scores):
    """Plot scatter of transaction amount vs risk score."""
    if 'amount' not in df.columns:
        return None
        
    plot_df = pd.DataFrame({
        'Transaction Amount': df['amount'],
        'Risk Score': risk_scores,
        'Fraud': df['is_fraud'] if 'is_fraud' in df.columns else np.zeros(len(df))
    })
    
    if PLOTLY_AVAILABLE:
        fig = px.scatter(
            plot_df,
            x='Transaction Amount',
            y='Risk Score',
            color='Fraud',
            title='Transaction Amount vs Risk Score',
            color_discrete_sequence=["blue", "red"],
            opacity=0.7
        )
        return fig
    else:
        # Matplotlib fallback - return Figure object instead of plt
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'is_fraud' in df.columns:
            for fraud_val in [0, 1]:
                subset = plot_df[plot_df['Fraud'] == fraud_val]
                ax.scatter(
                    subset['Transaction Amount'], 
                    subset['Risk Score'], 
                    color='blue' if fraud_val == 0 else 'red',
                    alpha=0.7,
                    label=f'{"Fraud" if fraud_val == 1 else "Legitimate"}'
                )
            ax.legend()
        else:
            ax.scatter(plot_df['Transaction Amount'], plot_df['Risk Score'], alpha=0.7)
            
        ax.set_title('Transaction Amount vs Risk Score')
        ax.set_xlabel('Transaction Amount')
        ax.set_ylabel('Risk Score')
        return fig

def plot_risk_timeline(df, risk_scores):
    """Plot risk scores over time."""
    if 'time' not in df.columns:
        return None
        
    plot_df = pd.DataFrame({
        'Time': df['time'],
        'Risk Score': risk_scores
    })
    plot_df = plot_df.sort_values('Time')
    
    if PLOTLY_AVAILABLE:
        fig = px.line(
            plot_df,
            x='Time',
            y='Risk Score',
            title='Risk Score Timeline'
        )
        fig.add_hline(y=80, line_dash="dash", line_color="red", 
                     annotation_text="High Risk Threshold")
        return fig
    else:
        # Matplotlib fallback - return Figure object instead of plt
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(plot_df['Time'], plot_df['Risk Score'])
        ax.axhline(y=80, linestyle='--', color='red', label='High Risk Threshold')
        ax.set_title('Risk Score Timeline')
        ax.set_xlabel('Time')
        ax.set_ylabel('Risk Score')
        ax.legend()
        return fig

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    if PLOTLY_AVAILABLE:
        fig = px.imshow(
            cm,
            text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=['Not Fraud', 'Fraud'],
            y=['Not Fraud', 'Fraud'],
            color_continuous_scale='Blues',
            title='Confusion Matrix'
        )
        return fig
    else:
        # Matplotlib fallback - return Figure object instead of plt
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'],
            ax=ax
        )
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        return fig

def plot_roc_curve(y_true, y_score):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name='ROC Curve',
            line=dict(color='royalblue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            legend=dict(y=0.5, traceorder='reversed', font_size=12)
        )
        return fig
    else:
        # Matplotlib fallback - return Figure object instead of plt
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label='ROC Curve', color='royalblue', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend()
        return fig

def plot_precision_recall_curve(y_true, y_score):
    """Plot precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            name='PR Curve',
            fill='tozeroy',
            line=dict(color='royalblue', width=2)
        ))
        
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision',
            yaxis=dict(range=[0, 1.05]),
            xaxis=dict(range=[0, 1])
        )
        return fig
    else:
        # Matplotlib fallback - return Figure object instead of plt
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='royalblue', linewidth=2)
        ax.fill_between(recall, precision, alpha=0.2, color='royalblue')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.set_ylim([0, 1.05])
        ax.set_xlim([0, 1])
        return fig

def plot_benford_distribution(first_digits):
    """Plot Benford's Law distribution."""
    # Count the occurrences of each first digit
    digit_counts = first_digits.value_counts().sort_index()
    
    # Calculate expected Benford's Law distribution
    digits = np.arange(1, 10)
    benford_expected = np.log10(1 + 1/digits) * len(first_digits)
    
    # Create DataFrame for plotting
    benford_df = pd.DataFrame({
        'Digit': digits,
        'Actual Count': [digit_counts.get(d, 0) for d in digits],
        'Expected Count': benford_expected
    })
    
    if PLOTLY_AVAILABLE:
        fig = px.bar(
            benford_df,
            x='Digit',
            y=['Actual Count', 'Expected Count'],
            title="First-Digit Distribution (Benford's Law)",
            barmode='group'
        )
        return fig
    else:
        # Matplotlib fallback - return Figure object instead of plt
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.35
        x = np.arange(len(digits))
        
        ax.bar(x - bar_width/2, benford_df['Actual Count'], bar_width, label='Actual')
        ax.bar(x + bar_width/2, benford_df['Expected Count'], bar_width, label='Expected (Benford\'s Law)')
        
        ax.set_xlabel('First Digit')
        ax.set_ylabel('Count')
        ax.set_title('First-Digit Distribution (Benford\'s Law)')
        ax.set_xticks(x)
        ax.set_xticklabels(digits)
        ax.legend()
        return fig

def plot_pca_visualization(pca_components, labels=None):
    """Plot PCA visualization of the data."""
    if pca_components.shape[1] < 2:
        return None
        
    # Use only the first two components
    x = pca_components[:, 0]
    y = pca_components[:, 1]
    
    if PLOTLY_AVAILABLE:
        if labels is not None:
            fig = px.scatter(
                x=x, y=y, 
                color=labels,
                title='PCA Visualization',
                labels={'x': 'First Principal Component', 'y': 'Second Principal Component'}
            )
        else:
            fig = px.scatter(
                x=x, y=y,
                title='PCA Visualization',
                labels={'x': 'First Principal Component', 'y': 'Second Principal Component'}
            )
        return fig
    else:
        # Matplotlib fallback - return Figure object instead of plt
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if labels is not None:
            for label in np.unique(labels):
                mask = labels == label
                ax.scatter(x[mask], y[mask], label=f'Class {label}', alpha=0.7)
            ax.legend()
        else:
            ax.scatter(x, y, alpha=0.7)
            
        ax.set_title('PCA Visualization')
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        return fig 