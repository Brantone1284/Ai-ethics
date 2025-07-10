import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

# Configure Streamlit
st.set_page_config(layout="wide")
st.title("COMPAS Recidivism Risk Score Bias Audit")
st.write("""
Auditing racial bias in COMPAS risk scores using AI Fairness 360 toolkit.
Data source: [ProPublica COMPAS Analysis](https://github.com/propublica/compas-analysis)
""")

@st.cache_data
def load_data():
    """Load and preprocess COMPAS dataset"""
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    df = pd.read_csv(url, usecols=['age', 'c_charge_degree', 'race', 'sex', 
                                  'priors_count', 'days_b_screening_arrest', 
                                  'decile_score', 'two_year_recid'])
    
    # Apply ProPublica's filtering criteria
    df = df[
        (df['days_b_screening_arrest'].between(-30, 30)) &
        (df['race'].isin(['African-American', 'Caucasian']))
    ].copy()
    
    # Create binary risk classification
    df['risk'] = (df['decile_score'] > 5).astype(int)
    return df

def create_aif_dataset(df):
    """Create AIF360 dataset object"""
    return BinaryLabelDataset(
        favorable_label=0,
        unfavorable_label=1,
        df=df,
        label_names=['risk'],
        protected_attribute_names=['race']
    )

def calculate_fairness_metrics(dataset):
    """Calculate fairness metrics"""
    metric = BinaryLabelDatasetMetric(
        dataset,
        unprivileged_groups=[{'race': 'African-American'}],
        privileged_groups=[{'race': 'Caucasian'}]
    )
    
    # For performance metrics, we need to compare against actual recidivism
    # Create a copy with true labels
    true_dataset = dataset.copy()
    true_dataset.labels = dataset.features[:, dataset.feature_names.index('two_year_recid')].reshape(-1, 1)
    
    class_metric = ClassificationMetric(
        dataset, true_dataset,
        unprivileged_groups=[{'race': 'African-American'}],
        privileged_groups=[{'race': 'Caucasian'}]
    )
    
    return metric, class_metric

def plot_disparities(metric):
    """Visualize performance disparities"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    metrics_data = {
        'African-American': [
            metric.false_positive_rate(False),
            metric.false_negative_rate(False),
            metric.error_rate(False)
        ],
        'Caucasian': [
            metric.false_positive_rate(True),
            metric.false_negative_rate(True),
            metric.error_rate(True)
        ]
    }
    
    # Create plot
    x = np.arange(3)
    width = 0.35
    
    ax.bar(x - width/2, metrics_data['African-American'], width, label='African-American')
    ax.bar(x + width/2, metrics_data['Caucasian'], width, label='Caucasian')
    
    # Add labels and styling
    ax.set_title('Model Performance Disparities by Race', fontsize=16)
    ax.set_ylabel('Rate', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['False Positive Rate', 'False Negative Rate', 'Error Rate'])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 0.6)
    
    # Add data labels
    for i, (aa, cauc) in enumerate(zip(metrics_data['African-American'], 
                                      metrics_data['Caucasian'])):
        ax.text(i - width/2, aa + 0.02, f'{aa:.2f}', ha='center')
        ax.text(i + width/2, cauc + 0.02, f'{cauc:.2f}', ha='center')
    
    plt.tight_layout()
    return fig

def apply_mitigation(dataset):
    """Apply reweighing bias mitigation"""
    RW = Reweighing(
        unprivileged_groups=[{'race': 'African-American'}],
        privileged_groups=[{'race': 'Caucasian'}]
    )
    return RW.fit_transform(dataset)

# Load data
df = load_data()
aif_data = create_aif_dataset(df)

# Split data
train, test = aif_data.split([0.7], shuffle=True, seed=42)

# Calculate metrics
orig_metric, class_metric = calculate_fairness_metrics(test)

# Display key metrics
st.subheader("Key Fairness Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Disparate Impact", f"{orig_metric.disparate_impact():.3f}", 
            "Fair" if orig_metric.disparate_impact() >= 0.8 else "Unfair", 
            delta_color="inverse")
col2.metric("Statistical Parity Difference", f"{orig_metric.statistical_parity_difference():.3f}",
            "0 = parity")
col3.metric("FPR Ratio (AA vs Caucasian)", 
            f"{class_metric.false_positive_rate(False)/class_metric.false_positive_rate(True):.2f}x")

# Display data summary
with st.expander("Dataset Overview"):
    st.write(f"**Total Records:** {len(df)}")
    st.write(f"**African-American Defendants:** {len(df[df['race'] == 'African-American'])}")
    st.write(f"**Caucasian Defendants:** {len(df[df['race'] == 'Caucasian'])}")
    st.dataframe(df.head())

# Performance disparities visualization
st.subheader("Performance Disparities")
st.pyplot(plot_disparities(class_metric))

# Mitigation section
st.subheader("Bias Mitigation with Reweighing")
st.write("Reweighing adjusts instance weights in the dataset to compensate for bias")

# Apply mitigation
test_transformed = apply_mitigation(test)
transf_metric, _ = calculate_fairness_metrics(test_transformed)

# Display mitigation results
col1, col2 = st.columns(2)
col1.metric("Disparate Impact (Original)", f"{orig_metric.disparate_impact():.3f}")
col1.metric("Disparate Impact (Mitigated)", f"{transf_metric.disparate_impact():.3f}", 
            f"+{(transf_metric.disparate_impact() - orig_metric.disparate_impact()):.3f}")

col2.metric("Stat. Parity Diff. (Original)", f"{orig_metric.statistical_parity_difference():.3f}")
col2.metric("Stat. Parity Diff. (Mitigated)", f"{transf_metric.statistical_parity_difference():.3f}", 
            f"{(transf_metric.statistical_parity_difference() - orig_metric.statistical_parity_difference()):.3f}")

# Findings and recommendations
st.subheader("Key Findings & Recommendations")
st.write("""
**Findings:**
1. Significant racial disparity in false positive rates: African-Americans have a **{:.2f}%** FPR vs **{:.2f}%** for Caucasians
2. Disparate impact ratio of **{:.2f}** (below 0.8 fairness threshold)
3. Error rate disparity: **{:.2f}%** for African-Americans vs **{:.2f}%** for Caucasians

**Remediation Steps:**
1. Implement preprocessing techniques like reweighing
2. Incorporate fairness constraints during model training
3. Establish ongoing bias monitoring with quarterly audits
4. Improve data collection to include socioeconomic context
5. Implement human review for high-risk cases
""".format(
    class_metric.false_positive_rate(False)*100,
    class_metric.false_positive_rate(True)*100,
    orig_metric.disparate_impact(),
    class_metric.error_rate(False)*100,
    class_metric.error_rate(True)*100
))

# Add footer
st.caption("""
**Note**: Based on analysis of COMPAS recidivism risk scores using AI Fairness 360 toolkit. 
This dashboard is for educational purposes only.
""")
