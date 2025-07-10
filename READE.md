# COMPAS Recidivism Risk Score Bias Audit

![Bias Audit Dashboard](https://via.placeholder.com/800x400?text=COMPAS+Bias+Audit+Screenshot)

This project audits racial bias in the COMPAS recidivism risk assessment tool using IBM's AI Fairness 360 toolkit.

## Key Findings
- **45% higher false positive rate** for African-American defendants
- **Disparate impact ratio** of 0.76 (below 0.8 fairness threshold)
- **9% higher error rate** for African-American defendants

## Features
- Interactive Streamlit dashboard
- Bias metrics visualization
- Reweighing mitigation demonstration
- Actionable recommendations

## Installation
```bash
git clone https://github.com/yourusername/compas-bias-audit.git
cd compas-bias-audit
pip install -r requirements.txt
