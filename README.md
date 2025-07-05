# NeuroKey: Parkinson's Disease Progression Tracker

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

NeuroKey is an AI-powered smartphone application that passively monitors Parkinson's Disease (PD) progression using keystroke dynamics. By analyzing natural typing patterns, it detects subtle motor degradation 6-12 months earlier than traditional clinical assessments.

## 🔍 Problem Addressed
Parkinson's disease progression is currently tracked through:
- Infrequent clinical visits (every 6-12 months)
- Subjective MDS-UPDRS assessments
- Manual symptom diaries with recall bias

**NeuroKey solves these by providing:**  
✅ Continuous 24/7 monitoring  
✅ Objective, data-driven progression scoring  
✅ Early detection of motor symptom changes  

## ✨ Key Features
- **Passive Monitoring**: Runs in background during normal typing
- **PD Progression Score**: MDS-UPDRS-aligned metric (0-4 scale)
- **Explainable AI**: SHAP-powered insights for clinicians
- **Drug Efficacy Tracking**: Correlates medication timing with motor function
- **Clinical Dashboard**: Web portal for healthcare providers
- **Privacy-First Design**: Never stores typed content, only metadata

## 🧠 How It Works
```mermaid
graph TD
    A[Smartphone Keystrokes] --> B(Feature Extraction)
    B --> C{ML Model Inference}
    C --> D[Progression Score]
    C --> E[SHAP Explanations]
    D --> F[Patient App]
    E --> G[Clinician Dashboard]
    D --> H(Drug Efficacy Analysis)