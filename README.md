# Machine Learning Projects - Council of Education and Development Programmes Pvt. Ltd., Thane

[![GitHub stars](https://img.shields.io/github/stars/abubakarpungiwale/ml-portfolio?style=social)](https://github.com/abubakarpungiwale/ml-portfolio/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/abubakarpungiwale/ml-portfolio?style=social)](https://github.com/abubakarpungiwale/ml-portfolio/network)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

**This repository showcases machine learning projects developed during a NSDC-certified AI/ML & Data Science Training at Council of Education and Development Programmes Pvt. Ltd., Thane.** It demonstrates expertise in predictive modeling, data analysis, and visualization across domains like finance, text classification, and real estate, using advanced algorithms and ensemble techniques.

**Key Projects**:
- **Bank Customer Churn Prediction**: Neural network model (~85% accuracy).
- **Spam SMS Detector**: NLP-based classifier (~98% accuracy).
- **Laptop Price Analysis**: Regression and visualization (~0.13 RMSE).
- **Amazon Products Analysis**: Statistical insights via visualizations.
- **Water Quality Predictor**: Ensemble model (~0.10 RMSE).
- **House Price Predictor**: Stacked ensemble (~0.12 RMSE, top 10%).

## Table of Contents

- [Project Overview](#project-overview)
- [Key Technologies](#key-technologies)
- [Projects](#projects)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Key Technologies

- **Libraries**: NumPy, Pandas, Scikit-learn, TensorFlow, XGBoost, LightGBM, CatBoost, Matplotlib, Seaborn.
- **Models**: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, SVM, Neural Networks, Ridge, Lasso, SVR.
- **Techniques**: Stacking, RandomizedSearchCV, 5-fold CV, TF-IDF, Box-Cox, RobustScaler, One-Hot Encoding.
- **Metrics**: Accuracy, F1-Score, ROC-AUC, RMSE, MAE.

## Projects

1. **Bank Customer Churn**: Deep learning with TensorFlow (~85% accuracy, 0.82 ROC-AUC).
2. **Spam SMS Detector**: TF-IDF + ensemble (~98% accuracy, 0.95 F1-Score).
3. **Laptop Price Analysis**: Regression + visualization (~0.13 RMSE).
4. **Amazon Products Analysis**: Statistical tests and visualizations.
5. **Water Quality Predictor**: Ensemble model (~0.10 RMSE).
6. **House Price Predictor**: Stacked ensemble (~0.115 CV RMSE, ~0.12 test RMSE).

## Installation

```bash
git clone https://github.com/abubakarpungiwale/ml-portfolio.git
cd ml-portfolio
pip install -r requirements.txt
```

## Methodology

- **Preprocessing**: EDA, imputation (median/mode), outlier removal, Box-Cox, RobustScaler, One-Hot/TF-IDF encoding.
- **Feature Engineering**: Aggregated features (e.g., `TotalArea`), binary indicators.
- **Training**: Hyperparameter tuning via RandomizedSearchCV, 5-fold CV.
- **Ensemble**: Stacking with Ridge/Linear meta-learners for robust predictions.

## Performance Metrics

- **Churn Prediction**: Training: 87% accuracy, 0.84 ROC-AUC; Test: 85% accuracy, 0.82 ROC-AUC.
- **Spam Detector**: Training: 99% accuracy, 0.96 F1; Test: 98% accuracy, 0.95 F1.
- **Laptop Price**: Training: 0.12 RMSE; Test: 0.13 RMSE.
- **Water Quality**: Training: 0.09 RMSE; Test: 0.10 RMSE.
- **House Price**: Training: 0.115 RMSE; Test: ~0.12 RMSE (top 10%).
- **Insights**: Ensembles improved performance by 5-10%. Visualizations in notebooks enhance interpretability.

## Contributing

Fork and submit pull requests for enhancements.

## License

MIT License - see [LICENSE](LICENSE).

## Contact

- **Author**: Abubakar Maulani Pungiwale
- **Email**: abubakarp496@gmail.com
- **LinkedIn**: [linkedin.com/in/abubakarpungiwale](https://linkedin.com/in/abubakarpungiwale)
- **Contact**: +91 9321782858

Connect for ML project discussions or data science opportunities!

---
