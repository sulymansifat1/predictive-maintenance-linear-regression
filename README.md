# Predictive Maintenance with Linear Regression

## Project Overview

This repository contains my final assignment for the **Supervised Machine Learning: Regression** course on Coursera. The project demonstrates the application of various linear regression techniques to predict the Remaining Useful Life (RUL) of aircraft engines using the NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset.

## Dataset Description

The project uses the **NASA CMAPSS FD001** dataset, which contains:

- **Training Data**: 20,631 rows across 100 engine units
- **Test Data**: 13,096 rows across 100 engine units
- **Features**: 3 operational settings + 21 sensor measurements
- **Target Variable**: Remaining Useful Life (RUL) in cycles
- **Engine Life Range**: 128 to 362 cycles (median: 199 cycles)

### Data Structure
- **Operational Settings**: setting1, setting2, setting3
- **Sensor Measurements**: s1 through s21 (various engine parameters)
- **Identifiers**: unit (engine ID), cycle (time step)
- **Target**: RUL (Remaining Useful Life)

## Methodology

### 1. Data Preprocessing
- Constructed RUL labels for both training and test datasets
- Removed zero-variance features (7 columns eliminated)
- Final feature set: 17 variables (3 settings + 14 sensors)

### 2. Model Comparison
The project evaluated multiple linear regression approaches:

| Model | Description | Key Features |
|-------|-------------|--------------|
| **OLS** | Ordinary Least Squares | Baseline linear regression |
| **Ridge** | Ridge Regression | L2 regularization |
| **Lasso** | Lasso Regression | L1 regularization with feature selection |
| **ElasticNet** | Elastic Net | Combined L1 and L2 regularization |
| **Huber** | Huber Regression | Robust to outliers |
| **Poly2+Ridge** | Polynomial + Ridge | 2nd degree polynomial features |
| **PCA+Ridge** | PCA + Ridge | Dimensionality reduction |

### 3. Cross-Validation Strategy
- **GroupKFold (5 folds)**: Ensures no data leakage between engine units
- **Evaluation Metrics**: RMSE, MAE, R²
- **Hyperparameter Tuning**: Grid search with cross-validation

## Results

### Best Performing Model
**Poly2+Ridge** with α = 3.0 achieved the best performance:

- **Test RMSE**: 46.66 cycles
- **Test MAE**: 35.31 cycles  
- **Test R²**: 0.374

### Key Findings

1. **Model Performance**: The polynomial features combined with ridge regularization provided the best balance between complexity and generalization.

2. **Prediction Quality**: The model tracks true RUL reasonably well, though prediction variance increases for larger RUL values.

3. **Residual Analysis**: Residuals are centered around zero, indicating no major systematic bias in predictions.

4. **Feature Importance**: The combination of operational settings and sensor measurements provides valuable information for RUL prediction.

## Technical Implementation

### Libraries Used
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning models and preprocessing
- **Matplotlib**: Data visualization

### Key Techniques
- **Feature Engineering**: Polynomial feature expansion
- **Regularization**: Ridge regression to prevent overfitting
- **Cross-Validation**: GroupKFold to maintain temporal integrity
- **Hyperparameter Optimization**: Grid search with cross-validation

## Limitations and Future Improvements

### Current Limitations
1. **Moderate R² Score**: 0.374 indicates room for improvement in predictive accuracy
2. **Linear Assumptions**: Linear models may not capture complex non-linear relationships
3. **Feature Engineering**: Limited to polynomial features; could explore more sophisticated transformations

### Next Steps
1. **Advanced Models**: Implement non-linear models (Random Forest, XGBoost, Neural Networks)
2. **Feature Engineering**: Explore time-series specific features, rolling statistics, and lag variables
3. **Ensemble Methods**: Combine multiple models for improved predictions
4. **Domain Knowledge**: Incorporate engineering domain expertise for feature selection
5. **Real-time Deployment**: Develop pipeline for real-time RUL prediction

## Repository Structure

```
├── finalassignment-v1.ipynb    # Main Jupyter notebook with complete analysis
├── project overview.txt        # Project requirements and guidelines
└── README.md                   # This file
```

## Related Links

- **Kaggle Notebook**: [Predictive Maintenance with Linear Regression]([https://www.kaggle.com/code/sulymansifat/predictive-maintenance-linear-regression](https://www.kaggle.com/code/sulymansifat/predictive-maintenance-linear-regression))

## Usage

1. **Prerequisites**: Install required Python packages:
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```

2. **Data**: The notebook automatically downloads the NASA CMAPSS FD001 dataset from Kaggle

3. **Execution**: Run the Jupyter notebook `finalassignment-v1.ipynb` to reproduce the analysis

## Course Context

This project was completed as part of the **Supervised Machine Learning: Regression** course on Coursera, demonstrating:
- Data preprocessing and feature engineering
- Model selection and comparison
- Cross-validation strategies
- Performance evaluation and interpretation
- Communication of analytical findings

## License

This project is for educational purposes as part of the Coursera course curriculum.

---

**Author**: Md. Sulyman Islam Sifat  
**Course**: Supervised Machine Learning: Regression (Coursera)

