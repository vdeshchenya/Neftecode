# Neftecode Hackathon: Predicting Multi-Component Oil Viscosity Using Machine Learning Methods

<p align="center">
  <img width="1680" alt="Logo" src="https://github.com/user-attachments/assets/1d899006-3f66-4a4b-9914-3d9638adaef2" />
</p>

> This repository contains our solution for the **Neftecode Hackathon**, held in April 2024 by **ITMO University** in collaboration with the industrial partner interested in predictive modeling of lubricant properties. The main goal was to evaluate the applicability of modern machine learning methods for predicting the viscosity of oil mixtures based on an encrypted dataset.

## 📊 Hackathon Dataset Overview

<p align="center">
  <img width="1287" alt="Example" src="https://github.com/user-attachments/assets/79c90273-c189-477e-8c5a-70731ed1c1c5" />
</p>

The provided **encrypted** dataset describes various **oil blends** and includes:

- **Oil type**: Categorical identifier for the type of oil.
- **Oil properties**: Physical and chemical properties of the oil (e.g., density at different temperatures, viscosity of base oil, additives, ions compositions, and others).
- **Component classes**: Types of additives present in the oil.
- **Component properties**: Physical properties of each component (e.g., pour point, demulsification time, separated water volume, and others)
- **SMILES strings**: Text-based representations of the molecular structures of components.

The **target variable** is viscosity, measured by the industrial partner using the [standard D445-24](https://cdn.standards.iteh.ai/samples/117746/bf9f10325e2746a2884eb1df023eea82/ASTM-D445-24.pdf). A log-transformed distribution of the target is shown below.

<p align="center">
  <img width="800" alt="Hists" src="https://github.com/user-attachments/assets/70825667-421d-4957-9868-6aa0146c99d1" />
</p>

## 🎯 Hackathon Objectives

1. **Data Analysis and Literature Review**: Perform exploratory data analysis and review relevant publications to understand the domain and dataset.
2. **SMILES Conversion**: Convert SMILES strings to machine-readable embeddings using methods such as:
   - Transformers
   - Graph Neural Networks (GNNs)
   - Quantum chemical descriptors
3. **Model Development - Part 1**: Develop a model to predict viscosity using encrypted physical and chemical data, capable of handling missing values.
4. **Model Development - Part 2**: Build a model to predict viscosity from SMILES embeddings, accommodating varying numbers of SMILES per sample.
5. **Pipeline Integration**: Combine both models into a unified pipeline, perform hyperparameter optimization, and evaluate performance using standard regression metrics.

## 💡 Our Solution

Given the dataset's limited size (340 samples), we focused on **tree-based models**, known for their effectiveness with small datasets and interpretability—an essential factor for industrial applications.

### Models Evaluated

- **Decision Tree (DT)**
- **Random Forest (RF)**
- **Gradient Boosting (GB)**

### Training Strategy

- **Hyperparameter Tuning**: Utilized `GridSearchCV` from `scikit-learn` for systematic hyperparameter optimization.
- **Cross-Validation**: Employed 5-fold cross-validation to ensure model robustness.
- **Evaluation Metrics**: Assessed models using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- **Target Transformation**: Applied logarithmic transformation to the target variable (viscosity), which consistently improved model performance across all architectures.

### 🔍 Results

The **Gradient Boosting (GB)** model demonstrated superior performance, achieving the lowest MAE and RMSE. Consequently, it was selected as the final model for prediction tasks.

## 🚀 How to Run

### Repository Structure

```
.
├── dataset/             # Dataset files
├── model/               # Trained model
├── results/             # Prediction outputs
├── Predict.ipynb        # Inference pipeline
└── Train.ipynb          # Training pipeline
```

### Running the Notebooks

1. **Training the Model**:
   - Open `Train.ipynb` in Jupyter Notebook or JupyterLab.
   - Execute the cells sequentially to preprocess data, train the model, and evaluate performance.
   - The trained model will be saved in the `model/` directory.

2. **Making Predictions**:
   - Open `Predict.ipynb`.
   - Ensure the trained model is available in the `model/` directory.
   - Execute the cells to load the model and make predictions on new data.
