# Player Churn Prediction System

This repository contains an end-to-end data mining and machine learning project focused on predicting player churn for online gaming platforms. The project pipeline includes data preprocessing, feature engineering, and evaluating multiple classification models, specifically Logistic Regression and Decision Tree models.

It features an interactive web application built with Streamlit that allows users to upload gaming behavior datasets, dynamically process the data, and predict the churn probability and risk level for individual players.

## Repository Structure

The project repository follows the structure shown below:

```text
Churn Prediction/
|-- data/
|   |-- online_gaming_behavior_dataset.csv
|-- models/
|   |-- decision_tree.pkl
|   |-- feature_names.pkl
|   |-- label_encoders.pkl
|   |-- logistic_regression.pkl
|   |-- scalers.pkl
|-- notebooks/
|-- src/
|   |-- download_data.py
|   |-- preprocess.py
|   |-- train.py
|-- app.py
|-- report.tex
|-- requirements.txt
```

## Description of Key Components

*   **data/**: Contains the raw dataset `online_gaming_behavior_dataset.csv`.
*   **models/**: Trained models and serialized preprocessing objects like scalers and encoders.
*   **notebooks/**: Jupyter Notebooks for preprocessing, model training, and evaluation.
*   **src/**: Python source code for data downloading, preprocessing, and model training.
*   **app.py**: Main application script for the Streamlit web interface.
*   **report.tex**: LaTeX source code for the project report in IEEE format.
*   **requirements.txt**: Exact Python dependencies required to run the application.

## Getting Started

1.  **Clone the Repository** and navigate to the project root directory.

2.  **Install Dependencies**: Install the necessary Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit Application**: Launch the interactive web app to test predictions:
    ```bash
    streamlit run app.py
    ```

## Features

*   Upload raw CSV data directly via the UI.
*   Automatic handling of missing values and categorical encoding.
*   Model selection toggle (Logistic Regression vs. Decision Tree).
*   Dynamic risk assessment table (Low, Medium, High risk) sorted with corresponding probabilities.
*   Visual distribution of user risk profiles using Seaborn.
*   Feature importance visualizer for the Decision Tree model.
*   Downloadable result datasets.
