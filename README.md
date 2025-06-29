# ocean-plastic-contributor-analysis
River Plastic Classifier

This project uses supervised machine learning to classify rivers based on their plastic contribution to the ocean. It leverages real-world data and a Random Forest model to identify which rivers are high-risk contributors to marine plastic pollution.

Project Structure

data/ : Contains the dataset CSV file

notebooks/ : Jupyter notebooks for EDA and modeling

src/ : Python scripts for preprocessing and training

models/ : Saved trained models

reports/ : Evaluation reports and visualizations

README.md : Project overview and instructions

requirements.txt: Python dependencies

.gitignore : Files to exclude from version control

Dataset

Source: data/global_riverine_plastic_emissions_into_ocean.csv

Target Variable: plastic_contribution (1 = high contributor, 0 = low contributor)

Threshold: Rivers emitting more than 6008 metric tons/year are labeled as low priority (0), others as high priority (1)

Model

Algorithm: Random Forest Classifier

Features: Scaled numerical predictors from the dataset

Evaluation Metrics: Accuracy, Precision, Recall, and Feature Importance
