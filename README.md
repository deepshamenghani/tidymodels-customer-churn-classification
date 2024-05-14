# Customer Churn Classification with Tidymodels

This repository contains all the necessary files for building and evaluating customer churn classification models using the Tidymodels suite in R. The project demonstrates a structured approach to predictive modeling, focusing on the use of various Tidymodels packages to handle a common business problem: predicting customer churn.

## Project Structure

- **data/**: Directory containing the dataset used in the models.
- **bank_churn_classification.R**: R script with the complete code for the analysis, including data loading, preprocessing, model building, evaluation, and visualization.
- **tidymodels-customer-churn-classification.Rproj**: RStudio project file.

### Prerequisites

You need to have R and RStudio installed on your computer. Additionally, you need to install the following R packages:

```R
install.packages(c("tidyverse", "tidymodels", "skimr", "GGally"))

```

## Instructions

Clone this repository using Git:

[https://github.com/deepshamenghani/tidymodels-customer-churn-classification](https://github.com/deepshamenghani/tidymodels-customer-churn-classification)

## Data
[Binary Classification of Bank Churn Synthetic Data from Kaggle](https://www.kaggle.com/datasets/cybersimar08/binary-classification-of-bank-churn-synthetic-data). The dataset comprises the column "Exited" that denotes whether the customer left or not.

## Running the Analysis

Execute the script `bank_churn_classification.R` to perform the analysis from start to finish:

- **Data Loading**: Load the customer churn data.
- **Data Preprocessing**: Clean the data and prepare it for modeling.
- **Exploratory Data Analysis**: Analyze the data to understand patterns and relationships.
- **Model Building**: Build and train classification models.
- **Model Evaluation**: Evaluate the models using ROC curves and confusion matrices.
- **Feature Importance**: Analyze the importance of various features in the model.
