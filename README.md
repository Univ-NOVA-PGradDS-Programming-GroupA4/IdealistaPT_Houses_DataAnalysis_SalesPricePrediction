# Lisbon Metro Area Housing Price Prediction Model

## Overview

This repository contains all the code, data, and models used for the Lisbon Housing Price Prediction project. This project aims to build a proof of concept model to predict housing prices in the Lisbon area using machine learning techniques. The data was sourced from the Idealista API and includes various features such as location, size, number of bedrooms, number of bathrooms, floor level, and additional amenities like parking spaces.

## Repository Contents

### Notebooks
- **API_Idealista.ipynb**
  - This notebook contains the code used to access and retrieve housing data from the Idealista API. A limit of 100 queries per person was allowed, with 50 results per query.
  
- **Data_Cleaning.ipynb**
  - This notebook includes the data cleaning and preprocessing steps applied to the raw data to prepare it for analysis and modeling.

- **complete_eda.ipynb**
  - This notebook covers the Exploratory Data Analysis (EDA), where we examine the distribution and relationships of the variables in the dataset.

- **Data_Modeling.ipynb**
  - This notebook is dedicated to building, training, and evaluating the machine learning models used for predicting housing prices.

- **FastAPI_Housing.ipynb**
  - This notebook contains code for deploying the model using FastAPI, enabling the prediction service to be accessed via a web API.

- **llm_description_analysis.ipynb**
  - This notebook explores the use of large language models (LLMs) to analyze housing descriptions for quality assessment insights.

### Data
- **data_to_model2.csv**
  - The cleaned dataset used for modeling and analysis.

### Models
- **House_data_scaler.pkl**
  - Scaler object used for normalizing the features before feeding them into the machine learning models.

- **house_predictor_model.pkl**
  - Trained Random Forest Regressor model used for predicting housing prices and exported from Data_Modeling.ipynb


## Methodology

### Data Collection
- Data was collected from the Idealista API, covering housing listings in Lisbon and nearby municipalities such as Amadora, Cascais, Loures, Odivelas, Oeiras, and Vila Franca de Xira.
- The API provided a ca. 17,000 listings, which were then filtered and cleaned.

### Data Cleaning
- Duplicate entries and irrelevant columns were removed.
- Missing values were handled using various imputation techniques.
- Features were engineered to improve model performance, including merging complementary columns and encoding categorical variables.

### Exploratory Data Analysis (EDA)
- EDA was performed to understand the distribution and relationships between features.
- Key insights included the correlation between special features (e.g., video tours) and higher prices, and the identification of price variations across different municipalities.

### Machine Learning Models
- Multiple regression models were tested, including Linear Regression, K-Neighbors Regressor, Decision Tree, Random Forest Regressor, XGBRegressor, CatBoost Regressor, and AdaBoost Regressor.
- The Random Forest Regressor was selected as the best model based on RMSE and R2 scores.

### Deployment
- The final model was deployed using FastAPI to provide a RESTful API for making predictions.

## Results and Discussion
- The Random Forest Regressor model achieved an R2 score of approximately 0.75, indicating a good fit for the data.
- The project highlighted the importance of data quality and feature engineering in building effective predictive models.

## Future Work
- Expanding the dataset to include more regions and data sources.
- Incorporating additional features such as proximity to public transport and construction year.
- incorporating more data into the model training
- Exploring more advanced machine learning models and hyperparameter tuning.

## Authors
- Edgar Silva
- Luis Mourisco
- Nuno Machado
- Raquel Escalda

## Acknowledgments
Special thanks to Idealista for providing API access and documentation, and to our professors and colleagues for their support and feedback.

## References
- Idealista. (2022). Property for sale in Portugal: regions with the highest and lowest prices in 2022.
- OECD. (2024). Housing prices.
- Jha, S. B., Babiceanu, R. F., Pandey, V., & Jha, R. K. (2020). Housing Market Prediction Problem using Different Machine Learning Algorithms: A Case Study.
- Truong, Q., Nguyen, M., & Hy Dang, B. M. (2020). Housing Price Prediction via Improved Machine Learning Techniques.

## License
This project is licensed under the terms of the MIT license. See the LICENSE file for details.

---

Feel free to explore the notebooks and use the provided models for your own housing price prediction tasks. Contributions and feedback are welcome!