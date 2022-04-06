# Hamilton House Price Prediction: Spatial Regression vs. Machine Learning
* Created a tool that estimates house prices (**MSE ~ $57070.99, R^2 ~ 0.805**) to help predict house prices by census tract in the city of Hamilton, ON.
* Created a spatial regression tool that estimates house prices (**R^2 ~ 0.8548, MSE ~ 0.02 on Log transformed data**) based on the location of census tracts in Hamilton, ON.
* Compared spatial regression modelling to non-spatial machine learning modelling to conclude that **spatial modelling provides stronger models** for house price prediction than non-spatial machine learning models as **geograhic location influences house prices**.

# Data & Problem
* **Problem:** The goal of this problem is to compare the prediction accuracy of machine learning regression algorithms to spatial regression algorithms on house pricing data. This is because we wish to see if developing a spatial regression model for house prices would provide more accurate predictions because there appears to be a relationship between location/geography and house prices. In a statement,
> Will the Spatial Regression model provide better predictions than the Mahcine Learning Model for house price prediction?
* **Data Acquisition:** 
  1. For features - https://datacentre.chass.utoronto.ca/ 
  2. Geographic Data + Target Variable (House Prices): https://raw.githubusercontent.com/gisUTM/GGR376/master/Lab_1/houseValues.geojson
* **Success Metrics:** Whichever of the two modelling techniques (Spatial Regression or Machine Learning) achieves a lower MSE score with a higher R^2 value, we will propose that one technique is ***potentially*** superior for house price predictions over the other. 

# Code & Resources Used
* **Python Version:** 3.8
* **Environment:** Miniconda, Jupyter Notebook
* **Packages:** Pandas/GeoPandas, Scikit-Learn, PySal, NumPy, Matplotlib, Seaborn, Joblib

# EDA & Feature Engineering
After loading the datasets, merging the two datasets, and inspecting the merged dataframe's features (housing attributes) and target variable (house prices), I needed to clean the raw data and visualize it to better understand the relationship between the features, and the target. I did the following steps with the datasets:
* Founder the number of unique values in the dataset through creating a dictionary. 
* Seperated features and target columns to acquire summary statistics on the continuous features and target variable.
* Checked for total number of missing values in each column and found that there are **no missing values** in this dataset.
* Checked for duplicate values in the dataset and found **no duplicate values** in the dataset.
* Created univariate data visualizations of all the features through plotting:
  - Histograms:
  - Density Plots:
  - Box Plots:
* Developed bivariate data visualizations comparing all features to each other including the target variable (houseValues):
  - Scatter Plot Matrix:
  - Heatmap:
* The correlation matrix confirms that features priv_dwellings_by_bedroom, priv_dwellings_byrooms and house_by_person_per_room are postively correlated with a 1.0 correlation coefficient. Two of these features will be removed and only one of them will be kept for analysis. Since all features have the same correlation coefficient with the target variable houseValue, keeping the priv_dwellings_byrooms feature makes the most sense as it represents the occupied private homes by the number of rooms per household.
* Dropped correlated features to prevent multicollinearity. 
* Explored target variable through:
  - Density Plot:
  - Box Plot:
  - Removed Outliers through IQR Inspection:
* Concluded EDA by developing choropleth maps of the spatial data through mapping average house prices per census tract, and average after-tax per census tract:

# Model Building (Machine Learning)
First, I seperated the spatial features from the non-spatial features by creating a non-spatial dataset containing the house attributes, and the target variable, house prices. Next, I split the data into X (features) and y (target). Then, I split the X and y data into training and test sets with test set size of 20%. Based on the [Scikit-Learn Algorithm Selection Sheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html), I trained 10 regression algorithms and evaluated them using **R^2** as the primary evaluation metric because it is important to interpret how well the models predict house prices with the given features.

The 10 models I selected for the regression modelling:
1. LinearRegression()
2. Lasso()
3. ElasticNet()
4. DecisionTreeRegressor()
5. KNeighborsRegressor()
6. SVR()
7. AdaBoostRegressor()
8. GradientBoostingRegressor()
9. RandomForestRegressor()
10. ExtraTreeRegressor()

# Model Perfomance (Machine Learning)
From the baseline models, the Linear Regression, Lasso Regression, Random Forest Regressor, and Extra Trees Regressor models achieved **R^2 values above 0.7** on the test sets. However, data standardization will be required to get more accurate predictions out of the models as many of the features are not on the same scale.

After **standardizing the features**, significant improvements can be seen in the Elastic Net model while Random Forest Regressor, and Extra Trees Regressor models saw slight improvements. The other models remained unaffected, and SVR performed extremely poorly, so this model will be discarded entirely. 

The KNN, Lasso, Elastic Net, Random Forest, Extra Trees models had their hyperparameters tuned by hand and **RandomizedSearchCV**. The Elastic Net model achieved an R^2 value of ~0.81 and Extra Trees models achieved an R^2 of ~0.79. 

The Elastic Net and Extra Trees models had additional hyperparameters tuned with **GrdiSearchCV** to get the most out of these models and achieve the best possible model accuracy. The best hyperparameters were retained, and we concluded that **Elastic Net provided the best model** with a high R^2 value of ~0.81. The cross-validated metrics were obtained through computing the mean of the model's **RMSE** and **MAE** with 5-folds cross-validation. 

# Feature Importance
Which features contribute most to a model predicting Hamilton's house prices?
* monthly_housing_costs and income_after_tax are by far the most important features for determining/predicting Hamilton's house prices. This is likely because people living in larger houses tend to earn more and have more monthly costs. Hence, the house prices are likely to increase or decrease based on the income and monthly housing costs across the city's census tracts (CTUID feature from original dataframe).

* percent_mortage and avg_house_size are the least important features for determining/predicting Hamilton's house prices. Their low coefficient values are likely due to the fact that houses throughout the city have no relationship between how many people are living in them, and their price. Across the census tracts that comprise of the city, there is a large variation between the prices, and the number of people living in the home. Some CTUIDs (census tracts) have lower priced homes with more people living in them while others have less people living in the home, but the house's price is quite high. In addition, mortgage is practically irrelevant to the predicting a house's price as this feature relates more to an individual's income. We cannot make good predictions on a house price given the percentage of individuals with mortgages in a CTUID (census tracts).



# Spatial Regression Models
Since we have completed an end-to-end machine learning project on house price prediction for Hamilton, ON, it is now time to dive deeper into the potential insights that a spatial regression model would provide over the machine learning models developed earlier.

For the spatial regression modelling, we developed a **baseline OLS model** using PySAL without a spatial weights matrix, which would emphasize a connection between an observation and its n-nearest neighbors.

The model provided a **high R^2 value of ~0.84 (greater than machine learning models)**, and the model was also **statistically significant** based on the p-value of the F-Statistic. 

Developed a spatially lagged exogenous regression model, by spatially lagging the explanatory variable that influences the price of a home at a given location the most. From the baseline OLS model, it was apparent that monthly_housing_costs influences the price of the home most based on the coefficients of the OLS model. 

The spatially lagged model obtained a **slightly higher R^2 (~0.86)** than the baseline OLS model through lagging the monthly housing costs feature. Furthermore, the model was also **statistically significant** based on the p-value of the F-Statistic. 

Generally speaking, house prices are **drived heavily by location**, which is a spatial attribute.

The use of spatial models is likely going to improve prediction accuracy. We will compute the **MSE** as the evaluation metric for the spatial regression models just like the machine learning models to compare their results.

This will explicity determine if spatial models are better than non-spatial machine learning models for house price predictions where spatial attributes play a role in determining house prices.

We can see **no improvement in the MSE** of the spatial OLS model by lagging the best feature and adding a weight matrix, but when we compare the R^2 and adjuated of the spatially lagged OLS model to the OLS without weights or lag, we can see a slightly improvement in model performance.

# Conclusions from Model Comparisons

* The Spatially Lagged OLS model achieved a **higher R^2 (0.85)** than the both the cross-validated Elastic Net model, and the tuned Elastic Net. Although the tuned Elastic Net model achieved an **R^2 value of 0.8**, this was only acquired through exhaustive hyperparameter tuning, and compared with other non-spatial machine learning models. The spatially lagged OLS achieved a 0.85 R^2 with just the addition of a spatially lagged feature (the feature that influenced the house prices the most).
* The inclusion of **spatially lagged features provides a more accurate model** for house price prediction when space plays a role.
* Spatial models can provide better predictions than non-spatial machine learning models for house price prediction because house prices are heavily dependent upon location.
* Therefore, for house price prediction problems, it is a good idea to include **spatial regression models** in addition to the non-spatial machine learning models for prediction. This is because location/geography plays a big role in influencing the prices of homes. Hence, the prices of Hamilton houses across census tracts are **spatially dependent**.

**Limitations in Results:** 
* The dataset is unfortunately not large enough to reach solid conclusions with accurate predictions. More housing data will be required as well as far more features to truly develop accurate models.
* In addition, the spatial OLS model was developed using log house prices (target variable) while the machine learning models were not trained and tested on log house prices (the target not transformed using log). This means that we were unable to compare the RMSE, MSE or MAE metrics of the models as they were computed on different scales. Training the machine learning models with log house prices would be the next step to see which models provided lower RMSE and MAE values.
