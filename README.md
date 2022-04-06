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

# Feature Importance

# Spatial Regression Models

# Conclusions
