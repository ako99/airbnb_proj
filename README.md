# Machine Learning Price Estimator: NYC Airbnb Listings
* Cleaned and administered feature engineering on public NYC Airbnb dataset
* Performed in-depth exploratory data analysis on the ~50,000 data entries and 20 different features
* Utilized and optimized Linear Regression, Lasso Regression, Ridge Regression, and Random Forests machine learning models to develop the best model

## Resources Used
* Public NYC Airbnb dataset (https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)
* Python 3.7
* Jupyter Notebook
* Packages: pandas, numpy, sklearn, matplotlib, seaborn

## Data Cleaning
The public dataset was 10/10 in terms of usability. Very little had to be changed and some features were added based on the text description in the "name" column.
* Used the dates in "last_review" to create a "days_since_last_review" feature (based on the most recent date in the dataset)
* Made a column for whether or not the listing has been reviewed before to supplement "days_since_last_review"
* Made columns for whether or not a description advertises its proximity to a landmark:
  * Airport
  * Stadium
  * Mall
  * Subway Accessibility
* Compounded the length of descriptions into its own column
* Performed a log transformation on the price distribution to normalize for better statistical analysis

## EDA
data_eda.ipynb takes a look at the distributions of all the features extacted from the dataset as well as their multicollinearity, relationship to price, and relationship to location.
Here are a few highlights.

**Average Price vs Room Type per Borough**

![alt text](https://github.com/ako99/airbnb_proj/blob/master/Images/Average%20Price%20vs%20Room%20Type%20per%20Borough.png)

**Correlation Matrix**

![alt text](https://github.com/ako99/airbnb_proj/blob/master/Images/corrmat.png)

**Room Type by Location**

![alt text](https://github.com/ako99/airbnb_proj/blob/master/Images/Roomtype%20by%20coordinate.png)

**Wordcloud of Listing Names**

![alt text](https://github.com/ako99/airbnb_proj/blob/master/Images/Wordcloud.png)

## Model Building
First, based on the conclusions made from the EDA, insignificant features were excluded from model building. 
A pipeline was then made where empty values would be imputed with average values and categorical features would be transformed via One Hot Encoding.
The data was split into train and test sets (test size of 30%).
To initially judge the effectiveness of different models, I first evaluated their Mean Absolute Error and R^2 values when fit onto the dataset with default hyperparameters.
The following regression models were used:
* Linear Regression
* Lasso Regression
* Ridge Regression
* Random Forests

After initial evaluation, Grid Search CV was used to tune the hyperparameters to best suit the project.

## Model Performanace
With optimized hyperparameters, Random Forest outperformed the other models by a sizable margin.
* **Random Forest**: MAE = 0.311
* **Multiple Linear Regression**: MAE = 0.337
* **Lasso Regression**: MAE = 0.351
* **Random Forest**: MAE = 0.337

![alt text](https://github.com/ako99/airbnb_proj/blob/master/Images/MAE%20plot.png)
