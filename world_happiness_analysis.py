#!/usr/bin/env python
# coding: utf-8


# **Name:** Ashton Reyes
# 
# This project will explore what makes people in different countries happy using the World Happiness Report data from 2015-2019. Data will be cleaned a merged and exploratory data analysis will be performed to understand key factors and trends that are related to happiness. Machine learning models will be incorporated to predict 2019 happiness rankings and design a formula that outputs a happiness score.

# ## Notebook Outline
# 
# 1. Setup 
# 2. Data loading 
# 3. Data cleaning and merging  
# 4. Exploratory data analysis (EDA)  
# 5. Modeling
# 6. Personal happiness formula and conclusions

# ## 1. Setup 

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Personal settings
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="whitegrid")
pd.set_option("display.max_columns", 50)
pd.set_option("display.precision", 3)


# ## 2. Data loading 

# In[2]:


# Loading each CSV into its own DataFrame
df2015 = pd.read_csv("2015.csv")  
df2016 = pd.read_csv("2016.csv")   
df2017 = pd.read_csv("2017.csv")
df2018 = pd.read_csv("2018.csv")   
df2019 = pd.read_csv("2019.csv")  


# In[3]:


# Charting a DataFrame for basic understanding of data
df2015.head()


# There are now five separate pandas DataFrames:
# 
# - `df2015` – World Happiness Report data for 2015  
# - `df2016` – World Happiness Report data for 2016  
# - `df2017` – World Happiness Report data for 2017  
# - `df2019` – World Happiness Report data for 2019  
# 
# Each DataFrame contains a rows with a specific country and columns such as the respective countries happiness score, rank, GDP per captia, etc.

# ## 3. Data cleaning and merging

# After charting the DataFrames individually, I noticed there are some differences within each of the DataFrames. I will first inspect the column names and try to understand the basic strucutre of each DataFrame. I want to identify the common variables and how their names differ before moving on.

# In[4]:


# Inspect column names for each year
print("2015 columns:\n", df2015.columns, "\n")
print("2016 columns:\n", df2016.columns, "\n")
print("2017 columns:\n", df2017.columns, "\n")
print("2018 columns:\n", df2018.columns, "\n")
print("2019 columns:\n", df2019.columns, "\n")


# From the column names I've printed, I can see that there are common variables (rank, score, GDP, family/social support, health, freedom, generosity, and corruption) that appear in all years (with some having a variation of the name). I need to note the variation in names such as 'Happiness Rank', 'Happiness.Rank', or 'Overall rank'. This applies to the GDP column as well. For the sake of analysis and convenience  I will be renmaing the columns to a common name and add a 'Year' column label to each DataFrame.

# ### 3.1 Standardizing column names and adding an additional Year column

# This will just be tedious modification of DataFrames to make analaysis easier.

# In[5]:


# 2015 DataFrame
df2015["Year"] = 2015

df2015 = df2015.rename(columns={
    "Country": "Country",
    "Region": "Region",
    "Happiness Rank": "Rank",
    "Happiness Score": "Score",
    "Economy (GDP per Capita)": "GDP",
    "Family": "Social_support", # family support
    "Health (Life Expectancy)": "Health",
    "Freedom": "Freedom",
    "Trust (Government Corruption)": "Corruption",
    "Generosity": "Generosity",
    "Dystopia Residual": "Dystopia_residual" # n2s (note to self): keeping but may drop later
})

# 2016 DataFrame
df2016["Year"] = 2016

df2016 = df2016.rename(columns={
    "Country": "Country",
    "Region": "Region",
    "Happiness Rank": "Rank",
    "Happiness Score": "Score",
    "Economy (GDP per Capita)": "GDP",
    "Family": "Social_support",
    "Health (Life Expectancy)": "Health",
    "Freedom": "Freedom",
    "Trust (Government Corruption)": "Corruption",
    "Generosity": "Generosity",
    "Dystopia Residual": "Dystopia_residual"
})

# 2017 DataFrame
df2017["Year"] = 2017

df2017 = df2017.rename(columns={
    "Country": "Country",
    "Happiness.Rank": "Rank",
    "Happiness.Score": "Score",
    "Economy..GDP.per.Capita.": "GDP",
    "Family": "Social_support",
    "Health..Life.Expectancy.": "Health",
    "Freedom": "Freedom",
    "Trust..Government.Corruption.": "Corruption",
    "Generosity": "Generosity",
    "Dystopia.Residual": "Dystopia_residual"
})

# 2018 DataFrame
df2018["Year"] = 2018

df2018 = df2018.rename(columns={
    "Country or region": "Country",
    "Overall rank": "Rank",
    "Score": "Score",
    "GDP per capita": "GDP",
    "Social support": "Social_support",
    "Healthy life expectancy": "Health",
    "Freedom to make life choices": "Freedom",
    "Generosity": "Generosity",
    "Perceptions of corruption": "Corruption"
    # n2s: 2018 does not have Region in this file.
})

# 2019 DataFrame
df2019["Year"] = 2019

df2019 = df2019.rename(columns={
    "Country or region": "Country",
    "Overall rank": "Rank",
    "Score": "Score",
    "GDP per capita": "GDP",
    "Social support": "Social_support",
    "Healthy life expectancy": "Health",
    "Freedom to make life choices": "Freedom",
    "Generosity": "Generosity",
    "Perceptions of corruption": "Corruption"
})


# In[6]:


# Check standardized columns for a couple of years
df2015.columns, df2018.columns, df2019.columns


# ### 3.2 Merging the DataFrames into a single one

# In[7]:


# Columns that I will use for analysis
common_cols = [
    "Country",
    "Year",
    "Rank",
    "Score",
    "GDP",
    "Social_support",
    "Health",
    "Freedom",
    "Generosity",
    "Corruption",
    "Region" # NaN for 2018
]
# .reindex keeps only the columns in common_cols and add missing ones for NaN
df2015_clean = df2015.reindex(columns=common_cols)
df2016_clean = df2016.reindex(columns=common_cols)
df2017_clean = df2017.reindex(columns=common_cols)
df2018_clean = df2018.reindex(columns=common_cols)
df2019_clean = df2019.reindex(columns=common_cols)

# Merging
df_all = pd.concat(
    [df2015_clean, df2016_clean, df2017_clean, df2018_clean, df2019_clean],
    ignore_index=True
)

# Charting the new DataFrame
df_all.head(), df_all.shape


# The `df_all` DataFrame is the merged DataFrame with a single table containing 782 rows and 11 columns. Each row represents one country with a year and includes my standardized variables.

# ## 4. Exploratory data analysis (EDA)

# ### 4.1 Central tendency of happiness scores over time

# In[8]:


# Compute mean and median happiness score for each year
score_summary = df_all.groupby("Year")["Score"].agg(["mean", "median", "min", "max", "count"])
score_summary


# The table above summarizes the distribution of the happiness score by year. Just from a quick analysis, it seems the data by year are very similar with small deviations.

# In[9]:


# Line plot of average happiness score by year for visualization
avg_score_by_year = df_all.groupby("Year")["Score"].mean().reset_index()

plt.figure(figsize=(6, 4))
sns.lineplot(data=avg_score_by_year, x="Year", y="Score", marker="o")
plt.title("Average World Happiness Score by Year (2015–2019)")
plt.ylabel("Average happiness score")
plt.xlabel("Year")
plt.ylim(avg_score_by_year["Score"].min() - 0.1, avg_score_by_year["Score"].max() + 0.1)
plt.tight_layout()
plt.show()


# From 2015 to 2019, the graph helps us see that the average world happiness score remained fairly stable, fluctuating between around 5.35 and 5.41. There is a small decrease in 2017, but by 2019 the mean score (approximately 5.41) is slightly higher than in 2015 (approximately 5.38), suggesting a modest overall increase in reported happiness over the period.

# ### 4.2 Checking for missing values

# In[10]:


# Overview 
df_all.info()


# In[11]:


# Count of missing values in each column
df_all.isna().sum()


# The 'df_all.info()' summary shows that all the main variables that I will be using in the analysis are stored as numeric types. The missing values counts show that 'Region' is only available for a subset of observations (315/782 rows), and there is a missing value in 'Corruption'. Other than those two variables, there are no missing values, therefore, no additional cleaning would be required.

# In[12]:


# Drop the single row with missing Corruption value
df_all = df_all[df_all["Corruption"].notna()].reset_index(drop=True)

# Verify that there are no more missing Corruption values
df_all.isna().sum()


# Because there is only a single missing value in `Corruption` (out of 782 rows), I remove that one row from the combined dataset. This simplifies later correlation analysis and modeling without materially affecting the results.

# ### 4.3 Country-level rank stability and improvement analysis

# I will now examine how countries' happiness rankings change over time. I will focus on countries that appear in all five years (2015–2019), compute a measure of rank stability, and identify which countries improved their rankings the most between 2015 and 2019.

# In[13]:


# Find countries that appear in all five years 
countries_per_year = df_all.groupby("Country")["Year"].nunique()
countries_all_years = countries_per_year[countries_per_year == 5].index

# Check variable counts
len(countries_all_years), list(countries_all_years)[:10]  


# In[14]:


# Keep rows with the countries in countries_all_years
df_panel = df_all[df_all["Country"].isin(countries_all_years)]

# Assigns rank stats and make it a DataFrame
rank_stats = (
    df_panel
    .pivot_table(index="Country", columns="Year", values="Rank")
    .rename(columns={
        2015: "Rank_2015",
        2016: "Rank_2016",
        2017: "Rank_2017",
        2018: "Rank_2018",
        2019: "Rank_2019"
    })
)

rank_stats["rank_std"] = df_panel.groupby("Country")["Rank"].std()
rank_stats["rank_change"] = rank_stats["Rank_2015"] - rank_stats["Rank_2019"]

# Chart
rank_stats.head()


# In[15]:


# Table with the most stable rankings
most_stable = rank_stats.sort_values("rank_std").head(10)
most_stable


# In[16]:


# Table with the most improved rankings
most_improved = rank_stats.sort_values("rank_change", ascending=False).head(10)
most_improved


# Among countries that appear in all five years, the most stable rankings (smallest `rank_std`) are found for countries such as New Zealand, Australia, Iceland, Denmark, and the Netherlands. Their happiness ranks deviate very little from 2015 to 2019, showing consistently high positions in the global ranking.
# 
# The `rank_change` column shows that countries such as Benin, Ivory Coast, Honduras, Hungary, and Gabon improved their rankings the most between 2015 and 2019. Here I define improvement as a decrease in the numerical rank (which means it is getting closer to 1), which I capture as `Rank_2015 – Rank_2019`, so larger positive values indicate larger gains in the happiness ranking.

# ### 4.4 Relationships between happiness and key predictors

# I will now visualize how the happiness score relates to other variables in the dataset, such as the GDP per capita, social support, health (life expectancy), freedom, generosity, and perceptions of corruption. I will start with visualization of the data by using scatterplots of the happiness score against certain features.

# In[17]:


# Scatter plot with happiness score vs GDP
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df_all, x="GDP", y="Score", alpha=0.6)
plt.title("Happiness score vs GDP per capita")
plt.xlabel("GDP per capita (log-scale units)")
plt.ylabel("Happiness score")
plt.tight_layout()
plt.show()


# In[18]:


# Scatterplot with happiness score vs social support
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df_all, x="Social_support", y="Score", alpha=0.6)
plt.title("Happiness score vs social support")
plt.xlabel("Social support")
plt.ylabel("Happiness score")
plt.tight_layout()
plt.show()


# In[19]:


# Scatterplot with happiness score vs health
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df_all, x="Health", y="Score", alpha=0.6)
plt.title("Happiness score vs health (life expectancy)")
plt.xlabel("Health (life expectancy)")
plt.ylabel("Happiness score")
plt.tight_layout()
plt.show()


# In[20]:


# Correlation matrix for the numeric variables
numeric_cols = ["Score", "GDP", "Social_support", "Health",
                "Freedom", "Generosity", "Corruption"]

corr_matrix = df_all[numeric_cols].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
plt.title("Correlation between happiness score and predictors")
plt.tight_layout()
plt.show()


# After analyzing all the scatterplots and the correlation matrix, there is a strong indication that the happiness score are positively associated with GDP per capita, social support, and life expectancy, with correlation coefficients around 0.79, 0.75, and 0.70 respectively. Freedom also shows a moderate positive correlation with happiness (approx. 0.5), while generosity and corruption having much weaker relationships with the happiness scores (approx. 0.16 and approx. 0.43). I can infer that economic prosperity, strong social networks, and good health outcomes are the main numeric predictors of higher happiness scores in this dataset, whereas generosity and corruption have less of an impact.

# ## 5. Modeling

# To model, I will use data from 2015-2018 to train on machine learning models and use the 2019 as the test set to predict the happiness scores and rankings. I will compare the model-based rankings with the 2019 dataset to evaluate which model is performing more optimally.

# In[21]:


# Creating DataFrames for training data and test data
train_df = df_all[df_all["Year"] < 2019].copy() # Training data (2015-2018)
test_df  = df_all[df_all["Year"] == 2019].copy() # Test data (2019)

train_df["Year"].unique(), test_df["Year"].unique()


# In[22]:


# Choosing features and targets.
feature_cols = ["GDP", "Social_support", "Health", "Freedom", "Generosity", "Corruption"]
target_col = "Score"

X_train = train_df[feature_cols] # passed into the trained model to get predictions
y_train = train_df[target_col] # used for fitting the model (2015-2018)

X_test = test_df[feature_cols] # used for fitting the model
y_test = test_df[target_col] # only for evaluation (2019)


# ### 5.1 Linear regression (Model 1)

# Multiple linear regression will be used as my baseline model to predict the happiness score from GFP per capita, social support, health, freedom, generosity, and corription. This model will assume a linear relationship between my chosen predictors and the target. It will fit coefficients that descript how much the expected happiness score changes when each feature increases by one unit, keeping the others constant. This model is simple and provides an interpretable benchmark allowing for more adaptable learning models.

# In[23]:


# Libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Initialize and fit the linear regression model on the training data
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# Predict happiness scores
y_pred_linreg = linreg.predict(X_test)

# Evaluate prediction accuracy on test data scores
mse_linreg = mean_squared_error(y_test, y_pred_linreg)
mae_linreg = mean_absolute_error(y_test, y_pred_linreg)
r2_linreg = r2_score(y_test, y_pred_linreg)

mse_linreg, mae_linreg, r2_linreg


# In[24]:


# Build a DataFrame with true and predicted scores for test dat
results_linreg = test_df[["Country", "Rank", "Score"]].copy()
results_linreg["Predicted_Score"] = y_pred_linreg

# Convert predicted scores into predicted rankings 
results_linreg["Predicted_Rank"] = (
    results_linreg["Predicted_Score"]
    .rank(ascending=False, method="first") 
    .astype(int)
)

# Compute absolute error in rank for evaluation
results_linreg["Rank_Error"] = (results_linreg["Rank"] - results_linreg["Predicted_Rank"]).abs()

# Chart the results
results_linreg.head()


# In[25]:


mean_rank_error_linreg = results_linreg["Rank_Error"].mean()

print("Mean Rank Error:", mean_rank_error_linreg)

print("\nBest Cases (lowest rank error):")
display(results_linreg.sort_values("Rank_Error").head(5))

print("\nWorst Cases (highest rank error):")
display(results_linreg.sort_values("Rank_Error", ascending=False).head(5))


# In[26]:


# Scatter plot for linear regression
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred_linreg, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red", linestyle="--", label="Perfect prediction")
plt.xlabel("True 2019 happiness score")
plt.ylabel("Predicted 2019 happiness score (linear regression)")
plt.title("Linear regression: predicted vs true 2019 scores")
plt.legend()
plt.tight_layout()
plt.show()


# The linear regression model achieves a mean absolute error of about 0.42 on the test data happiness scores and an R² of approximately 0.75. This means 75% of the data is explained by the variation in the test data scores using only GDP, social support, health, freedom, generosity, and corruption. When I then converted the predicted scores into a ranking and compared them with the original test data rankings, the average rank error is about 15-16 ranks, with a couple countries predicted almost accurately and other off by more than 50 ranks.
# 
# The scatter plot of true versus predicted test data scores show that most of the points lie near the diagonal "perfect prediction" line. However, there is still quite a bit of spread, especially the countries with lower happiness scores. To conclude, linear regression serves as a good baseline model, however, there can still be improvement to the model as the lower happiness scores have significant deviations.

# ### 5.2 Random forest regression (Model 2)

# This will be a more flexible model, where it can predict the happiness score from the same set of features I used for linear regression. A random forest builds a collection of decision trees, each trained from a bootstrap sample of the training data and a random subset of features, and averages their predictions. What makes random forest regression distrinct from linear regression is that it can capture non-linear relationships so this is a natural model to proceed linear regression.

# In[27]:


# Libraries
from sklearn.ensemble import RandomForestRegressor

# Initialize the random forest regressor
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,   
    min_samples_leaf=3
)

# Fitting training data
rf.fit(X_train, y_train)

# Predict test data scores
y_pred_rf = rf.predict(X_test)

# Evaluate on test data scores
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

mse_rf, mae_rf, r2_rf


# In[28]:


# Build a results DataFrame for the random forest model
results_rf = test_df[["Country", "Rank", "Score"]].copy()
results_rf["Predicted_Score"] = y_pred_rf
# Convert predicted scores to predicted ranks
results_rf["Predicted_Rank"] = (
    results_rf["Predicted_Score"]
    .rank(ascending=False, method="first")
    .astype(int)
)
# Rank error for test data
results_rf["Rank_Error"] = (results_rf["Rank"] - results_rf["Predicted_Rank"]).abs()

# Average rank error and a few best/worst cases
mean_rank_error_rf = results_rf["Rank_Error"].mean()

print("Mean Rank Error:", mean_rank_error_rf)

print("\nBest Cases (lowest rank error):")
display(results_rf.sort_values("Rank_Error").head(5))

print("\nWorst Cases (highest rank error):")
display(results_rf.sort_values("Rank_Error", ascending=False).head(5))


# In[29]:


# Plotting random forest regression
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red", linestyle="--", label="Perfect prediction")
plt.xlabel("True 2019 happiness score")
plt.ylabel("Predicted 2019 happiness score (random forest)")
plt.title("Random forest: predicted vs true 2019 scores")
plt.legend()
plt.tight_layout()
plt.show()


# The random forest model achieves a mean absolute error of about 0.48 on the test data happiness scores and an R² of approximately 0.71. This means 71% of the data is explained by the variation in the test data scores using only GDP, social support, health, freedom, generosity, and corruption. The average absolute rank error is about 17 ranks, and like linear regression, there were couple countries that were accurately predicted while other were off by more than 50 ranks.
# 
# After using this random forest model, I found it surprisingly worse than the linear regression on this data set. I assume this might be as a result of the small dataset (approx. 150 countries) and the relationship between features like GDP, social support, and health versus happiness score to be somewhat linear. Because random forest model is more complex than linear regression, it would benefit fro larger datasets and patterns that are more non-linear. I can conclude that from my models, that linear regression would be a better fit here. 

# ### 5.3 KNN regression (Model 3)

# For the third model, I will use KNN regression. KNN makes predictions for a country by looking at the k most similar countries in the training data (based on the features) and averaging their scores, with the closer neighbors having more influence. This model can perform better as it captures local patterns in the data, but can be more sensitive to noise and the number chosen for k.

# In[30]:


from sklearn.neighbors import KNeighborsRegressor

# Initialize KNN regressor
# n_neighbors controls how many nearest neighbors we average over
knn = KNeighborsRegressor(
    n_neighbors=10, # n2s: mess with this 
    weights="distance"
)

# Fit training data
knn.fit(X_train, y_train)

# Predict test data scores
y_pred_knn = knn.predict(X_test)

# Evaluate on test data scores
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse_knn = mean_squared_error(y_test, y_pred_knn)
mae_knn = mean_absolute_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)

mse_knn, mae_knn, r2_knn


# In[31]:


# Results DataFrame for KNN model
results_knn = test_df[["Country", "Rank", "Score"]].copy()
results_knn["Predicted_Score"] = y_pred_knn

# Predicted ranks from predicted scores
results_knn["Predicted_Rank"] = (
    results_knn["Predicted_Score"]
    .rank(ascending=False, method="first")
    .astype(int)
)

# Rank error for test data
results_knn["Rank_Error"] = (results_knn["Rank"] - results_knn["Predicted_Rank"]).abs()

# Average rank error and a few best/worst cases
mean_rank_error_knn = results_knn["Rank_Error"].mean()

print("Mean Rank Error:", mean_rank_error_knn)

print("\nBest Cases (lowest rank error):")
display(results_knn.sort_values("Rank_Error").head(5))

print("\nWorst Cases (highest rank error):")
display(results_knn.sort_values("Rank_Error", ascending=False).head(5))


# In[32]:


# Plotting KNN regression
plt.figure(figsize=(6, 4))
sns.scatterplot(x=y_test, y=y_pred_knn, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red", linestyle="--", label="Perfect prediction")
plt.xlabel("True 2019 happiness score")
plt.ylabel("Predicted 2019 happiness score (KNN)")
plt.title("KNN: predicted vs true 2019 scores")
plt.legend()
plt.tight_layout()
plt.show()


# The KNN model achieves a mean absolute error of about 0.38 on the test data happiness scores and a R² of 0.80. This means 80% of the variation in test data scores can be explained in the test data scores. The average absolute rank error is about 14 ranks and a similar effect can be observerd with a couple countries being accurately predicted and a couple off by a margin of ranks.
# 
# The KNN model performs better than the previous two models achieving better statistics. The coefficient of determination increases to 80% from the 75% in the multiple linear regression. The KNN model also achieves the lowest mean absolute error of 0.38 compared to that 0.48 found from multiple linear regression. This suggests that for this dataset, prediciting a happiness score based on its nearest neighbors in feature space is more effective than linear regression or random forest regression. Although KNN is better, the model still struggles with the same countries the previous models struggles with, specficially the countries with a low true happiness score.

# In[33]:


# Charting statistics of each model for easy comaprison
model_metrics = pd.DataFrame({
    "Model": ["Linear regression", "Random forest", "KNN (k=10)"],
    "MAE_score": [mae_linreg, mae_rf, mae_knn],
    "R2_score": [r2_linreg, r2_rf, r2_knn],
    "Mean_rank_error": [mean_rank_error_linreg, mean_rank_error_rf, mean_rank_error_knn]
})

model_metrics


# As a final analysis, we can see that the random forest model performed worse than both models. This does not mean the model is bad, rather is not compatible with this dataset. Linear regression is a good baseline model and a useful reference when trying to achieve better models. The KNN model visibly has better statistics and we can can conclude that it was the best of the three models used on this dataset to predict the happiness score on the test data.

# ### 6. Personal happiness score formula

# After EDA and the results of my models, I will define my own personal happiness score that puts more weight on social support and health, with slight decrease in GDP, freedom, generosity and corruption. Using my modificatoins, I will computer an alternative ranking for the test data and compare it to the official World Happiness ranking.

# In[34]:


# My weights
w_gdp = 0.20
w_social = 0.25
w_health = 0.25
w_freedom = 0.15
w_generosity = 0.05
w_corruption = 0.10

# Apply the formula to test data
custom_2019 = test_df.copy()

custom_2019["MyScore"] = (
    w_gdp        * custom_2019["GDP"] +
    w_social     * custom_2019["Social_support"] +
    w_health     * custom_2019["Health"] +
    w_freedom    * custom_2019["Freedom"] +
    w_generosity * custom_2019["Generosity"] +
    w_corruption * custom_2019["Corruption"]
)

# Convert MyScore into a custom ranking
custom_2019["MyRank"] = (
    custom_2019["MyScore"]
    .rank(ascending=False, method="first")
    .astype(int)
)

# Compare custom rank vs official test rank
custom_2019["Rank_Diff"] = custom_2019["Rank"] - custom_2019["MyRank"]
custom_2019.sort_values("MyRank").head(10)  


# In[35]:


# Charting countries whose ranked changed the most
custom_2019.sort_values("Rank_Diff").head(10)      
custom_2019.sort_values("Rank_Diff", ascending=False).head(10)  


# After messing around with the weights, I've settled on weights for my personal equation. 0.20 for GDP, 0.25 for social support, 0.25 for health, 0.15 for freedom, 0.05 for generosity, and 0.10 for corruption. The resulting test data ranking placed countries like Singapore, Norway, Switzerland, Luxembourg, and Denmark at the top. Some of these countries experienced large ranked changes, for example, Singapore went from 34 to 1 which is a 33 rank difference. There are many countires that didn't experience that drastic of a change in my top 10 indicating that they must've already been a happy country.
# 
# On the other hand, some countries that are highly ranked in the official list move slightly down in my ranking (for example, Finland drops from rank 1 to rank 10), because my formula gives relatively more weight to economic output and health compared to the exact combination used in the official happiness score. Overall, my personal ranking emphasizes social support and health outcomes with stable GPD even more than the World Happiness Report, leading to noticeable differences for countries whose economic and health profiles are stronger than their reported happiness score.

# # Conclusions
# 
# My analysis through EDA and machine learning models indicate that social support, life expectancy, and GDP are the most important numeric data pieces of national happiness, with freedom have some impact, and generosity and corruption have much smaller effects. If I were to lead a country with the goal of making its citizens happy, I would prioritize policies that strengthen the social safety net and construct community support systems to help build strong social support. I would also ensure public health and easy-access to healthcare to maintain high life expectancy. I would work to reduct corruption all in efforts to increase happiness levels.
# 
