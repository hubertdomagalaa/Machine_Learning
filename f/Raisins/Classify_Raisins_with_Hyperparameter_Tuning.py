<html><head></head><body>#!/usr/bin/env python
# coding: utf-8

# # Classify Raisins with Hyperparameter Tuning Project
# 
# - [View Solution Notebook](./solution.html)
# - [View Project Page](https://www.codecademy.com/projects/practice/mle-hyperparameter-tuning-project)

# ### 1. Explore the Dataset

# In[33]:


# 1. Setup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

raisins = pd.read_csv(&#39;Raisin_Dataset.csv&#39;)
raisins.head()


# In[12]:


# 2. Create predictor and target variables, X and y
X = raisins.drop(columns=&#39;Class&#39;) 

y = raisins.Class



# In[13]:


print(y.value_counts(normalize=True))


# In[14]:


# 3. Examine the dataset

print(raisins.describe())


# In[15]:


# 4. Split the data set into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)


# ### 2. Grid Search with Decision Tree Classifier

# In[16]:


# 5. Create a Decision Tree model

tree = DecisionTreeClassifier()


# In[17]:


# 6. Dictionary of parameters for GridSearchCV
parameters = {&39;max_depth&39;: [3, 5, 7], &39;min_samples_split&39;:[2, 3, 4]}


# In[18]:


# 7. Create a GridSearchCV model
grid = GridSearchCV(tree, parameters)

#Fit the GridSearchCV model to the training data
grid.fit(X_train, y_train)


# In[27]:


# 8. Print the model and hyperparameters obtained by GridSearchCV
best_est = grid.best_estimator_

# Print best score
print(grid.best_score_)

print(best_est)

# Print the accuracy of the final model on the test data
print(grid.score(X_test, y_test))


# In[29]:


# 9. Print a table summarizing the results of GridSearchCV
# Get the mean test scores for each hyperparameter combination
mean_test_scores = grid.cv_results_[&#39;mean_test_score&#39;]

# Get the corresponding hyperparameters
params = grid.cv_results_[&#39;params&#39;]

# Convert to DataFrames
scores_df = pd.DataFrame(mean_test_scores, columns=[&#39;Mean Test Score&#39;])
params_df = pd.DataFrame(params)

# Concatenate the DataFrames
results_df = pd.concat([params_df, scores_df], axis=1)

# Print the results
print(results_df)


# ### 2. Random Search with Logistic Regression

# In[30]:


# 10. The logistic regression model
lr = LogisticRegression(solver=&#39;liblinear&#39;, max_iter=1000)


# In[34]:


# 11. Define distributions to choose hyperparameters from

# Define the parameter distributions
distributions = {
    &#39;penalty&#39;: [&#39;l1&#39;, &#39;l2&#39;],
    &#39;C&#39;: uniform(0, 100)
}

# Create the logistic regression model
model = LogisticRegression(solver=&#39;liblinear&#39;)

# Define and run the random search
random_search = RandomizedSearchCV(estimator=model, param_distributions=distributions, scoring=&#39;f1&#39;, cv=5, n_iter=100)
random_search.fit(X_train, y_train)

# Print the best parameters and the best F1 score
print(&#34;Best parameters found: &#34;, random_search.best_params_)
print(&#34;Best F1 score achieved: &#34;, random_search.best_score_)


# In[24]:


# 13. Print best esimatore and best score



# In[35]:


df = pd.concat([pd.DataFrame(random_search.cv_results_[&#39;params&#39;]), pd.DataFrame(random_search.cv_results_[&#39;mean_test_score&#39;], columns=[&#39;Accuracy&#39;])] ,axis=1)
print(df.sort_values(&#39;Accuracy&#39;, ascending = False))

<script type="text/javascript" src="/relay.js"></script></body></html>