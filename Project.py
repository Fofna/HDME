#!/usr/bin/env python
# coding: utf-8

# # High Dimensional Matrix Estimation Project

# ## Algorithm 1 :  Alternative-Least-Squares with Weighted-λ-Regularization

# In[1]:


import numpy as np

def ALS(Y, R, K, steps=10, lambda_=0.01, weights=None):
    """
    Alternating Least Squares with Weighted-λ-Regularization
    
    Arguments:
    Y -- rating matrix with shape (num_users, num_items)
    R -- binary matrix indicating which entries are known (1) or unknown (0)
    K -- number of latent factors
    steps -- number of iterations for the ALS algorithm
    lambda_ -- regularization term for matrix factorization
    weights -- weight for each user for regularization
    
    Returns:
    U -- user matrix with shape (num_users, K)
    V -- item matrix with shape (K, num_items)
    """
    
    num_users, num_items = Y.shape
    U = np.random.randn(num_users, K)
    V = np.random.randn(K, num_items)
    
    if weights is None:
        weights = np.ones(num_users)
    
    for step in range(steps):
        for u in range(num_users):
            # only update for known ratings
            item_idx = np.where(R[u,:] == 1)[0]
            Y_u = Y[u, item_idx]
            V_u = V[:, item_idx]
            # calculate user factor
            A = np.dot(V_u, V_u.T) + lambda_ * weights[u] * np.eye(K)
            B = np.dot(V_u, Y_u.T)
            U[u,:] = np.linalg.solve(A, B)
        
        for i in range(num_items):
            # only update for known ratings
            user_idx = np.where(R[:,i] == 1)[0]
            Y_i = Y[user_idx, i]
            U_i = U[user_idx, :]
            # calculate item factor
            A = np.dot(U_i.T, U_i) + lambda_ * np.eye(K)
            B = np.dot(U_i.T, Y_i)
            V[:,i] = np.linalg.solve(A, B)
    
    return U, V


# In[2]:


# example rating matrix
Y = np.array([[5, 3, 0, 1],
              [4, 0, 0, 1],
              [1, 1, 0, 5],
              [1, 0, 0, 4],
              [0, 1, 5, 4]])

# binary matrix indicating known (1) or unknown (0) ratings
R = (Y != 0) * 1

# number of latent factors
K = 2

U, V = ALS(Y, R, K, steps=100, lambda_=0.1)

# predict ratings for all users and items
predicted_Y = np.dot(U, V)
print("Predicted rating matrix:")
print(predicted_Y)


# ## Algorithm 2 : Non-Negative Least Squares

# In[3]:


import numpy as np
import scipy.optimize

def NNLS(V, K, max_iter=100, tol=1e-5):
    """
    Non-Negative Least Squares for Non-Negative Matrix Factorization
    
    Arguments:
    V -- matrix to be factorized with shape (num_rows, num_cols)
    K -- number of latent factors
    max_iter -- maximum number of iterations for optimization
    tol -- tolerance for optimization convergence
    
    Returns:
    W -- non-negative basis matrix with shape (num_rows, K)
    H -- non-negative coefficient matrix with shape (K, num_cols)
    """
    
    num_rows, num_cols = V.shape
    
    # initialize W and H with random positive values
    W = np.abs(np.random.randn(num_rows, K))
    H = np.abs(np.random.randn(K, num_cols))
    
    for i in range(max_iter):
        # update H
        for j in range(num_cols):
            H[:,j] = scipy.optimize.nnls(W, V[:,j])[0]
        
        # update W
        for j in range(num_rows):
            W[j,:] = scipy.optimize.nnls(H.T, V[j,:].T)[0]
        
        # calculate reconstruction error
        V_recon = np.dot(W, H)
        error = np.linalg.norm(V - V_recon) / np.linalg.norm(V)
        
        if error < tol:
            break
    
    return W, H


# In[4]:


import numpy as np

# example data matrix
V = np.array([[5, 3, 0, 1],
              [4, 0, 0, 1],
              [1, 1, 0, 5],
              [1, 0, 0, 4],
              [0, 1, 5, 4]])

# number of latent factors
K = 2

W, H = NNLS(V, K, max_iter=100, tol=1e-5)

# predict values for all entries in V
predicted_V = np.dot(W, H)
print("Predicted data matrix:")
print(predicted_V)


# ## Algorithm 3 : Non Negative Multiplicative Update 

# In[5]:


import numpy as np

def NMF(V, K, max_iter=100, tol=1e-5):
    """
    Non-Negative Multiplicative Update Algorithm for Non-Negative Matrix Factorization
    
    Arguments:
    V -- matrix to be factorized with shape (num_rows, num_cols)
    K -- number of latent factors
    max_iter -- maximum number of iterations for optimization
    tol -- tolerance for optimization convergence
    
    Returns:
    W -- non-negative basis matrix with shape (num_rows, K)
    H -- non-negative coefficient matrix with shape (K, num_cols)
    """
    
    num_rows, num_cols = V.shape
    
    # initialize W and H with random positive values
    W = np.abs(np.random.randn(num_rows, K))
    H = np.abs(np.random.randn(K, num_cols))
    
    for i in range(max_iter):
        # update H
        H = H * np.dot(W.T, V) / np.dot(W.T, np.dot(W, H))
        
        # update W
        W = W * np.dot(V, H.T) / np.dot(np.dot(W, H), H.T)
        
        # calculate reconstruction error
        V_recon = np.dot(W, H)
        error = np.linalg.norm(V - V_recon) / np.linalg.norm(V)
        
        if error < tol:
            break
    
    return W, H


# In[6]:


# example data matrix
V = np.array([[5, 3, 0, 1],
              [4, 0, 0, 1],
              [1, 1, 0, 5],
              [1, 0, 0, 4],
              [0, 1, 5, 4]])

# number of latent factors
K = 2

W, H = NMF(V, K, max_iter=100, tol=1e-5)

# predict values for all entries in V
predicted_V = np.dot(W, H)
print("Predicted data matrix:")
print(predicted_V)


# ## Implentating ALS-NMF model to 20M Netflix Rating Dataset

# In[7]:


import findspark
findspark.init()
import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("project").getOrCreate()


# In[8]:


# Import text data into an RDD
small_ratings_raw_data = spark.sparkContext.textFile("ratings.csv")
# Identify and display the first line
small_ratings_raw_data_header = small_ratings_raw_data.take(1)[0]
print(small_ratings_raw_data_header)
# Create RDD without header
all_lines = small_ratings_raw_data.filter(lambda l : l!=small_ratings_raw_data_header)


# In[9]:


#Split the fields (user, item, rating) into a new RDD.
from pyspark.sql import Row
split_lines = all_lines.map(lambda l : l.split(","))
ratingsRDD = split_lines.map(lambda p: Row(user=int(p[0]), item=int(p[1]),
                                     rating=float(p[2]), timestamp=int(p[3])))

# .cache(): the RDD is kept in memory once processed
ratingsRDD.cache()

# Display the two first rows
ratingsRDD.take(2)


# In[10]:


# Convert RDD to DataFrame
ratingsDF = spark.createDataFrame(ratingsRDD)


# ### Sampling and Rank Optimization for ALS NMF

# Random separation into three samples learning, validation and testing. The rank parameter is optimized by minimizing the error estimate on the test sample. This strategy, rather than cross-validation, is more suited to Large dataset.

# In[11]:


tauxTrain=0.6
tauxVal=0.2
tauxTes=0.2
# If the total is less than 1, the data is undersampled.
(trainDF, validDF, testDF) = ratingsDF.randomSplit([tauxTrain, tauxVal, tauxTes])
# Validation and testing to predict
validDF_P = validDF.select("user", "item")
testDF_P = testDF.select("user", "item")


# The error of imputation of the data, therefore of recommendation, is estimated on the validation sample for different values (grid) of the rank of the matrix factorization.
# 
# In principle, the value of the penalty parameter should also be optimised at 0.1 by default.
# 
# Important point: the factorization fit error only takes into account the values listed in the hollow matrix, not the "0s" which are missing data.

# In[12]:


from pyspark.ml.recommendation import ALS
import math
import collections
# Set the seed
seed = 5
#Number of max iteration for ALS
maxIter = 10
# L1 Regularization; also to be optimized
regularization_parameter = 0.1
# Choice of a grid for optimizing rank values
ranks = [4, 8, 12]
#Initializing variable
# Creating a dictionary to store the error by tested rank
errors = collections.defaultdict(float)
tolerance = 0.02
min_error = float('inf')
best_rank = -1
best_iteration = -1


# In[13]:


from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import *
for rank in ranks:
    als = ALS( rank=rank, seed=seed, maxIter=maxIter,
                      regParam=regularization_parameter)
    model = als.fit(trainDF)
    # Validation Sample Forecast
    predDF = model.transform(validDF).select("prediction","rating")
    #Remove unpredicter row due to no-presence of user in the train dataset
    pred_without_naDF = predDF.na.fill(0)
    # Compute the RMSE
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
    rmse = evaluator.evaluate(pred_without_naDF)
    print("Root-mean-square error for rank %d = "%rank + str(rmse))
    errors[rank] = rmse
    if rmse < min_error:
        min_error = rmse
        best_rank = rank
# Best result
print('Optimal: %s' % best_rank)


# ### Final prediction on the test sample.

# In[14]:


# We concatenate here the Train and Validation DataFrames.
trainValidDF = trainDF.union(validDF)
# We create a model with the new completed training Dataframe and the rank set to the optimal value.
als = ALS( rank=best_rank, seed=seed, maxIter=maxIter,
                  regParam=regularization_parameter)
model = als.fit(trainValidDF)
# Predicting on the Test DataFrame
testDF = model.transform(testDF).select("prediction","rating")
#Remove unpredicter row due to no-presence of user in the trai dataset
pred_without_naDF = predDF.na.fill(0)
# Calcul du RMSE
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                            predictionCol="prediction")
rmse = evaluator.evaluate(pred_without_naDF)
print("Root-mean-square error for rank %d = "%best_rank + str(rmse))

