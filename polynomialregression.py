# POLYNOMIAL REGRESSION

# Importing liberaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing dataset
dataset = pd.read_csv("Position_Salaries.csv")

# Seperating dependent and independent variables
# We should always try to keep x as a matrix and y as a vector
# For x we did used [:,1:2] instead of [:,1] so that it is treated as a matrix 
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values




# Here we dont need to split data into training and testing set as we already have very less data to train.
# And secondly its a trend dataset hence we want to anayse the whole trend at once to get accurate model predictions.
# To get more accurate results we need to have maximum information as possible to perfectly understand the correlation in this dataset. 
# Splitting the dataset into Training set and Test set
'''
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
'''


# We dont need feature scaling as the polynomial regression liberary/function takes care of it.
# Feature Scaling -> Normalizing the range of data/vairiable values
'''
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
'''


# Emplimenting Linear regression just for reference
# Training the Linear Regression model on the whole dataset
'''
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
'''


# Fitting Polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
# Below 2 lines of code will change our current feature matrix x to a new feature matrix "x_poly" which will contain the feature values and alo their power values Ex. x,x^2,x^3.... etc.
# We need to pass the number of degrees uptil which power of feature we want in our new feature matrix.
# we can vary the value of degree and rebuild the model again and again to get the best model and the respective value of power.
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
# Unlike multiple linear equation we dont have to add constant by adding the column of one's(1's) (Intercept) it is done automatically by PolynomialFeatures()
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)





# Emplimenting Linear regression just for reference
# Visualising the Linear Regression results
'''
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
'''



# Visualising the Polynomial Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
# If we want the graph to be more accurate by making inputs complicated.
# It gives a much smoother curve.
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



# Emplimenting Linear regression just for reference
# Now suppose we want predict a salary on a random level
'''
# Emplimenting Linear regression just for reference
# Predicting a new result with Linear Regression
print(lin_reg.predict([[6.5]]))
'''


# Predicting a new result with Polynomial Regression
# Now suppose we want predict a salary on a random level
# Suppose we want to find the salary at level 6.5
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))
