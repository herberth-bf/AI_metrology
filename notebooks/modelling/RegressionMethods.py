# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:37:30 2019

REGRESSION METHODS


@author: Herberth Frohlich
"""

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


class RegressionMethods:
    
	def __init__(self):
		pass
	
	
    def random_forest_regressor(self, X_train, y_train, n_estimators, min_samples_split, max_depth):
        reg = RandomForestRegressor(n_estimators=n, oob_score=True, 
									verbose=0, max_depth=max_depth, min_samples_split=min_samples_split, random_state=0,
									criterion='mae', max_features='sqrt') 
		regn=reg
		reg.fit(X_train, y_train)
		return reg, regn
	
	def gradient_boosting_regressor(self, X_train, y_train, n_estimators, min_samples_split, max_depth):
        reg = GradientBoostingRegressor(n_estimators=n, learning_rate=0.5,
                            verbose=0, max_depth=max_depth, min_samples_split=min_samples_split, random_state=0,
                            criterion='mae', max_features='sqrt')
		regn = reg				
		reg.fit(X_train, y_train)
		return reg, regn
	
		
	def knn_regressor(self, X_train, y_train, n_neighbors, leaf_size=30):
        reg = GradientBoostingRegressor(n_eneighbors=n_neighbors, algorithm='auto',
                            verbose=0, leaf_size=leaf_size)
		regn=reg
		reg.fit(X_train, y_train)
		return reg, regn
	
	def voting_regressor(self, X_train, y_train, weights=None, *models):
		reg = []
		for m in models:
			reg.append(m)
        ereg = VotingRegressor(reg, weights=weights)
		ereg.fit(X_train, y_train)
		return ereg
		
	def lasso_regression(self, X_train, y_train, alpha):
		reg = Lasso(alpha=alpha)
		regn=reg
		reg.fit(X_train, y_train)
		return reg, regn