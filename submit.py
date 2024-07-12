import numpy as np
import sklearn.svm
from scipy.linalg import khatri_rao

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points

	feat = np.empty((X.shape[0], 2*X.shape[1]-1))
	for i in range(X.shape[0]):

		d = 1 - 2 * X[i, :31]
		feat[i, :31] = d
		d = np.append(d, 1 - 2*X[i, 31])
		cumulative_products = np.cumprod(d[::-1])[::-1]
		feat[i, 31:] = cumulative_products

	return feat

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y0_train, y1_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your models using training CRPs
	# X_train has 32 columns containing the challenge bits
	# y0_train contains the values for Response0
	# y1_train contains the values for Response1
	
	# THE RETURNED MODELS SHOULD BE TWO VECTORS AND TWO BIAS TERMS
	# If you do not wish to use a bias term, set it to 0
  model = sklearn.svm.LinearSVC(C=100, tol=1e-5, penalty='l2', max_iter=1000, loss='squared_hinge', dual=False)
  feat = my_map(X_train)
  model.fit(feat, y0_train)
  w0 = model.coef_
  b0 = model.intercept_

  model.fit(feat, y1_train)
  w1 = model.coef_
  b1 = model.intercept_
  return w0, b0, w1, b1
