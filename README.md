# Black-Box-Regularizer
Ridge &amp; Lasso like regularisation for black box models.

Use one of "Noise addition" or "Dropout" to achieve Ridge-like regularization on any black box models

Use one of "Robust" to achieve Lasso-like regularization

A black box model is any function that takes (X_train, y_train) and return a function that takes (X_test) and return (y_test)

## Performance on simulated datasets
Performance on simulated datasets with a linear Data Generating Process. R2 is controlled at 80%. # of predictors, correlation among predictors and true beta are controlled to vary between p=(25, 50), corr=(0.25, 0.5), true beta = ("sparse", "dense"). 
Number of train observations = 100, Number of test observations = 500. Best Parameters are tuned using 5-fold CV.
![alt text](https://github.com/johncky/Black-Box-Regularizer/blob/main/pic/res.png?raw=true)


## Basic Usage
```python
# See code comments for detailed tuning parameters
blackbox_tuner(model, X_train, y_train, reg_method='dropout') 
blackbox_tuner(model, X_train, y_train, reg_method='noiseaddition')
blackbox_tuner(model, X_train, y_train, reg_method='robust')
```