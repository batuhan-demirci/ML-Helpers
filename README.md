# ML-Helpers

This repository contains some code snippets that I find it useful.

## Subplotting
```python
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline # to avoid using plt.show() in Jupyter Notebook

x = np.linspace(0, 5, 11)
y = x ** 2

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
plt.plot

axes[0].plot(x, x**2, 'g--')
axes[0].set_xlabel('x')
axes[0].set_ylabel('x ** 2')
axes[0].set_title('title for first plot')

axes[1].plot(y, y**3, 'r-x')
axes[1].set_xlabel('y')
axes[1].set_ylabel('y ** 3')
axes[1].set_title('title for second plot')

fig
plt.tight_layout() # to prevent overlapping between plots
```

## Correlation heatmap
```python
import seaborn as sns
%matplotlib inline

tips = sns.load_dataset('tips')
sns.heatmap(tips.corr(), cmap='coolwarm', annot=True)
```

## Basic Linear Regression template
```python
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

%matplotlib inline

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

lm = LinearRegression()
lm.fit(X_train, y_train)

predictions = lm.predict(X_test)

# visualize model results
plt.scatter(y_test, predictions)
plt.xlabel('Y')
plt.ylabel('Predicted Y')

# print evaluation metrics
print('MAE: ', metrics.mean_absolute_error(y_test, predictions))
print('MSE: ', metrics.mean_squared_error(y_test, predictions))
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# view coeffecients in DataFrame form
coeffecients = pd.DataFrame(lm.coef_, X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients
```

## Visualization of null values
```python
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

plt.figure(figsize=(12,4))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
```

## Choosing a K-value for KNN
```python
import matplotlib.pyplot as plt

%matplotlib inline

error = []

for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test)) # get the mean of wrong predictions

# plot for better understanding
plt.figure(figsize=(10,6))
plt.plot(range(1, 40), error, color = 'blue', linestyle='dashed', marker='o'
        ,markerfacecolor='red', markersize=5)
plt.title('Error vs K')
plt.xlabel('K')
plt.ylabel('Error')
```

## GridSearchCV template
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C':[0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(), param_grid, verbose=3)
grid.fit(X_train, y_train)
print(grid.best_estimator_)
```

## NLP Pipeline template
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB()),
])

pipeline.fit(X_train,y_train)
```

## Keras - EarlyStopping & Dropout
```python
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout

%matplotlib inline

# stop training when a monitored quantity has stopped improving
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

model = Sequential()

model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5)) # randomly half the neuron of each batch are turned off

model.add(Dense(15, activation='relu'))
model.add(Dropout(0.5))

# binary classification
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test),
         callbacks=[early_stop])

# plot model history
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()

# classification report
predictions = model.predict_classes(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print('\n')
print(confusion_matrix(y_test, predictions))
```