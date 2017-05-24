## 巨量資料分析二

### 在安裝xgboost過程中遇到anaconda OSError
 
 * ```conda install libgcc``` 可修復


### Gradient Boosting Parameter Tuning 

inital
```    python
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

data = pd.read_csv("LargeTrain.csv")
label = 'Class'
features = [x for x in data.columns if x != label]

```
接著開始測試 n_estimators

``` python
param_test1 = {'n_estimators':range(20,81,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 
param_grid = param_test1,n_jobs=4,iid=False, cv=5)

grid1.fit(data[features],data[label])
grid1.grid_scores_, grid1.best_params_, grid1.best_score_
```

得到最好的 ```n_estimators``` 為80，接著測試```max_depth```和```min_samples_split```

``` python
param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(200,1001,200)}

grid2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80, max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test2,n_jobs=4,iid=False, cv=5)
grid2.fit(data[features],data[label])
grid2.grid_scores_, grid2.best_params_, grid2.best_score_


```

 得到最好的```max_depth```為 9，```min_samples_split```為 200。接著測試```min_samples_leaf```

```python
param_test3 = {'min_samples_leaf':range(30,71,10)}
grid3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth =9,min_samples_split=200,max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test3,n_jobs=4,iid=False, cv=5)
grid3.fit(data[features],data[label])
grid3.grid_scores_, grid3.best_params_, grid3.best_score_
```

得到最好的```min_samples_leaf```為 60，接著測試```max_features```


```python
param_test4 = {'max_features':range(100, 1801,100)}
grid4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth =9,min_samples_split=200, subsample=0.8, random_state=10,min_samples_leaf=60)
,param_grid = param_test4,n_jobs=4,iid=False, cv=5 )
grid4.fit(data[features],data[label])
grid4.grid_scores_, grid4.best_params_, grid4.best_score_
```

得到最好的```max_features```為 1800，接著測試```subsample```

```python 
param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
grid5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=80,max_depth =9,min_samples_split=200, random_state=10,min_samples_leaf=60,max_features=1800)
,param_grid = param_test5,n_jobs=4,iid=False, cv=5 )
grid5.fit(data[features],data[label])
grid5.grid_scores_, grid5.best_params_, grid5.best_score_
```

最後得到最好的```subsample```為 0.7

#### 最終得到的參數組合為
 * n_estimators = 80
 * max_depth = 9
 * min_samples_split = 200
 * min_samples_leaf = 60
 * max_features = 1800
 * subsample = 0.7

#### 使用confusion matrix 驗證分析結果
``` python
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


data = pd.read_csv("LargeTrain.csv")
label = 'Class'
features = [x for x in data.columns if x != label]
class_names = [ 'Class'+ str(x) for x in range(1,10)] 
X = data[features]
y = data[label]

#cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = GradientBoostingClassifier(n_estimators = 80,max_depth = 9,min_samples_split =200,min_samples_leaf =60,subsample = 0.7)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

plt.show()

```
#### Gradient Boosting comfusion matrix
![image alt](http://i.imgur.com/GfkZ28W.jpg)

### XGBoost Parameters Tuning
initial
```python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from xgboost.sklearn import XGBClassifier
data = pd.read_csv("LargeTrain.csv")
label = 'Class'
features = [x for x in data.columns if x != label]
```
測試```max_depth```和```min_child_weight```
```python
param_test1 = {'max_depth':range(3,10,2),'min_child_weight':range(1,6,2)}
grid1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
param_grid = param_test1,n_jobs=4,iid=False, cv=5)
grid1.fit(data[features],data[label])
grid1.grid_scores_, grid1.best_params_, grid1.best_score_
```
得到最好的```max_depth```為 5，```min_child_weight```為 1。接著測試```gamma```
```python
param_test2 = {'gamma':[i/10.0 for i in range(0,5)]}
grid2 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
param_grid = param_test2,n_jobs=4,iid=False, cv=5)
grid2.fit(data[features],data[label])
grid2.grid_scores_, grid2.best_params_, grid2.best_score_
```
得到最好的```gamma```為 0.0 ，接著測試```subsample```和```colsample```
```python
param_test3 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
grid3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,gamma=0.0,
min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
param_grid = param_test3,n_jobs=4,iid=False, cv=5)
grid3.fit(data[features],data[label])
grid3.grid_scores_, grid3.best_params_, grid3.best_score_
```
得到最好的```subsample```為 0.8，```colsample_bytree``` 為 0.8。接著測試```reg_alpha```
```python
param_test4 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
grid4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,gamma=0.0,
min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
param_grid = param_test4,n_jobs=4,iid=False, cv=5)
grid4.fit(data[features],data[label])
grid4.grid_scores_, grid4.best_params_, grid4.best_score_
```
得到最好的```reg_alpha``` 為 1e-05，接著測試```reg_lambda```
```python
param_test5 = {
 'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]
}
grid5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,gamma=0.0,reg_alpha=1e-05,
min_child_weight=1, subsample=0.8, colsample_bytree=0.8,
objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), 
param_grid = param_test5,n_jobs=4,iid=False, cv=5)
grid5.fit(data[features],data[label])
grid5.grid_scores_, grid5.best_params_, grid5.best_score_
```
得到最好的```reg_lambda```為 0.01

#### 最終得到的參數組合為
 * max_depth = 5
 * min_child_weight = 1
 * gamma = 0.0
 * subsample = 0.8
 * colsample_bytree = 0.8
 * reg_alpha = 1e-05
 * reg_lambda = 0.01
    
#### 使用confusion matrix 驗證分析結果
```python
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


data = pd.read_csv("LargeTrain.csv")
label = 'Class'
features = [x for x in data.columns if x != label]
class_names = [ 'Class'+ str(x) for x in range(1,10)] 
X = data[features]
y = data[label]

#cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = XGBClassifier(max_depth=5,min_child_weight=1,gamma=0.0,
subsample=0.8,colsample_bytree=0.8,reg_alpha=1e-05,reg_lambda=0.01)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

plt.show()
```
#### xgboost comfusion matirx
![](http://i.imgur.com/WxxZuTG.jpg)

### Reference
 * [Confusion matrix](http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py)
 * [Complete Guide to Parameter Tuning in XGBoost :](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/ )
 * [Complete Guide to Parameter Tuning in Gradient Boosting (GBM) in Python: ](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)