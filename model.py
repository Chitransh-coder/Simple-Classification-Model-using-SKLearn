import pandas as pd
import numpy as np
import matplotlib.pyplot as mp
import joblib as jb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# Importing the dataset
dataset = pd.read_excel('./assets/bean.xlsx')

x = dataset[['Area','Perimeter','MajorAxisLength','MinorAxisLength','AspectRatio','Eccentricity','ConvexArea','EquivDiameter','Extent','Solidity','roundness','Compactness','ShapeFactor1','ShapeFactor2','ShapeFactor3','ShapeFactor4']]
y = dataset['Class']
y = y.replace('SEKER',0)
y = y.replace('BARBUNYA',1)
y = y.replace('BOMBAY',2)
y = y.replace('CALI',3)
y = y.replace('HOROZ',4)
y = y.replace('SIRA',5)
y = y.replace('DERMASON',6)

features = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
            'AspectRatio', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent',
            'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2',
            'ShapeFactor3', 'ShapeFactor4']

label = ['Class']
labelnames= ['Seker', 'Barbunya', 'Bombay', 'Cali', 'Dermosan', 'Horoz', 'Sira']

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)

# regularization parameter
reg = 0.01

# Define preprocessing for numeric columns (scale them)
numeric_features = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)])

# Create preprocessing and training pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', LogisticRegression(C=1/reg, solver='lbfgs', multi_class='auto', max_iter=10000))])

# fit the pipeline to train a linear regression model on the training set
model = pipeline.fit(x_train, y_train)
res = model.predict(x_test)
prob = model.predict_proba(x_test)
auc = roc_auc_score(y_test, prob, multi_class='ovr')
fpr = {}
tpr = {}
thresh ={}
n_class = 7
for i in range(n_class):
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, prob[:,i], pos_label=i)
print('AUC Score: %.3f' % auc)


# ROC Curve
mp.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='Seker vs Rest')
mp.plot(fpr[1], tpr[1], linestyle='--',color='green', label='Barbunya vs Rest')
mp.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='Bombay vs Rest')
mp.plot(fpr[3], tpr[3], linestyle='--',color='red', label='Cali vs Rest')
mp.plot(fpr[4], tpr[4], linestyle='--',color='yellow', label='Dermosan vs Rest')
mp.plot(fpr[5], tpr[5], linestyle='--',color='purple', label='Horoz vs Rest')
mp.plot(fpr[6], tpr[6], linestyle='--',color='black', label='Sira vs Rest')
mp.title('Multiclass ROC curve')
mp.xlabel('False Positive Rate')
mp.ylabel('True Positive rate')
mp.legend(loc='best')
mp.show()

# Saving the model
jb.dump(model, './assets/model.pkl')