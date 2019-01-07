import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.expand_frame_repr', False)
trainData = pd.read_csv('C:/Users/ajju/Desktop/ml/ml projects/3revenuepredict/train.csv')
testData = pd.read_csv('C:/Users/ajju/Desktop/ml/ml projects/3revenuepredict/test.csv')
trainData = trainData.drop('Id', axis=1)
testData = testData.drop('Id', axis=1)

trainData['Open Date'] = pd.to_datetime(trainData['Open Date'], format='%m/%d/%Y')   
testData['Open Date'] = pd.to_datetime(testData['Open Date'], format='%m/%d/%Y')

trainData['OpenDays']=""
testData['OpenDays']=""

dateLastTrain = pd.DataFrame({'Date':np.repeat(['12/16/2018'],[len(trainData)]) })
dateLastTrain['Date'] = pd.to_datetime(dateLastTrain['Date'], format='%m/%d/%Y')  
dateLastTest = pd.DataFrame({'Date':np.repeat(['12/16/2018'],[len(testData)]) })
dateLastTest['Date'] = pd.to_datetime(dateLastTest['Date'], format='%m/%d/%Y')  

trainData['OpenDays'] = dateLastTrain['Date'] - trainData['Open Date']
testData['OpenDays'] = dateLastTest['Date'] - testData['Open Date']

trainData['OpenDays'] = trainData['OpenDays'].astype('timedelta64[D]').astype(int)
testData['OpenDays'] = testData['OpenDays'].astype('timedelta64[D]').astype(int)

trainData = trainData.drop('Open Date', axis=1)
testData = testData.drop('Open Date', axis=1)


cityPerc = trainData[["City Group", "revenue"]].groupby(['City Group'],as_index=False).mean()
sns.barplot(x='City Group', y='revenue', data=cityPerc)

citygroupDummy = pd.get_dummies(trainData['City Group'])
trainData = trainData.join(citygroupDummy)

citygroupDummyTest = pd.get_dummies(testData['City Group'])
testData = testData.join(citygroupDummyTest)

trainData = trainData.drop('City Group', axis=1)
testData = testData.drop('City Group', axis=1)



#Regression on everything
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")

import numpy
xTrain = pd.DataFrame({'OpenDays':trainData['OpenDays'].apply(numpy.log),
                      'Big Cities':trainData['Big Cities'], 'Other':trainData['Other'],
                      'P2':trainData['P2'], 'P8':trainData['P8'], 'P22':trainData['P22'],
                      'P24':trainData['P24'], 'P28':trainData['P28'], 'P26':trainData['P26']})
#xTrain = trainData.drop(['revenue'], axis=1)
#xTrain['OpenDays'] = xTrain['OpenDays'].apply(numpy.log)
yTrain = trainData['revenue'].apply(numpy.log)
xTest = pd.DataFrame({'OpenDays':testData['OpenDays'].apply(numpy.log),
                      'Big Cities':testData['Big Cities'], 'Other':testData['Other'],
                     'P2':testData['P2'], 'P8':testData['P8'], 'P22':testData['P22'],
                      'P24':testData['P24'], 'P28':testData['P28'], 'P26':testData['P26']})

from sklearn import linear_model

cls = RandomForestRegressor(n_estimators=150)
cls.fit(xTrain, yTrain)
pred = cls.predict(xTest)
pred = numpy.exp(pred)
cls.score(xTrain, yTrain)



pred2 = []
for i in range(len(pred)):
    if pred[i] != float('Inf'):
        pred2.append(pred[i])

m = sum(pred2) / float(len(pred2))

for i in range(len(pred)):
    if pred[i] == float('Inf'):
        print("haha")
        pred[i] = m

testData = pd.read_csv('C:/Users/ajju/Desktop/ml/ml projects/3revenuepredict/test.csv')
submission = pd.DataFrame({
        "Id": testData["Id"],
        "Prediction": pred
    })
submission.to_csv('Revenuesubmission.csv',header=True, index=False)        
