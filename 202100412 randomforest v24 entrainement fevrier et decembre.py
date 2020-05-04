# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 19:18:31 2020
Modified on Mon May 4 18:25:31 2020

@author: Hugo
@author: Johnny
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
from pyrsgis import raster
from pyrsgis.convert import changeDimension
from scipy.signal import convolve2d as conv2
import pandas as pd
import matplotlib.pyplot as plt
import scipy


os.chdir("C:\\Users\Hugo\Desktop\Data for good\Images\\32 bits - 1ere sessions")
 # premier image : seredou 2017

MSI = 'Diecke_20170205_MSI.tif'
all_bands='seredou_20170205_allbands.tif'
truth_1='Diecke_20170205_GroundTruth.tif'



# Read the rasters as array
ds1, feature_MSI = raster.read(MSI, bands='all')
ds2, feature_all_bands = raster.read(all_bands, bands='all')
ds3, truth_1= raster.read(truth_1, bands='all')

# sur all bands
#feature_all_bands=np.delete(feature_all_bands,np.s_[8951:9120],axis=2)

# sur truth
#truth=np.delete(truth,6141,axis=0)

# sur MSI
#feature_MSI=np.delete(feature_MSI,8951,axis=1)

# test min max des 3 images+

print("Max pour truth:", truth_1.max() )
print("Min pour truth:", truth_1.min() )
print("Max pour all bands", feature_all_bands.max() )
print("Min pour all bands", feature_all_bands.min() )
print("Max pour MSI", feature_MSI.max() )
print("Min pour MSI", feature_MSI.min() )

print("-------------------")

# checks distribution de truth

# on remaplce les 0 de truth par des 3 - test pour 16 bits
#truth[truth==0]=3

print("distribution de truth ")
unique, counts = np.unique(truth_1, return_counts=True)
print(np.asarray((unique, counts/truth_1.size)).T)

print("-------------------")

# on remplace les valeurs négatives par 0, si il y en a

feature_MSI[feature_MSI<0]=0
feature_all_bands[feature_all_bands<0]=0

# on remaplce les 4 par des 3 dans le truth -- cas 32 bits uniquement
#truth[truth==4]=3


# on normalise chaque bande séparément en divisant par le max de la bande.

feature_MSI = feature_MSI / feature_MSI.max()

i=0
while i<10:
    feature_all_bands[i]=feature_all_bands[i]/feature_all_bands[i].max()
    i=i+1

# on vérifie que ça a marché

print("Max pour MSI", feature_MSI.max() )
print("Min pour MSI", feature_MSI.min() )    
print("-------------------")

i=0
while i<10:    
    print("Max pour all bands ",i,": ", feature_all_bands[i].max() )
    print("Min pour all bands ",i,": ", feature_all_bands[i].min() )
    i=i+1
    print("-------------------")



# test conv2
#test=np.array([[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3],[4,4,4,4,4],[5,5,5,5,5]])
#test_avg = (conv2(test,np.ones((3,3)),boundary='symm',mode='same')/9)

#çamarche mais il suppose que les valeurs en dehors de l'image valent 0 --> impact sur les valeurs aux frontières de l'image
# partiellement corrigé avec bounday = sym
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html#scipy.signal.convolve2d
    
# on calcule les moyennes des cases adjacentes - creation de 11 nouvelles colonnes


feature_MSI_avg = (conv2(feature_MSI,np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_0=(conv2(feature_all_bands[0],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_1=(conv2(feature_all_bands[1],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_2=(conv2(feature_all_bands[2],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_3=(conv2(feature_all_bands[3],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_4=(conv2(feature_all_bands[4],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_5=(conv2(feature_all_bands[5],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_6=(conv2(feature_all_bands[6],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_7=(conv2(feature_all_bands[7],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_8=(conv2(feature_all_bands[8],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_9=(conv2(feature_all_bands[9],np.ones((13,13)),boundary='symm',mode='same')/169)

# sur la truth 
# on enlève la première colonne et la dernière ligne - plus la peine maintenant

#truth=np.delete(truth,4664,0)
#truth=np.delete(truth,0,1)



# on passe en tabulaire

feature_all_bands = changeDimension(feature_all_bands)
feature_MSI = changeDimension(feature_MSI)
truth_1= changeDimension(truth_1)
feature_MSI_avg = changeDimension(feature_MSI_avg)
feature_all_bands_avg_0 = changeDimension(feature_all_bands_avg_0)
feature_all_bands_avg_1 = changeDimension(feature_all_bands_avg_1)
feature_all_bands_avg_2 = changeDimension(feature_all_bands_avg_2)
feature_all_bands_avg_3 = changeDimension(feature_all_bands_avg_3)
feature_all_bands_avg_4 = changeDimension(feature_all_bands_avg_4)
feature_all_bands_avg_5 = changeDimension(feature_all_bands_avg_5)
feature_all_bands_avg_6 = changeDimension(feature_all_bands_avg_6)
feature_all_bands_avg_7 = changeDimension(feature_all_bands_avg_7)
feature_all_bands_avg_8 = changeDimension(feature_all_bands_avg_8)
feature_all_bands_avg_9 = changeDimension(feature_all_bands_avg_9)

# concaténée des 22 champs 
# truth reste à part 

data_1=np.column_stack((feature_all_bands,feature_MSI,feature_all_bands_avg_0,feature_all_bands_avg_1,feature_all_bands_avg_2,feature_all_bands_avg_3,feature_all_bands_avg_4,feature_all_bands_avg_5,feature_all_bands_avg_6,feature_all_bands_avg_7,feature_all_bands_avg_8,feature_all_bands_avg_9,feature_MSI_avg))




# idem pour 2015

MSI_2 = 'Seredou_20151203_MSI.tif'
all_bands_2='Seredou_20151203_allbands.tif'
truth_2='Seredou_20151203_truth.tif'

# Read the rasters as array
ds1_2, feature_MSI_2 = raster.read(MSI_2, bands='all')
ds2_2, feature_all_bands_2 = raster.read(all_bands_2, bands='all')
ds3_2, truth_2 = raster.read(truth_2, bands='all')


#truth_2=np.delete(truth_2,4664,0)
#truth_2=np.delete(truth_2,0,1)


# on remaplce les 4 par des 3 dans le truth -- cas 32 bits uniquement
#truth_2[truth_2==4]=3


# sur all bands
#feature_all_bands_2=np.delete(feature_all_bands_2,np.s_[8951:9120],axis=2)



print("-------------------")

print("Max pour truth:", truth_2.max() )
print("Min pour truth:", truth_2.min() )
print("Max pour all bands", feature_all_bands_2.max() )
print("Min pour all bands", feature_all_bands_2.min() )
print("Max pour MSI", feature_MSI_2.max() )
print("Min pour MSI", feature_MSI_2.min() )

print("-------------------")

print("distribution de truth ")
unique, counts = np.unique(truth_2, return_counts=True)
print(np.asarray((unique, counts/truth_2.size)).T)

print("-------------------")

feature_MSI_2[feature_MSI_2<0]=0
feature_all_bands_2[feature_all_bands_2<0]=0

feature_MSI_2 = feature_MSI_2 / feature_MSI_2.max()

i=0
while i<10:
    feature_all_bands_2[i]=feature_all_bands_2[i]/feature_all_bands_2[i].max()
    i=i+1

# on vérifie que ça a marché

print("Max pour MSI", feature_MSI_2.max() )
print("Min pour MSI", feature_MSI_2.min() )    
print("-------------------")

i=0
while i<10:    
    print("Max pour all bands ",i,": ", feature_all_bands_2[i].max() )
    print("Min pour all bands ",i,": ", feature_all_bands_2[i].min() )
    i=i+1
    print("-------------------")
    




feature_MSI_avg_2 = (conv2(feature_MSI_2,np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_0_2=(conv2(feature_all_bands_2[0],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_1_2=(conv2(feature_all_bands_2[1],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_2_2=(conv2(feature_all_bands_2[2],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_3_2=(conv2(feature_all_bands_2[3],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_4_2=(conv2(feature_all_bands_2[4],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_5_2=(conv2(feature_all_bands_2[5],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_6_2=(conv2(feature_all_bands_2[6],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_7_2=(conv2(feature_all_bands_2[7],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_8_2=(conv2(feature_all_bands_2[8],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_9_2=(conv2(feature_all_bands_2[9],np.ones((13,13)),boundary='symm',mode='same')/169)


# on passe en tabulaire

feature_all_bands_2 = changeDimension(feature_all_bands_2)
feature_MSI_2 = changeDimension(feature_MSI_2)
truth_2= changeDimension(truth_2)
feature_MSI_avg_2 = changeDimension(feature_MSI_avg_2)
feature_all_bands_avg_0_2 = changeDimension(feature_all_bands_avg_0_2)
feature_all_bands_avg_1_2 = changeDimension(feature_all_bands_avg_1_2)
feature_all_bands_avg_2_2 = changeDimension(feature_all_bands_avg_2_2)
feature_all_bands_avg_3_2 = changeDimension(feature_all_bands_avg_3_2)
feature_all_bands_avg_4_2 = changeDimension(feature_all_bands_avg_4_2)
feature_all_bands_avg_5_2 = changeDimension(feature_all_bands_avg_5_2)
feature_all_bands_avg_6_2 = changeDimension(feature_all_bands_avg_6_2)
feature_all_bands_avg_7_2 = changeDimension(feature_all_bands_avg_7_2)
feature_all_bands_avg_8_2 = changeDimension(feature_all_bands_avg_8_2)
feature_all_bands_avg_9_2 = changeDimension(feature_all_bands_avg_9_2)

# concaténée des 22 champs 
# truth reste à part 

data_2=np.column_stack((feature_all_bands_2,feature_MSI_2,feature_all_bands_avg_0_2,feature_all_bands_avg_1_2,feature_all_bands_avg_2_2,feature_all_bands_avg_3_2,feature_all_bands_avg_4_2,feature_all_bands_avg_5_2,feature_all_bands_avg_6_2,feature_all_bands_avg_7_2,feature_all_bands_avg_8_2,feature_all_bands_avg_9_2,feature_MSI_avg_2))

# on concatène les 2 datasets

data=np.concatenate((data_1, data_2))
truth=np.concatenate((truth_1, truth_2))


# decoupage en train test 
xTrain, xTest, yTrain, yTest = train_test_split(data, truth, test_size=0.4, random_state=42)
print(xTrain.shape)
print(yTrain.shape)

print(xTest.shape)
print(yTest.shape)


#yTrain=yTrain-1
#yTest=yTest-1

# reshape


#test model 





# NEURAL NETWORK
#from tensorflow import keras

#xTrain = xTrain.reshape((xTrain.shape[0], 1, xTrain.shape[1]))
#xTest = xTest.reshape((xTest.shape[0], 1, xTest.shape[1]))
#data = data.reshape((data.shape[0], 1, data.shape[1]))

#nBands=22


# Define the parameters of the model
#model = keras.Sequential([
   # keras.layers.Flatten(input_shape=(1, nBands)),
   # keras.layers.Dense(14, activation='relu'),
   # keras.layers.Dense(4, activation='softmax')])

# Define the accuracy metrics and parameters
#model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Run the model
#model.fit(xTrain, yTrain, epochs=2)
#history=model.fit(xTrain, yTrain,validation_split=0.33, epochs=100)

# summarize history for accuracy
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
# summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
   
   
#LOGISTIC REGRESSION

#from sklearn.linear_model import LogisticRegression
#model = LogisticRegression()
#model.fit(xTrain, yTrain)
#print('Accuracy of Logistic regression classifier on training set: {:.2f}'
#     .format(model.score(xTrain, yTrain)))
#print('Accuracy of Logistic regression classifier on test set: {:.2f}'
#     .format(model.score(xTest, yTest)))


## DECISION TREE

#from sklearn.tree import DecisionTreeClassifier
#model = DecisionTreeClassifier().fit(xTrain, yTrain)
#print('Accuracy of Decision Tree classifier on training set: {:.2f}'
#     .format(model.score(xTrain, yTrain)))
#print('Accuracy of Decision Tree classifier on test set: {:.2f}'
#     .format(model.score(xTest, yTest)))


# test de la précision
#from sklearn.metrics import confusion_matrix, precision_score, recall_score

# Predict for test data - without filter


#yTestPredicted = model.predict(xTest)



# Random forest 

# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 20 decision trees

model=RandomForestClassifier(n_estimators = 20, random_state = 42)
model.fit(xTrain, yTrain)



#yTestPredicted = yTestPredicted[:,1]

# Calculate and display the error metrics
#yTestPredicted = (yTestPredicted>0.5).astype(int)
#yTestPredicted=np.argmax(yTestPredicted, axis=1)


yTestPredicted = model.predict(xTest)



# test de la précision
from sklearn.metrics import confusion_matrix, precision_score, recall_score

cMatrix = confusion_matrix(yTest, yTestPredicted)
pScore = precision_score(yTest, yTestPredicted,average='micro')
rScore = recall_score(yTest, yTestPredicted,average='micro')

print("Confusion matrix: for 14 nodes\n", cMatrix)
print("\nP-Score: %.3f, R-Score: %.3f" % (pScore, rScore))
    

# export des resultats

predicted_1 = model.predict(data_1)
predicted_2 = model.predict(data_2)
# ici on a la proba pour chacune des caleurs possibles, on veut retourner la valeur avec laplus haute proba

#MEDIAN FILTER
#on applique un filtre sur limage 2d puis on teste au global contre la realité


prediction_1 = np.reshape(predicted_1, (ds1.RasterYSize, ds1.RasterXSize))
prediction_2 = np.reshape(predicted_2, (ds1.RasterYSize, ds1.RasterXSize))


#predicted_filter=scipy.signal.medfilt2d(prediction,kernel_size=3)


#predicted_filter = changeDimension(predicted_filter)



cMatrix = confusion_matrix(truth_1, predicted_1)
pScore = precision_score(truth_1, predicted_1,average='micro')
rScore = recall_score(truth_1, predicted_1,average='micro')

print("2017 precision ")
print("Confusion matrix: sur 2017," ,cMatrix)
print("\nP-Score: %.3f, R-Score: %.3f" % (pScore, rScore))

#--- 

cMatrix = confusion_matrix(truth_2, predicted_2)
pScore = precision_score(truth_2, predicted_2,average='micro')
rScore = recall_score(truth_2, predicted_2,average='micro')

print("2015 precision ")
print("Confusion matrix: sur 2015," ,cMatrix)
print("\nP-Score: %.3f, R-Score: %.3f" % (pScore, rScore))


#predicted=np.argmax(predicted, axis=1)

#print("distribution de predicted")
#unique, counts = np.unique(predicted, return_counts=True)
#print(np.asarray((unique, counts/predicted.size)).T)

#print("distribution de predicted filter")
#unique, counts = np.unique(predicted_filter, return_counts=True)
#print(np.asarray((unique, counts/predicted_filter.size)).T)


#Export raster
#prediction = np.reshape(predicted, (ds1.RasterYSize, ds1.RasterXSize))
outFile_1 = 'data_predicted_v24_randomforest_nofilter_Seredou_2017.tif'
raster.export(prediction_1,ds1, filename=outFile_1, dtype='float')

#◄prediction_filter = np.reshape(predicted_filter, (ds1.RasterYSize, ds1.RasterXSize-1))
outFile_2 = 'data_predicted_v24_randomforest_nofilter_Seredou_2015.tif'
raster.export(prediction_2, ds1, filename=outFile_2, dtype='float')

#enregistrement modele
from joblib import dump,load
dump(model, "randomforest_v24_from_seredou2015&2017.sav")

#-----------------------
## test sur DIECKE 2017

#from joblib import dump,load
#model=load("random_forest_median_filter_v22_from_seredou2015&2017.sav")

MSI_test = 'diecke_20170205_MSI.tif'
all_bands_test='diecke_20170205_allbands.tif'
truth_test='diecke_20170205_truth.tif'

# Read the rasters as array
ds1_test, feature_MSI_test = raster.read(MSI_test, bands='all')
ds2_test, feature_all_bands_test = raster.read(all_bands_test, bands='all')
ds3_test, truth_test = raster.read(truth_test, bands='all')


#truth_test=np.delete(truth_test,4664,0)
#truth_test=np.delete(truth_test,0,1)


# on remaplce les 4 par des 3 dans le truth -- cas 32 bits uniquement
#truth_test[truth_test==4]=3


# sur all bands
#feature_all_bands_test=np.delete(feature_all_bands_test,np.s_[8951:9120],axis=2)



print("-------------------")

print("Max pour truth:", truth_test.max() )
print("Min pour truth:", truth_test.min() )
print("Max pour all bands", feature_all_bands_test.max() )
print("Min pour all bands", feature_all_bands_test.min() )
print("Max pour MSI", feature_MSI_test.max() )
print("Min pour MSI", feature_MSI_test.min() )

print("-------------------")

print("distribution de truth ")
unique, counts = np.unique(truth_test, return_counts=True)
print(np.asarray((unique, counts/truth_test.size)).T)

print("-------------------")

feature_MSI_test[feature_MSI_test<0]=0
feature_all_bands_test[feature_all_bands_test<0]=0

feature_MSI_test = feature_MSI_test / feature_MSI_test.max()

i=0
while i<10:
    feature_all_bands_test[i]=feature_all_bands_test[i]/feature_all_bands_test[i].max()
    i=i+1

# on vérifie que ça a marché

print("Max pour MSI", feature_MSI_test.max() )
print("Min pour MSI", feature_MSI_test.min() )    
print("-------------------")

i=0
while i<10:    
    print("Max pour all bands ",i,": ", feature_all_bands_test[i].max() )
    print("Min pour all bands ",i,": ", feature_all_bands_test[i].min() )
    i=i+1
    print("-------------------")
    




feature_MSI_avg_test = (conv2(feature_MSI_test,np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_0_test=(conv2(feature_all_bands_test[0],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_1_test=(conv2(feature_all_bands_test[1],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_2_test=(conv2(feature_all_bands_test[2],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_3_test=(conv2(feature_all_bands_test[3],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_4_test=(conv2(feature_all_bands_test[4],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_5_test=(conv2(feature_all_bands_test[5],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_6_test=(conv2(feature_all_bands_test[6],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_7_test=(conv2(feature_all_bands_test[7],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_8_test=(conv2(feature_all_bands_test[8],np.ones((13,13)),boundary='symm',mode='same')/169)
feature_all_bands_avg_9_test=(conv2(feature_all_bands_test[9],np.ones((13,13)),boundary='symm',mode='same')/169)


# on passe en tabulaire

feature_all_bands_test = changeDimension(feature_all_bands_test)
feature_MSI_test = changeDimension(feature_MSI_test)
truth_test= changeDimension(truth_test)
feature_MSI_avg_test = changeDimension(feature_MSI_avg_test)
feature_all_bands_avg_0_test = changeDimension(feature_all_bands_avg_0_test)
feature_all_bands_avg_1_test = changeDimension(feature_all_bands_avg_1_test)
feature_all_bands_avg_2_test = changeDimension(feature_all_bands_avg_2_test)
feature_all_bands_avg_3_test = changeDimension(feature_all_bands_avg_3_test)
feature_all_bands_avg_4_test = changeDimension(feature_all_bands_avg_4_test)
feature_all_bands_avg_5_test = changeDimension(feature_all_bands_avg_5_test)
feature_all_bands_avg_6_test = changeDimension(feature_all_bands_avg_6_test)
feature_all_bands_avg_7_test = changeDimension(feature_all_bands_avg_7_test)
feature_all_bands_avg_8_test = changeDimension(feature_all_bands_avg_8_test)
feature_all_bands_avg_9_test = changeDimension(feature_all_bands_avg_9_test)

# concaténée des 22 champs 
# truth reste à part 

data_test=np.column_stack((feature_all_bands_test,feature_MSI_test,feature_all_bands_avg_0_test,feature_all_bands_avg_1_test,feature_all_bands_avg_2_test,feature_all_bands_avg_3_test,feature_all_bands_avg_4_test,feature_all_bands_avg_5_test,feature_all_bands_avg_6_test,feature_all_bands_avg_7_test,feature_all_bands_avg_8_test,feature_all_bands_avg_9_test,feature_MSI_avg_test))

predicted_test= model.predict(data_test)
prediction_test = np.reshape(predicted_test, (ds1_test.RasterYSize, ds1_test.RasterXSize))
#prediction_test_filter=scipy.signal.medfilt2d(prediction_test,kernel_size=5)

from sklearn.metrics import confusion_matrix, precision_score, recall_score

cMatrix = confusion_matrix(truth_test, predicted_test)
pScore = precision_score(truth_test, predicted_test,average='micro')
rScore = recall_score(truth_test, predicted_test,average='micro')

print(" test sur Diecke 2017")
print("Confusion matrix: sans filter", cMatrix)
print("\nP-Score: %.3f, R-Score: %.3f" % (pScore, rScore))



#cMatrix = confusion_matrix(truth_test, predict_test_filter)
#pScore = precision_score(truth_test, predicted_test_filter,average='micro')
#rScore = recall_score(truth_test, predicted_test_filter,average='micro')

#print(" test sur Seredou 2015 avec filtre ")
#print("Confusion matrix: after filter", cMatrix)
#print("\nP-Score: %.3f, R-Score: %.3f" % (pScore, rScore))



outFile = 'data_predicted_v24_randomforest_nofilter_Diecke 2017.tif'
raster.export(prediction_test, ds3_test, filename=outFile, dtype='float')