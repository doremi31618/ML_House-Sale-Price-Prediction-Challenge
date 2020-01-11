

# Section 1 : import libary
#############################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import matplotlib as mpl

from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense, Activation, Input

from sklearn import preprocessing


#Section 2 read data set 
#############################################################################
#read train data 
data_1 = pd.read_csv('./machine-learning-realtek-regression/train-v3.csv')
X_train = data_1.drop(['price','id','sale_yr','sale_month','sale_day'],axis=1).values
y_train = data_1['price'].values

#read valid data
data_2 = pd.read_csv('./machine-learning-realtek-regression/valid-v3.csv')
X_valid = data_2.drop(['price','id','sale_yr','sale_month','sale_day'],axis=1).values
y_valid = data_2['price'].values

#read test data
data_3 = pd.read_csv('./machine-learning-realtek-regression/test-v3.csv')
X_test = data_3.drop(['id','sale_yr','sale_month','sale_day',],axis=1).values


#Section3 data scaling 
#############################################################################
X_train = preprocessing.scale(X_train)

X_valid = preprocessing.scale(X_valid)

X_test  = preprocessing.scale(X_test)


#random setting 
#############################################################################
'''
np.random.seed(123)
np.random.shuffle(X_train)
'''
#Section4 training model
#############################################################################
#setting train-model
def build_model() : 
    model = Sequential()

    #add hidden layer with relu activation function 
    #kernel_initializer='Ones',
    model.add(Dense(units = 32,input_dim=X_train.shape[1],activation='relu'))
    model.add(Dense(units = 128,activation='relu'))
    model.add(Dense(units = 128,activation='relu'))
    model.add(Dense(units = 32,activation='relu'))
    model.add(Dense(X_train.shape[1],activation='relu'))
    
    #output layer
    model.add(Dense(1))
    model.compile(loss='mae',optimizer='adam')
    
    return model

#train only one times 

#cross validation
#############################################################################
k = 4
nb_val_samples = len(X_train)//k #每一折樣本數
epochs = 200
batch_size = 50#every bacth number 

for i in range(k):
    print("processing Fold #" + str(i))
    print((i+1)*nb_val_samples)
    X_val = X_train[i*nb_val_samples: (i+1)*nb_val_samples]
    Y_val = y_train[i*nb_val_samples: (i+1)*nb_val_samples]
    
    X_train_p = np.concatenate(
            [X_train[:i*nb_val_samples],
             X_train[(i+1)*nb_val_samples:]], axis = 0)
    
    Y_train_p = np.concatenate(
            [y_train[:i*nb_val_samples],
             y_train[(i+1) * nb_val_samples:]], axis = 0)
    
    print(X_train_p)
    print(Y_train_p)
    model = build_model()
    model.fit(X_train_p,Y_train_p,batch_size=batch_size,epochs=epochs,verbose=1)
    '''
       
    a,b=model.evaluate(X_train_p,Y_train_p)
    print(b)
    TB = TensorBoard(log_dir = 'logs/' + fn,
                     histogram_freq = 0)
    '''

    
    '''
              ,
              validation_data=(X_valid,y_valid),
              callbacks=[TB]
              loss,accuracy = model.evaluate(X_val,Y_val)
    print("準確度 = {:.2f}".format(accuracy))
    '''
    
#create a new string object
fn = str(epochs) + '_' + str(batch_size)
model.save(fn + '.h5')

y_predict = model.predict(X_train)
y_predict = np.insert(y_predict, 0, values=data_3["id"].values.astype(int),axis=1)
np.savetxt('test_first_try_5.csv' ,y_predict,delimiter=',',header="id,price")


y_preview = pd.read_csv('./test_first_try_5.csv')




