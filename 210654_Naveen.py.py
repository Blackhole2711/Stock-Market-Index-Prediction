import pandas as pd
import numpy as np
import datetime
import sklearn
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import keras




def evaluate():
    # Input the csv file
    """
    Sample evaluation function
    Don't modify this function
    """
    df = pd.read_csv('sample_input.csv')
     
    actual_close = np.loadtxt('sample_close.txt')
    
    pred_close = predict_func(df)
    
    # Calculation of squared_error
    actual_close = np.array(actual_close)
    pred_close = np.array(pred_close)
    mean_square_error = np.mean(np.square(actual_close-pred_close))


    pred_prev = [df['Close'].iloc[-1]]
    pred_prev.append(pred_close[0])
    pred_curr = pred_close
    
    actual_prev = [df['Close'].iloc[-1]]
    actual_prev.append(actual_close[0])
    actual_curr = actual_close

    # Calculation of directional_accuracy
    pred_dir = np.array(pred_curr)-np.array(pred_prev)
    actual_dir = np.array(actual_curr)-np.array(actual_prev)
    dir_accuracy = np.mean((pred_dir*actual_dir)>0)*100

    print(f'Mean Square Error: {mean_square_error:.6f}\nDirectional Accuracy: {dir_accuracy:.1f}')
    



def predict_func(df):
    """
    Modify this function to predict closing prices for next 2 samples.
    Take care of null values in the sample_input.csv file which are listed as NAN in the dataframe passed to you 
    Args:
        data (pandas Dataframe): contains the 50 continuous time series values for a stock index

    Returns:
        list (2 values): your prediction for closing price of next 2 samples
    """
    #Now i am uploading the lstm model which had minimum mse among all the lstm and bi-directional lstm models i created.
    model = tf.keras.models.load_model('best_lstm_model.h5')
    #model.summary()
    #Converting the loaded data(df in this case) a Data Frame in Pandas
    
    df = pd.DataFrame(df)
    
    """"
    Considering only the close column as we have to only predict the closing price for the next two days 
    and all the columns are independent of each other
    """
    df = df[['Close']]
    
    #Converting the index to datetime so pandas can easily understand the data.
    df.index = pd.to_datetime(df.index, format = '%Y-%m-%d')
    
    #Interpolating the nan values by cubic interpolation.
    #In the code where i created the model i tried various types of interpolation such as linear, cubic and spline interpolation.
    #I aslo tried iterative imputer to impute the nan values.
    #But the best results were from cubic interpolation so i used it here.
    df = df.interpolate(method = 'cubic')
    df_values = df.values
    
    #Scaline the values of close column between 0 and 1 as LSTM models are sensotive to scale.
    scaler = MinMaxScaler()
    scaler.fit(df_values)
    df_scaled_values = scaler.transform(df_values)
    
    #Storing the scales values from the given dataset and converting it to a 3 dimensional array.
    #Because LSTM models take input as 3d array with parameters as follows :
    #First batch_size which tells about the size of the batch which is 1 in our case as we have only one dataset(data from last 50 days)
    #Second timestep which is 50 in our case as our LSTM model takes input as the data of past 50 days and predict the value for next day
    #Last is n_features which is 1 as we are only predicting for closing price
    
    x_test = list()
    x_test.append(df_scaled_values)
    x_test = np.array(x_test)
    
    #Predicting the next first day's closing price. 
    pred_price = model.predict(x_test)
    pred_price = scaler.inverse_transform(pred_price)
    
    #Printing first predicted price.
    print(pred_price)
    Price1 = np.reshape(pred_price, (1,))
    
    #Now scaling the predicted value between 0 and 1 and adding it to the list of scaled values
    pred_price = scaler.transform(pred_price)
    df_scaled_values = np.concatenate((df_scaled_values, pred_price))
    
    #Doing prediction for the next second day's closing price.
    x_test = list()
    
    #Here i gave the input to my model as last 49 day's data and one form the previous predicted data for next day's prediction.
    #Because our model works on timesteps = 50.
    x_test.append(df_scaled_values[1:])
    x_test = np.array(x_test)
    pred_price = model.predict(x_test)
    pred_price = scaler.inverse_transform(pred_price)
    
    #Printing second predicted price
    print(pred_price)
    Price2 = np.reshape(pred_price, (1,))
    
    #Storing the predicted prices in a list.
    ans = [Price1[0], Price2[0]]
    
    return ans



if __name__== "__main__":
    evaluate()

