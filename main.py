from sklearn import linear_model
import numpy as np
import pandas as pd

#getting the data from the csv files
def get_data(name,x_start,x_end):

    dataset = np.loadtxt(name,delimiter=",",skiprows=1)
    return dataset[:,x_start:x_end]

def write_data(name,data):
    pd.DataFrame(data).to_csv(name,header=["y"],index_label='Id')



if __name__ == '__main__':
    x = get_data('Data/train.csv',2,12)
    y = get_data('Data/train.csv',1,2)
    reg = linear_model.LinearRegression()
    reg.fit(x,y)

    x_test = get_data('Data/test.csv',1,11)
    y_pred = reg.predict(x_test)
    write_data("Data/prediction.csv",y_pred)


