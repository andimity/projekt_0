from sklearn import linear_model
import numpy as np

#getting the data from the csv files
def get_data(name,x_start,x_end):

    dataset = np.loadtxt(name,delimiter=",",skiprows=1)
    return dataset[:,x_start:x_end]




if __name__ == '__main__':
    x = get_data('Data/train.csv',2,12)
    y = get_data('Data/train.csv',1,2)
    reg = linear_model.LinearRegression()
    reg.fit(x,y)
    print(reg.coef_)

