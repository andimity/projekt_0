from sklearn import linear_model
import numpy as np

dataset = np.loadtxt('Data/train.csv',delimiter=",",comments="#")
print(dataset[1:2,:])
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Anmo')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
