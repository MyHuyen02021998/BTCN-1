import pandas as pd
import numpy as np

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def read_file(file_train, file_test):

	df_train = pd.read_csv("train.csv")
	df_test = pd.read_csv("test.csv")
	print(df_train.shape)
	print(df_test.shape)

	return df_train, df_test

def xu_ly_data(df_train):
	# Thực hiện chuẩn hóa dữ liệu về đoạn [0, 1]
	X = df_train.drop('label', axis = 1)/255
	# Lấy cột nhãn
	Y = to_categorical(df_train['label'])

	# Chia dữ liệu thành 80% train và 20% test
	X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=101)

	# Chuyển dữ liệu về thành ma trận để tiện thực hiện
	X_train = np.array(X_train).T
	Y_train = Y_train.T

	X_test = np.array(X_test).T
	Y_test = Y_test.T

	return X_train, Y_train, X_test, Y_test

def initial(input_nodes, hidden_nodes, output_nodes):
	#Khởi tạo tham số
	W1 = 0.01*np.random.randn(input_nodes, hidden_nodes)   # tham số giữa lớp input và lớp ẩn (784, 100)
	b1 = np.zeros((hidden_nodes, 1))                       # Bias giữa lớp input và lớp ẩn (100, 1)
	W2 = 0.01*np.random.randn(hidden_nodes, output_nodes)  # tham số giữa lớp ẩn và lớp output (100, 10)
	b2 = np.zeros((output_nodes, 1))                       # Bias giữa lớp ẩn và lớp output (10, 1)

	return W1, b1, W2, b2


def softmax(V):
    e_V = np.exp(V)
    Z = e_V / e_V.sum(axis = 0)
    return Z

def cost(Y, Yhat):
    return -np.sum(Y*np.log(Yhat))/Y.shape[1]


def run(X_train, Y_train):
	# Tạo các tham số
	input_nodes = X_train.shape[0]  # Số chiều dữ liệu
	hidden_nodes =  100             # Kích thước hidden layer
	output_nodes = C = 10           # Số Class
	N = X_train.shape[1]            # Kích thước tập huấn luyện
	eta = 1                         # learning rate

	W1, b1, W2, b2 = initial(input_nodes, hidden_nodes, output_nodes)

	for i in range(1000):
	    ## feedforward
	    Z1 = np.dot(W1.T, X_train) + b1
	    A1 = np.maximum(Z1, 0)
	    Z2 = np.dot(W2.T, A1) + b2
	    Yhat = softmax(Z2)
	    
	    ## In hàm lỗi sau 100 vòng lặp
	    if i % 100 == 0:
	         # Tính lỗi: trung bình cross-entropy 
	        loss = cost(Y_train, Yhat)
	        print("iter %d, loss: %f" %(i, loss))
	        
	    ## Backpropagation
	    E2 = (Yhat - Y_train) / N
	    dW2 = np.dot(A1, E2.T)
	    db2 = np.sum(E2, axis = 1, keepdims = True)
	    E1 = np.dot(W2, E2)
	    E1[Z1 <= 0] = 0 # gradient of ReLU
	    dW1 = np.dot(X_train, E1.T)
	    db1 = np.sum(E1, axis = 1, keepdims = True)
	    
	    W1 += -eta*dW1
	    b1 += -eta*db1
	    W2 += -eta*dW2
	    b2 += -eta*db2

	return W1, b1, W2, b2

def Test(X_test, Y_test):
	Z1 = np.dot(W1.T, X_test) + b1
	A1 = np.maximum(Z1, 0)
	Z2 = np.dot(W2.T, A1) + b2

	Yhat = softmax(Z2)
	predicted_class = np.argmax(Yhat, axis=0)
	Y = np.argmax(Y_test, axis=0)
	print('test accuracy: %.2f %%' % (100*np.mean(predicted_class == Y)))

if __name__ == "__main__":
	file_train = file_test = argv
	df_train, df_test = read_file(file_train, file_test)
	X_train, Y_train, X_test, Y_test = xu_ly_data(df_train)
	W1, b1, W2, b2 = run(X_train, Y_train)
	Test(X_test, Y_test)




