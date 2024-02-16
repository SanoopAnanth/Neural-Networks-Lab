import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
class NeuralNetwork:
    def __init__(self,input_size,hidden_size,output_size):
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        
        self.W1=np.random.randn(hidden_size,input_size)*0.01
        self.b1=np.zeros((hidden_size,1))
        self.W2=np.random.randn(output_size,hidden_size)*0.01
        self.b2=np.zeros((output_size,1))
        
    def forward_propogation(self,X):
        self.Z1=np.dot(self.W1,X)+self.b1
        self.A1=np.tanh(self.Z1)
        self.Z2=np.dot(self.W2,self.A1)+self.b2
        self.A2=self.sigmoid(self.Z2)
        return self.A2
    
    def back_propagation(self,X,Y):
        m=X.shape[1]
        dZ2=self.A2-Y
        dW2=(1/m)*np.dot(dZ2,self.A1.T)
        db2=(1/m)*np.sum(dZ2,axis=1,keepdims=True)
        dZ1=np.dot(self.W2.T, dZ2)*(1-np.power(self.A1,2))
        dW1=(1/m)*np.dot(dZ1,X.T)
        db1=(1/m)*np.sum(dZ1,axis=1,keepdims=True)
        self.W2-=dW2
        self.b2-=db2
        self.W1-=dW1
        self.b1-=db1
        
    def train(self,X,Y,num_epochs,learning_rate):
        for epochs in range(num_epochs):
            predictions=self.forward_propogation(X)
            loss=self.cross_entropy_loss(Y,predictions)
            self.back_propagation(X,Y)
            if epochs%100==0:
                print(f"Epoch {epochs}, loss: {loss}")
                
    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z))
    
    def cross_entropy_loss(self,Y,A):
        m=Y.shape[1]
        return -(1/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
  X,Y=make_moons(n_samples=1000,noise=0.2,random_state=42)
X=X.T
Y=Y.reshape(1,-1)
plt.scatter(X[0,:],X[1,:],c=Y.ravel(),cmap=plt.cm.Spectral)
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.title("planar data classification")
plt.show()
input_size=2
hidden_size=4
output_size=1
model=NeuralNetwork(input_size,hidden_size,output_size)
model.train(X,Y,num_epochs=1000,learning_rate=0.01)
