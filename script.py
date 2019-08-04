import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    
    k=0;
    df1=[];
    df2=[];
    df3=[];
    df4=[];
    df5=[];
    df = np.concatenate((X,y), axis=1);
    for i in df[:,2]:
        if(i == 1):
            df1.append(df[k,:]);
        if(i == 2):
            df2.append(df[k,:]);
        if(i == 3):
            df3.append(df[k,:]);
        if(i == 4):
            df4.append(df[k,:]);
        if(i == 5):
            df5.append(df[k,:]);
        k=k+1;
    means=[]
    means.append(np.mean(df1,axis=0))
    means.append(np.mean(df2,axis=0))
    means.append(np.mean(df3,axis=0))
    means.append(np.mean(df4,axis=0))
    means.append(np.mean(df5,axis=0))
    df_t = np.transpose(df)
    covmat = np.cov(df_t[0:2,:])
        
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    k=0;
    df1=[];
    df2=[];
    df3=[];
    df4=[];
    df5=[];
    df = np.concatenate((X,y), axis=1);
    for i in df[:,2]:
        if(i == 1):
            df1.append(df[k,:]);
        if(i == 2):
            df2.append(df[k,:]);
        if(i == 3):
            df3.append(df[k,:]);
        if(i == 4):
            df4.append(df[k,:]);
        if(i == 5):
            df5.append(df[k,:]);
        k=k+1;
    means=[]
    means.append(np.mean(df1,axis=0))
    means.append(np.mean(df2,axis=0))
    means.append(np.mean(df3,axis=0))
    means.append(np.mean(df4,axis=0))
    means.append(np.mean(df5,axis=0))
    df_t1 = np.transpose(df1);
    df_t2 = np.transpose(df2);
    df_t3 = np.transpose(df3);
    df_t4 = np.transpose(df4);
    df_t5 = np.transpose(df5);
    covmats=[];
    covmats.append(np.cov(df_t1[0:2,:]))
    covmats.append(np.cov(df_t2[0:2,:]))
    covmats.append(np.cov(df_t3[0:2,:]))
    covmats.append(np.cov(df_t4[0:2,:]))
    covmats.append(np.cov(df_t5[0:2,:]))
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    X_norm1 = Xtest - means[0][0:2]
    X_norm2 = Xtest - means[1][0:2]
    X_norm3 = Xtest - means[2][0:2]
    X_norm4 = Xtest - means[3][0:2]
    X_norm5 = Xtest - means[4][0:2]
    
    covmat_inv = np.linalg.inv(covmat);
    
    pred1=[];
    for i in X_norm1:
        pred1.append(np.dot(np.dot(np.transpose(i),covmat_inv),i))
    pred2=[];
    for i in X_norm2:
        pred2.append(np.dot(np.dot(np.transpose(i),covmat_inv),i))
    pred3=[];
    for i in X_norm3:
        pred3.append(np.dot(np.dot(np.transpose(i),covmat_inv),i))
    pred4=[];
    for i in X_norm4:
        pred4.append(np.dot(np.dot(np.transpose(i),covmat_inv),i))
    pred5=[];
    for i in X_norm5:
        pred5.append(np.dot(np.dot(np.transpose(i),covmat_inv),i))
        
    pred =[];
    pred.append(pred1);
    pred.append(pred2);
    pred.append(pred3);
    pred.append(pred4);
    pred.append(pred5);
    
    ypred =[];
    for i in range(0,len(ytest)):
        for j in range(0,5):
            if pred[j][i] == np.amin([pred[0][i],pred[1][i],pred[2][i],pred[3][i],pred[4][i]]):
                ypred.append(j+1);
    count=0;
    for i in range(0,len(ytest)):
        if ypred[i]-ytest[i][0]==0:
            count=count+1;
                
    acc=count/len(ytest)
    # ypred - N x 1 column vector indicating the predicted labels
    ypred=np.array(ypred);
    # IMPLEMENT THIS METHOD
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    
    # Outputs
    X_norm1 = Xtest - means[0][0:2]
    X_norm2 = Xtest - means[1][0:2]
    X_norm3 = Xtest - means[2][0:2]
    X_norm4 = Xtest - means[3][0:2]
    X_norm5 = Xtest - means[4][0:2]
    
    covmat_inv1 = np.linalg.inv(covmats[0])
    covmat_inv2 = np.linalg.inv(covmats[1])
    covmat_inv3 = np.linalg.inv(covmats[2])
    covmat_inv4 = np.linalg.inv(covmats[3])
    covmat_inv5 = np.linalg.inv(covmats[4])
    
    pred1=[];
    for i in X_norm1:
        pred1.append(np.dot(np.dot(np.transpose(i),covmat_inv1),i))
    pred2=[];
    for i in X_norm2:
        pred2.append(np.dot(np.dot(np.transpose(i),covmat_inv2),i))
    pred3=[];
    for i in X_norm3:
        pred3.append(np.dot(np.dot(np.transpose(i),covmat_inv3),i))
    pred4=[];
    for i in X_norm4:
        pred4.append(np.dot(np.dot(np.transpose(i),covmat_inv4),i))
    pred5=[];
    for i in X_norm5:
        pred5.append(np.dot(np.dot(np.transpose(i),covmat_inv5),i))
        
    pred =[];
    pred.append(pred1);
    pred.append(pred2);
    pred.append(pred3);
    pred.append(pred4);
    pred.append(pred5);
    
    ypred =[];
    for i in range(0,len(ytest)):
        for j in range(0,5):
            if pred[j][i] == np.amin([pred[0][i],pred[1][i],pred[2][i],pred[3][i],pred[4][i]]):
                ypred.append(j+1);
    count=0;
    for i in range(0,len(ytest)):
        if ypred[i]-ytest[i][0]==0:
            count=count+1;
                
    acc=count/len(ytest)
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels
    ypred=np.array(ypred);
    # IMPLEMENT THIS METHOD
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
    w = np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),np.dot(np.transpose(X),y));
    # IMPLEMENT THIS METHOD                                                   
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD  
    
    w = np.dot((np.linalg.inv(np.add(np.dot(np.transpose(X),X), lambd*np.eye(np.shape(X)[1])))),np.dot(np.transpose(X),y));
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    mse = sum((np.dot(Xtest,w) - ytest)**2)/len(ytest);
    # IMPLEMENT THIS METHOD
    return mse

def regressionObjVal(w, X, y, lambd):
    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda
    w = np.asmatrix(w)
    w = w.transpose()
    error = 0.5 * sum(np.dot(np.transpose((y - np.dot(X,w))),(y - np.dot(X,w))),lambd*np.dot(np.transpose(w),w));
    error_grad = np.subtract(lambd*w, np.dot(np.transpose(X),np.subtract(y,np.dot(X,w))));
    error_grad = np.squeeze(np.array(error_grad))
    # IMPLEMENT THIS METHOD                                             
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
    
    Xp = np.zeros((x.shape[0],p+1))
    for i in range(0,p+1):
        Xp[:,i] = pow(x,i)
    # IMPLEMENT THIS METHOD
    return Xp

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

yt=[]
for i in range(0,len(ytest)):
    yt.append(ytest[i][0])

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=yt)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=yt)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
lambda_min_position = np.argmin(mses3)
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(w_i)
plt.title('Weights for LearnOLERegression')
plt.subplot(1, 2, 2)
plt.plot(w_l)
plt.title('Weights for LearnRidgeRegression')
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])


# Problem 5
pmax = 7
lambda_opt = lambdas[lambda_min_position] # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))

plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
