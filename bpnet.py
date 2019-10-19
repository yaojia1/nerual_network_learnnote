import numpy as np
#import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import scipy.io as sio
import matplotlib.pyplot as plt

# matlab文件名
matfn = u'E:/matlabproj/spectra_data.mat'
data = sio.loadmat(matfn)
plt.close('all')
xi = data['NIR']
yi = data['octane']
xi=np.array(xi)
yi=np.array(yi)
print(xi.shape,yi.shape)



class bpnet(object):
    def __init__(self,inputlayer,outputlayer,learnrate,hidelayer):#从一开始
        self.inputlayer=inputlayer
        self.outputlayer=outputlayer
        self.learnrate=learnrate
        self.hidelayer=hidelayer
        self.ekout = np.zeros((self.outputlayer, self.inputlayer))
    def addlayer(self):
        self.weight=np.random.rand(self.hidelayer,self.inputlayer,self.inputlayer)#, dtype = float, order = 'C')#all weight,and hight=input highth每行是一个神经元
        print("weight",self.weight)
        self.bias=np.random.rand(self.hidelayer+1)#bias
        self.outweight=np.random.rand(self.outputlayer,self.inputlayer)#, dtype = float, order = 'C')
        self.__X=np.zeros((self.hidelayer,self.inputlayer),dtype = float, order = 'C')#中间的x=weight*input+bias
        self.active=np.zeros((self.hidelayer,self.inputlayer),dtype = float, order = 'C')#激活值a
    def sigmoid(self,x):
        s = 1 / (1 + np.exp(-x))
        return s
    def sigmoidprime(self,x):
        return np.multiply(x,(1-x))
    def relu(self,x):
        s = np.where(x < 0, 0, x)
        return s
    def ReLuPrime(self,x):
        # ReLu 导数
        #x[x <= 0] = 0
        x[x > 0] = 1
        return x
    def forward(self,inputnum):
        for i in range(self.hidelayer):#每一层的激活值存在active[i]里
            #print('input',inputnum,self.weight[i],self.bias[i])
            self.__X[i]=np.matmul(self.weight[i],inputnum)+self.bias[i]#
            #print('X', i + 1, '层', self.__X[i])
            self.active[i]=self.relu(self.__X[i])#self.sigmoid(self.__X[i])#1 / (1 + np.exp(-self.__X[i]))#激活值
            #print('active',i+1,'层',self.active[i])
            inputnum=self.active[i]#输入值得传递
        outputnum=np.matmul(self.outweight,self.active[self.hidelayer-1])+self.bias[self.hidelayer]
        return outputnum
    def backforward(self,inputnum,outputParameter,outputnum):
        #print("11212121212121111",self.ekout,outputParameter-outputnum,(outputParameter-outputnum)*self.outweight)
        #self.ekout=np.zeros((self.outputlayer,self.inputlayer))
        '''
        for k in range(self.outputlayer):
            self.ek[k] = outputParameter[k] - outputnum[k]'''
        '''算误差'''
        self.ekout=(outputParameter-outputnum)*self.outweight
        #print("output err",self.ekout)
        self.ekhide=np.zeros((self.hidelayer,self.inputlayer))
        lastek=self.ekout[:]
        for i in range(self.hidelayer):
            self.ekhide[self.hidelayer-1-i]=np.matmul(self.weight[self.hidelayer-1-i],np.ones((self.inputlayer)))*lastek
            '''print("误差值计算=",self.weight[self.hidelayer-1-i],"x",np.ones((self.inputlayer)),"=",np.matmul(self.weight[self.hidelayer-1-i],np.ones((self.inputlayer))))
            print(self.ekhide[self.hidelayer - 1 - i], "=",
                  np.matmul(self.weight[self.hidelayer - 1 - i], np.ones((self.inputlayer))), "*", lastek)'''
            lastek=self.ekhide[self.hidelayer-1-i][:]
        #print("ekhide:",self.ekhide)
        lastek=np.vstack((self.ekhide,self.ekout))
        #print("eek:",lastek)
        #self.J=np.sum(np.dot(self.ek,self.ek)/2.0)
        '''算隐含层梯度'''
        '''这个喵？'''
        miao=np.vstack((inputnum,self.active[:self.hidelayer-1]))#,)
        dweight0=self.learnrate*self.ekhide*self.active*self.ReLuPrime(self.active)#self.sigmoidprime(self.active)#(1-self.active)#乘输入值
        self.dweight=np.ones((self.hidelayer,self.inputlayer,self.inputlayer))
        #print('miao:',miao,self.dweight[0].shape)
        #print('active:',self.active[:self.hidelayer-1])
        #print('input',inputnum)
        for k in range(self.hidelayer):
            kk=dweight0[k].reshape(self.inputlayer,1)
            self.dweight[k]=self.dweight[k]*miao[k]*kk
        self.dweightout=self.learnrate*self.ekout*self.active[self.hidelayer-1]*outputnum*self.ReLuPrime(outputnum)#(1-outputnum)
        #print("dweight",self.dweight)
        #print("dweightout",self.dweightout)
        '''for p in range(self.outputlayer):
            self.J = self.J + self.ek[k] * self.ek[k] / 2.0#误差能量
        for(int i=0;i<IM;i++){
    		for(int j=0;j<RM;j++)
    		{
    			for(int k=0;k<OM;k++)
    			{
    				dWin[i][j]=dWin[i][j]+learnRate*(Ek[k]*Wout[j][k]*XjActive[j]*(1-XjActive[j])*Xi[i]);
    				}//每个神经元的修正值
    			Win[i][j]=Win[i][j]+dWin[i][j]+alfa*(oldWin[i][j]-old1Win[i][j]);
    			old1Win[i][j]=oldWin[i][j];
    			oldWin[i][j]=Win[i][j];
    		}
    	}
        self.dweight1=self.ek*inputnum*outputnum*self.active*(1-self.active)#每行是一层,神经元nxn的
        dy=inputnum[:]
        #self.dweight2 =np.zeros((self.hidelayer,self.inputlayer,self.inputlayer))
        #for i in range(self.hidelayer):
        #    self.dweight2[i]=np.multiply(dy,self.dweight1[i])*self.learnrate
        #    dy=self.active[i]
        #print("123",self.dweight2,self.dweight1)
        #dy=self.active[self.hidelayer-1]
        self.dweightout=self.ek*inputnum*(1-outputnum)*self.learnrate'''
        '''算bias误差'''
        #print("sum axis=1",np.sum(self.dweight,axis=(1,2)))
        self.dbias=np.sum(self.dweight,axis=(1,2))*self.learnrate#每层神经元和
        dd=np.sum(self.dweightout)
        self.dbias=np.hstack((self.dbias,dd))
        '''修正'''
        self.weight-=self.dweight
        #print("bias",self.bias,"dbias",self.dbias)
        self.bias-=self.dbias
        #print("bias",self.bias)
        self.outweight-=self.dweightout
    def train(self,inputnum,outputnum,s):
        self.addlayer()
        #inputnum, outputnum=self.onetrain(inputnum,outputnum)
        for i in range(len(inputnum)):
            mix=100
            while(abs(mix)>s):
                outp=self.forward(inputnum[i])
                self.backforward(inputnum[i],outputnum[i],outp)
                mix=np.sum(outputnum[i]-outp)
                print("miss=",mix,outp,outputnum[i])
                print("train:"+str(i))
    def test(self,inputnum):
        #inputnum=self.onetest(inputnum)
        outputnum=np.ones(len(inputnum))
        for i in range(len(inputnum)):
            outputnum[i]=self.forward(inputnum[i])#self.oneback(self.forward(inputnum[i]))
            print("喵喵",outputnum[i])#,inputnum[i])
        return outputnum
'''
    def onetrain(self,inputnum,outputnum):
        self.min=np.min(inputnum,axis=1)
        self.max=np.max(inputnum,axis=1)
        for i in range(len(inputnum)):
            inputnum[i]=(inputnum[i]-self.min[i])/(self.max[i]-self.min[i])
        self.outmin = np.min(outputnum)
        self.outmax = np.max(outputnum)
        for i in range(np.size(outputnum,0)):
            outputnum[i] = (outputnum[i] - self.outmin) / (self.outmax - self.outmin)
        print(inputnum,outputnum)
        return inputnum,outputnum
    def onetest(self,inputnum):
        tmin = np.min(inputnum, axis=1)
        tmax = np.max(inputnum, axis=1)
        for i in range(len(inputnum)):
            inputnum[i]=(inputnum[i]-tmin[i])/(tmax[i]-tmin[i])
        return inputnum
    def oneback(self,outputnum):
        for i in range(len(outputnum)):
            outputnum[i]*=(self.outmax - self.outmin)
            outputnum[i]+=+ self.outmin
        return outputnum'''
print(yi.shape,yi,np.size(yi,0),yi[5])

''''''
'''归一化'''


mm = MinMaxScaler()
xi = mm.fit_transform(xi)
mm2=MinMaxScaler()
yi2 = mm2.fit_transform(yi)
yi2=yi2.reshape(-1)
yi=yi.reshape(-1)
print(xi,yi2)
bp1=bpnet(401,1,0.1,2)
print("training...")
bp1.train(xi[:50],yi2[:50],0.1)
print("test...")
yii=bp1.test(xi[50:])
yii=yii.reshape(10,1)
print(yii)
yii=mm2.inverse_transform(yii)
#除以最大最小误差-min     (x-min)/(max-min)
print(yii,yi[15:20])
plt.plot(np.arange(10),yi[50:],'ro',np.arange(10),yii,'bs')
plt.show()
plt.savefig('figure2.png')