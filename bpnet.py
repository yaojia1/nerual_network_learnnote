import numpy as np
#import tensorflow as tf
b=[[1,1,1],[3,3,3],[5,5,5]]
print(len(b),np.sum(b,axis=1))
'''
a= np.arange(3).reshape(3)

c=a*b
print(c)
c=np.matmul(b,a)
print(c)
a= np.random.rand(5,1,1)
print("ddddd",a)
def train1():
    inputParameter=np.arange(0,20,dtype =  float)#input
    outputParameter=np.array(inputParameter*2+23)#outpute
    learnRate = 0.55
def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ'''
class bpnet(object):
    def __init__(self,inputlayer,outputlayer,learnrate,hidelayer):#从一开始
        self.inputlayer=inputlayer
        self.outputlayer=outputlayer
        self.learnrate=learnrate
        self.hidelayer=hidelayer
    def addlayer(self):
        self.weight=np.random.rand(self.hidelayer,self.inputlayer,self.inputlayer)#, dtype = float, order = 'C')#all weight,and hight=input highth每行是一个神经元
        print(self.weight[0][0][0])
        self.bias=np.random.rand(self.hidelayer+1)#bias
        self.outweight=np.random.rand(self.outputlayer,self.inputlayer)#, dtype = float, order = 'C')
        self.__X=np.zeros((self.hidelayer,self.inputlayer),dtype = float, order = 'C')#中间的x=weight*input+bias
        self.active=np.zeros((self.hidelayer,self.inputlayer),dtype = float, order = 'C')#激活值a
    def forward(self,inputnum):
        for i in range(self.hidelayer):#每一层的激活值存在active[i]里
            self.__X[i]=np.matmul(self.weight[i],inputnum)+self.bias[i]#
            self.active[i]=1 / (1 + np.exp(-self.__X[i]))#激活值
            inputnum=self.active[i]#输入值得传递
        outputnum=np.matmul(self.outweight,self.active[self.hidelayer-1])+self.bias[self.hidelayer]
        return outputnum
    def backforward(self,inputnum,outputParameter,outputnum):
        self.ekout=np.zeros((self.outputlayer,self.inputlayer))
        '''
        for k in range(self.outputlayer):
            self.ek[k] = outputParameter[k] - outputnum[k]'''

        self.ekout=(outputParameter-outputnum)*self.outweight
        self.ekhide=np.zeros((self.hidelayer,self.inputlayer))
        lastek=self.ekout[:]
        for i in range(self.hidelayer):
            self.ekhide[self.hidelayer-1-i]=np.matmul(self.weight[self.hidelayer-1-i],np.ones((self.inputlayer)))*lastek
            lastek=self.ekhide[self.hidelayer-1-i][:]
        print("ek:",self.ekhide)
        lastek=np.vstack((self.ekhide,self.ekout))
        print("eek:",lastek)
        #self.J=np.sum(np.dot(self.ek,self.ek)/2.0)
        miao=np.vstack((self.active[:self.hidelayer-1],inputnum))
        dweight0=self.learnrate*self.ekhide*self.active*(1-self.active)
        print("aaa0",dweight0,miao,"//",self.active)
        self.dweight=np.ones((self.hidelayer,self.inputlayer,self.inputlayer))
        for k in range(self.hidelayer):
            self.dweight[k]=self.dweight[k]*miao[k]*dweight0[k]
        print("??",self.dweight)
        self.dweightout=self.learnrate*self.ekout*self.active[self.hidelayer-1]*outputnum*(1-outputnum)
        print("dweight",self.dweight)
        print("out",self.dweightout)
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
        print("11",self.dweightout)
        self.dbias=np.sum(np.sum(self.dweight,axis=1),axis=1)*self.learnrate#每层神经元和
        dd=np.sum(self.dweightout,axis=1)
        dd=np.sum(dd)
        self.dbias=np.hstack((self.dbias,dd))
        self.weight-=self.dweight
        print("bias",self.bias,"dbias",self.dbias)
        for j in range(self.hidelayer+1):
            self.bias[j]-=self.dbias[j]
        print("bias",self.bias)
        self.outweight-=self.dweightout
    def train(self,inputnum,outputnum,s):
        self.addlayer()
        for i in range(len(inputnum)):
            mix=100
            while(mix>s):
                outp=self.forward(inputnum[i])
                self.backforward(inputnum[i],outputnum[i],outp)
                mix=np.sum(outputnum[i]-outp)
                print("train:"+str(i))
    def test(self,inputnum):
        for i in range(len(inputnum)):
            print("喵喵",inputnum[i],self.forward(inputnum[i]))
a=np.ones((2,1))
b=np.ones((5,1,2))
print("kkk",np.matmul(a,b))
print(len([1,2,3]))
bp1=bpnet(2,1,0.1,3)
bp1.train([[1,2],[3,4]],[5,7],0.1)
bp1.test([[1,2],[3,4],[5,6],[7,8]])





