import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.io as sio
from random import seed
from random import randint
from random import shuffle
'''测试数据'''
matfn = u'E:/matlabproj/spectra_data.mat'
data = sio.loadmat(matfn)
plt.close('all')
NIR = data['NIR']
octane = data['octane']
NIR=np.array(NIR)
octane=np.array(octane)
input_size=401
output_size=1
hide_size=3
'''随机生成训练测试数据'''
intnum=50
testnum=10
datanum=60
seednum= randint(0,100)
seed(seednum)
shuffle(NIR)
seed(seednum)
shuffle(octane)
xi=NIR
yi=octane
'''设置参数'''
# 设置学习率
learning_rate = 0.01
# 设置训练次数
train_steps = 1000

#placeholder，用来存放数据的容器
xs = tf.placeholder(tf.float32,[None,input_size]) # 样本数未知，特征数为1，占位符最后要以字典形式在运行中填入
ys = tf.placeholder(tf.float32,[None,output_size])

'''设置隐层输出层'''

def addLayer(inSize,outSize,hidesize):
    Weights = tf.Variable(tf.random_normal([hidesize,inSize,inSize]))
    basis = tf.Variable(tf.zeros([hidesize,outSize])+0.1)
    return Weights,basis
def forward(inputData,Weights,basis,activity_function = None):
    weights_plus_b = tf.matmul(inputData,Weights)+basis
    if activity_function is None:
        ans = weights_plus_b
    else:
        ans = activity_function(weights_plus_b)
    return ans
#l1=addLayer(xs,input_size,output_size,tf.nn.relu)
#第一层隐层
'''
weights1 = tf.Variable(tf.random_normal([input_size,hide_size]))
basis = tf.Variable(tf.zeros([1,hide_size])+0.1)
weights_plus_b = tf.matmul(xs,weights1)+basis
l1=tf.nn.relu(weights_plus_b)'''
'''wrong test
lay1=[1]*hide_size
weightn,basin=addLayer(input_size,output_size,hide_size)
print(weightn,basin)
lay1[0]=forward(xs,weightn,basin,0,tf.nn.relu)
for i in range(1,hide_size):
    lay1[i]=forward(lay1[i-1],weightn,basin,i,tf.nn.relu)#隐层前向传播'''
lay1=[1]*hide_size
weightn,basin=addLayer(input_size,output_size,hide_size)
lay1[0]=forward(xs,weightn[0],basin[0],tf.nn.relu)
for i in range(1,hide_size):
    lay1[i]=forward(lay1[i-1],weightn[i],basin[i],tf.nn.relu)
l0=forward(xs,weightn[0],basin[0],tf.nn.relu)
l5=forward(xs,weightn[1],basin[1],tf.nn.relu)
l1=forward(xs,weightn[2],basin[2],tf.nn.relu)
weights01 = tf.Variable(tf.random_normal([input_size,input_size]))
basis01 = tf.Variable(tf.zeros([1,output_size])+0.1)
l11=forward(xs,weights01,basis01,tf.nn.relu)
weights02 = tf.Variable(tf.random_normal([input_size,input_size]))
basis02 = tf.Variable(tf.zeros([1,output_size])+0.1)
l12=forward(xs,weights02,basis02,tf.nn.relu)
weights03 = tf.Variable(tf.random_normal([input_size,input_size]))
basis03 = tf.Variable(tf.zeros([1,output_size])+0.1)
#l1=forward(xs,weights03,basis03,tf.nn.relu)
#输出层
#l2 = addLayer(l1,10,1,activity_function=None)
weights2 = tf.Variable(tf.random_normal([input_size,output_size]))
basis2 = tf.Variable(tf.zeros([1,1])+0.1)
weights_plus_b2 = tf.matmul(lay1[hide_size-1],weights2)+basis2
l2=weights_plus_b2

#loss
loss = tf.reduce_mean(tf.reduce_sum(tf.square((ys-l2)),reduction_indices = [1]))#需要向相加索引号，redeuc执行跨纬度操作
train =  tf.train.GradientDescentOptimizer(0.1).minimize(loss) # 选择梯度下降法
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#开始训练
for i in range(10000):
    sess.run(train,feed_dict={xs:xi[:50],ys:yi[:50]})
    if i%500 == 0:
        print(sess.run(loss,feed_dict={xs:xi[50:],ys:yi[50:]}))
'''saver = tf.train.Saver()
file_path = '/test'
save_path = saver.save(sess, file_path)
#进行预测
weights_plus_c = tf.matmul(xs,weights1)+basis
l3=tf.nn.relu(weights_plus_c)
weights_plus_b3 = tf.matmul(l3,weights2)+basis2
result=sess.run(weights_plus_b3,feed_dict={xs:xi[50:]})'''
'''wrong test
ll=[1]*hide_size
ll[0]=forward(xs,weightn,basin,0,tf.nn.relu)
for i in range(1,hide_size):
    ll[i]=forward(ll[i-1],weightn,basin,i,tf.nn.relu)#隐层前向传播'''
#ll11=forward(xs,weights01,basis01,tf.nn.relu)
#ll12=forward(xs,weights02,basis02,tf.nn.relu)
#ll1=forward(xs,weights03,basis03,tf.nn.relu)
'''
ll11=forward(xs,weightn[0],basin[0],tf.nn.relu)
ll12=forward(xs,weightn[1],basin[1],tf.nn.relu)
ll1=forward(xs,weightn[2],basin[2],tf.nn.relu)'''
lay2=[1]*hide_size
lay2[0]=forward(xs,weightn[0],basin[0],tf.nn.relu)
for i in range(1,hide_size):
    lay2[i]=forward(lay2[i-1],weightn[i],basin[i],tf.nn.relu)
#输出层
#l2 = addLayer(l1,10,1,activity_function=None)
#weights_plus_b2 = tf.matmul(ll1,weights2)+basis2
#weights_plus_b3=weights_plus_b2
weights_plus_b3=forward(lay2[hide_size-1],weights2,basis2,tf.nn.relu)
result=sess.run(weights_plus_b3,feed_dict={xs:xi[50:]})

#计算误差
subtract_op=tf.square(tf.subtract(result,ys))/2
error=sess.run(subtract_op,feed_dict={ys:yi[50:]})#add_op,feed_dict={ys:y_data})
xx=[1,2,3,4,5,6,7,8,9,10]
#可视化
plt.plot(xx,result,"y")
plt.plot(xx,yi[50:],"ro")
plt.plot(xx,error,"c")
plt.savefig('figure3.png')

sess.close()