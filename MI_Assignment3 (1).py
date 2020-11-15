#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

np.random.seed(0)


# In[2]:


data = pd.read_csv('LBW_Dataset.csv')


# In[3]:


data.head(5)


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# In[6]:


for col in data.columns:
    print("{}:{}".format(col,data[col].nunique()))


# In[7]:


data.describe()


# In[8]:


data['Community'].value_counts()
#NO null values in community column


# In[9]:


data['Residence'].value_counts()


# In[10]:


# FIll Age ,Weight ,HB and BP  Null values with mean
# Delivery phase with 1 since the coulmn takes binary values
# Education with 5
#Residence with 1


# In[11]:


data.columns


# In[12]:


x=data['HB'].mean()
print(x)


# In[13]:


values = { 'Age':data['Age'].mean(), 'Weight':data['Weight'].mean(), 'Delivery phase':1, 'HB':9.0, 'BP':data['BP'].mean(),
       'Education':5, 'Residence':1}


# In[14]:


data = data.fillna(value=values)


# In[15]:


data.head(3)


# In[16]:


data.isnull().sum()


# In[17]:


def sigmoid(x):
    return 1/(1+np.exp(-x))
#activation function


# In[18]:


data['Result'].value_counts()


# In[19]:


data=data.drop(columns='Education')


# In[20]:


#75% of the result column is 1 so the model might over fit 


# In[21]:


#


# In[22]:


class ANN_layers:
    def __init__(self,num_inputs ,num_cells):
        self.weights = 0.1 * np.random.randn(num_inputs,num_cells) #in range[-1,1]
        self.biases = np.zeros((1,num_cells))
    def fpropogation(self , inputs):
        self.inputs= inputs
        self.output= np.dot(inputs ,self.weights)
    def Bpropogation(self,dval):
        self.dweights= np.dot(self.inputs.T,dval)
        self.dbiases = np.sum(dval,axis=0,keepdims=True)
        self.dinputs =np.dot(dval,self.weights.T)

def confidenceTolist(x):
    rows = x.shape[0]
    cols = x.shape[1]
    predict=[None]*rows
    for i in range(0,rows):
        if(x[i,0]>x[i,1]):
            predict[i]=0
        else:
            predict[i]=1
    return predict
        
        
    
        
class activation:
    def fRelu(self,x):
        self.inputs=x
        self.output = np.maximum(0,x)
        #sigmoid
        #self.inputs =x
        #self.output = 1/(1+np.exp(-x))
        #self.output = np.tanh(x) #tanh
    def sigmoid(self,x):
        self.inputs =x
        self.output = 1/(1+np.exp(-x))
    
    def tanh(self,x):
        self.inputs =x
        self.output = np.tanh(x)
        
    def back(self,dval):
        self.dinputs = dval.copy()
        self.dinputs[self.inputs <=0] =0 
        

class act_softmax :       
    def fsoftmax(self,inputs):
        self.inputs = inputs
        exp1 = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        prob = exp1 /np.sum(exp1 ,axis=1, keepdims=True)
        self.output = prob
        
    def bsoftmax(self,dval):
        self.dinputs = np.empty_like(dval)
        
        for i ,(single_o,single_dval) in enumerate(zip(self.output,dval)):
            single_o = single_o.reshape(-1,1)
            jacobian = np.diagflat(single_o)-np.dot(single_o,single_o.T)
            self.dinputs[i] = np.dot(jacobian,single_dval)
        
class loss:
    def cal(self,output,y):
        sampleLosses = self.check(output,y)
        data_loss= np.mean(sampleLosses)
        return data_loss
    
class CCE(loss):
    def check(self,y_pred,y_true):
        x = len(y_pred)
        y_pred_clipped = np.clip(y_pred ,1e-7,1 - 1e-7)
        confidence = y_pred_clipped[range(x),y_true]
        negLikely = -np.log(confidence)
        return negLikely
    
    def backward(self,dval,y_true):
        
        x=len(dval)
        label=len(dval[0])
        if len(y_true.shape)==1:
            y_true=np.eye(label)[y_true] #
        self.dinputs =-y_true/dval
        self.dinputs =self.dinputs/x


# In[23]:


class optimizer:
    
    def __init__(self , l_rate=1.,decay=0.,p=0.):
        self.l_rate = l_rate
        self.curr_l_rate=l_rate
        self.decay = decay
        self.iter =0
        self.p =p
        
    def initial_update_param(self):
        if self.decay:
            self.curr_l_rate = self.l_rate *(1./(1+self.decay*self.iter))
            
    def update_params(self,layer):
        if self.p:
            if not hasattr(layer,'wieght_momentums'):
                layer.weight_p = np.zeros_like(layer.weights)
                layer.bias_p = np.zeros_like(layer.biases)
            
            w_updates= self.p*layer.weight_p - self_curr_l_rate*layer.dweights
            layer.weight_p = w_updates
            
            b_updates= self.p*layer.bias_p - self_curr_l_rate*layer.dbiases
            layer.bias_p = b_updates
        
        else:
            w_updates =-self.curr_l_rate * layer.dweights
            b_updates =-self.curr_l_rate * layer.dbiases
            
        layer.weights+=w_updates
        layer.biases+=b_updates
        
    def post_update_params(self):
        self.iter+=1


# In[24]:


class Activation_softmax_LCCE():
    
    def __init__(self):
        self.activation=act_softmax()
        self.loss=CCE()
    
    def forward(self,inputs,y_true):
        self.activation.fsoftmax(inputs)
        self.output=self.activation.output
        
        return self.loss.cal(self.output,y_true)
        
    def backpass(self,dval,y_true):
        x=len(dval)
        if len(y_true.shape)==2:
            y_true = np.argmax(y_true,axis=1)
        
        self.dinputs =dval.copy()
        self.dinputs[range(x),y_true]-=1
        self.dinputs= self.dinputs/x

"""
Hlayer1 =ANN_layers(8,8)
Hlayer2 =ANN_layers(8,5)
Output_layer=ANN_layers(5,2)
input1= data.loc[:,['Community', 'Age', 'Weight', 'Delivery phase', 'HB', 'IFA', 'BP',
        'Residence' ]]
#Hlayer1.fpropogation(input1)
#print(Hlayer1.output)
print(Hlayer1.output.shape[0])
a1=activation()
a1.fRelu(Hlayer1.output)
#print(a1.output)
Hlayer2.fpropogation(a1.output)
#print(Hlayer2.output)
a2=activation()
a2.fRelu(Hlayer2.output)
Output_layer.fpropogation(a2.output)
a3=act_softmax()
a3.fsoftmax(Output_layer.output)
print(a3.output[:10])
#Categorical cross-entropy is explicitly used to compare
#a “ground-truth” probability ( y or “ targets ”)
result=data["Result"].tolist()
result1=data['Result']
print(result)
loss_calc=CCE()
predict= confidenceTolist(a3.output)
loss=loss_calc.cal(a3.output,result)
print(loss)
#print(len(result1.shape))
"""
# In[25]:


input2=data.loc[:,['Community','Age', 'Weight', 'Delivery phase', 'HB', 'IFA', 'BP', 'Residence']] 


# In[26]:


result=data['Result']


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


input1,x_test, result1, y_test = train_test_split(input2, result, test_size=0.33, random_state=42)


# In[29]:


Hlayer1=ANN_layers(8,8)


# In[30]:


Hlayer2=ANN_layers(8,5)


# In[31]:


Output_layer=ANN_layers(5,2)


# In[32]:


act1=activation()


# In[33]:


act2=activation()


# In[34]:


loss_activation=Activation_softmax_LCCE()


# In[35]:


optimize = optimizer(l_rate=0.025,decay=5e-7)


# In[36]:


for epoch in range(10001):
    Hlayer1.fpropogation(input1)
    act1.fRelu(Hlayer1.output)
    Hlayer2.fpropogation(act1.output)
    act2.fRelu(Hlayer2.output)
    Output_layer.fpropogation(act2.output)
    loss = loss_activation.forward(Output_layer.output,result1)
    prediction = np.argmax(loss_activation.output,axis=1)
    if len(result1.shape)==2:
        result1=np.argmax(result1,axis=1)
    accuracy= np.mean(prediction==result1)
    if not epoch %100:
        print(f'epoch:{epoch} ,'+
              f'accuracy:{accuracy:.3f} ,'+
              f'loss:{loss} ,'+
              f'lr:{optimize.curr_l_rate},')
    loss_activation.backpass(loss_activation.output,result1)
    Output_layer.Bpropogation(loss_activation.dinputs)
    act2.back(Output_layer.dinputs)
    Hlayer2.Bpropogation(act2.dinputs)
    act1.back(Hlayer2.dinputs)
    Hlayer1.Bpropogation(act1.dinputs)
              
    optimize.initial_update_param()
    optimize.update_params(Hlayer1)
    optimize.update_params(Hlayer2)
    optimize.update_params(Output_layer)
    optimize.post_update_params()


# In[37]:


Hlayer1.fpropogation(x_test)
act1.fRelu(Hlayer1.output)
Hlayer2.fpropogation(act1.output)
act2.fRelu(Hlayer2.output)
Output_layer.fpropogation(act2.output)
loss = loss_activation.forward(Output_layer.output,y_test)
prediction = np.argmax(loss_activation.output,axis=1)
if len(y_test.shape)==2:
    result1=np.argmax(y_test,axis=1)
accuracy= np.mean(prediction==y_test)
print(f'epoch:{epoch} ,'+f'accuracy:{accuracy:.3f} ,'+
      f'lr:{optimize.curr_l_rate},')


# In[38]:


#since dataset is biased towards 1 we will try SMOTE approach


# In[39]:


data2=data[(data['Result']==0)]
data2.head(3)


# In[40]:


new_data=pd.concat([data,data2])
#new_data=pd.concat([new_data,data2])
new_data.head(3)


# In[41]:


new_data=new_data.sample(frac=1)
new_data.head(3)


# In[42]:


new_data['Result'].value_counts()


# In[43]:


new_input=new_data.loc[:,['Community','Age', 'Weight', 'Delivery phase', 'HB', 'IFA', 'BP', 'Residence']]
new_result=new_data['Result']
input11,x_test11, result11, y_test11 = train_test_split(new_input, new_result, test_size=0.3, random_state=42)
Hlayer11=ANN_layers(8,24)
#Hlayer12=ANN_layers(16,8)
Output_layer1=ANN_layers(24,2)
act11=activation()
#act12=activation()
loss_activation1=Activation_softmax_LCCE()
optimize1 = optimizer(l_rate=0.025,decay=5e-7)
for epoch in range(11001):
    Hlayer11.fpropogation(input11)
    act11.fRelu(Hlayer11.output)
    #Hlayer12.fpropogation(act11.output)
    #act12.fRelu(Hlayer12.output)
    Output_layer1.fpropogation(act11.output)
    loss1 = loss_activation1.forward(Output_layer1.output,result11)
    prediction1 = np.argmax(loss_activation1.output,axis=1)
    if len(result11.shape)==2:
        result11=np.argmax(result11,axis=1)
    accuracy1= np.mean(prediction1==result11)
    if not epoch %100:
        print(f'epoch:{epoch} ,'+
              f'accuracy:{accuracy1:.3f} ,'+
              f'loss:{loss1:.3f} ,'+
              f'lr:{optimize.curr_l_rate},')
        
    loss_activation1.backpass(loss_activation1.output,result11)
    Output_layer1.Bpropogation(loss_activation1.dinputs)
    act11.back(Output_layer1.dinputs)
    #Hlayer12.Bpropogation(act12.dinputs)
    #act11.back(Hlayer12.dinputs)
    Hlayer11.Bpropogation(act11.dinputs)
              
    optimize.initial_update_param()
    optimize.update_params(Hlayer11)
    #optimize.update_params(Hlayer12)
    optimize.update_params(Output_layer1)
    optimize.post_update_params()


# In[44]:


#x_test=x_test11
#y_test=y_test11
Hlayer11.fpropogation(x_test)
act11.fRelu(Hlayer11.output)
#Hlayer12.fpropogation(act11.output)
#act12.fRelu(Hlayer12.output)
Output_layer1.fpropogation(act11.output)
loss1 = loss_activation1.forward(Output_layer1.output,y_test)
prediction1 = np.argmax(loss_activation1.output,axis=1)
if len(y_test.shape)==2:
    result11=np.argmax(y_test,axis=1)
accuracy1= np.mean(prediction1==y_test)
print(f'epoch:{epoch} ,'+f'accuracy:{accuracy1:.3f} ,'+
       f'loss:{loss1:.3f} ,'+
      f'lr:{optimize.curr_l_rate}')


# In[ ]:





# In[ ]:




