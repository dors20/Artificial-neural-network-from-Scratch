IMPLEMENTATION:


* Data Pre-processing: The given dataset consisted of the columns Community, Age, Weight, Delivery Phase, HB, IFA, BP, Education, Residence and Result. To handle the null values in features of Age ,Weight ,HB and BP, we filled it with their means; the features Delivery phase and Residence were filled with their mode (i.e, 1) since these columns take binary values, and the Education feature was filled with 5 because all the values were 5 in the given dataset. We decided to drop the Community column because it didn’t seem to yield any meaningful insight in our analysis.        


*  Building of Neural Network:
The following classes were defined to build the ANN:
1. NN - Primarily used for the confusion matrix function.
2. ANN Layers - Used to create layer objects that can independently modify and store the layers’ weights and biases.
3. Activation - Consists of a variety of activation functions, each can be independently called. The activation function we used was Relu (between layer1 and layer2). The class also has functions for sigmoid and tanh because we tried using those activation functions as well. However, we noticed that the dataset performed better under the more linear activation of relu as compared to sigmoid. 
4. Act_softmax - It consists of the softmax activation functions used at the output layer. It has two functions, one for forward pass and another for backward pass.
5. Loss - Calculates the loss for model output and ground truth values.
6. CCE (Cross Entropy Loss) - It inherits the Loss class and performs sample-wise negative loss calculation.
7. Optimizer - It is an SGD optimizer used to update the weights and biases of the class.
8. Adam_op - It is an Adam optimizer. 
9. Activation_softmax_LCCE - It combines the softmax activation and cross-entropy loss for faster backpropagation. 




LIST OF HYPERPARAMTERS: 


* Model 1 - 
1. Layer 1 - 7 input neurons
2. Layer 2 - 16 inputs neurons
3. Output layer - 16 inputs neurons , 2 output
4. Test-size: 33%
5. Learning rate - 0.025
6. Decay -  5 * e^-7
7. Epoch - 10000
8. Optimizer used - SGD


The learning rate for small datasets is usually small, so we kept it at 0.05 after several attempts of trial and error with several values.


* Model 2 -
1. Layer 1 - 7 input neurons
2. Layer 2 - 128 input neurons
3. Output layer - 2 neurons
4. Learning rate - 0.05
5. Decay - 5 * e^-7
6. Epoch -11000
7. Optimizer used - Adam, beta1=0.9, beta2=0.999




KEY DESIGN FEATURES: 


1. SMOTE (Synthetic Minority Oversampling Technique) approach was used for pre-processing. The given dataset is largely biased towards the target class  1 because 74 out of 96 records were having the target value as ‘1’. Any learning network would have trouble predicting the actual 0 values as so, causing an imbalance in the minority class. So we synthetically replicate more 0 value records.  
Since SMOTE version requires scikit-learn 0.231 which conda doesn’t have, to use SMOTE, follow this- https://stackoverflow.com/questions/62436243/attributeerror-smote-object-has-no-attribute-validate-data the latest scikit-learn.
2. Implementation of Adam Optimizer.
3. Activation_softmax_LCCE - It combines the softmax activation and cross-entropy loss for faster backpropagation. 
4. The code is modularized using classes and objects, allowing reusability and making modifications easy.


IMPLEMENTATION OF CONCEPTS BEYOND THE BASICS:


* Adam Optimizer: Performs better than SGD when the dataset is small.
* SMOTE: For pre-processing
* OOP: Leveraging the class prototype structure in OOPs for achieving modularity.




STEPS TO RUN THE FILE:


Module Requirements-
pandas
numpy
sklearn


The source code for pre-processing and building the neural network is in the same file called assignment3.py (or assignment3.ipynb).
We are submitting a .py file as well as a .ipynb file of the code. 


For ipynb- On every run of the file, the kernel has to be restarted since every object stores a copy of inputs and outputs. So future runs will be affected by past executions.