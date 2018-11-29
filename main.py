
# coding: utf-8

# ## Logic Based FizzBuzz Function [Software 1.0]

# In[45]:


import pandas as pd

def fizzbuzz(n):
    
     # Logic Explanation: 
    # This program is a simple FizzBuzz problem.
    # Our task is to check if an integer is divisible by 3 then to return `Fizz`.
    # If it is divisible by 5, then the output should be `Buzz`.
    # If the integer is divisible by both 3 and 5, then the output should be `FizzBuzz`.
    # If not, then it should print `Other`.
    # % (modulo) operator checks the remainder.
    
     # Check whether a number is divisible by both 3 and 5.
    if n % 3 == 0 and n % 5 == 0:
        return 'FizzBuzz'
     # Check whether a number is divisible by both 3.
    elif n % 3 == 0:
        return 'Fizz'
    # Check whether a number is divisible by both 5.
    elif n % 5 == 0:
        return 'Buzz'
     # If not any of the above, then return Other.
    else:
        return 'Other'


# ## Create Training and Testing Datasets in CSV Format

# In[46]:


def createInputCSV(start,end,filename):
    
    # Why list in Python?
    # List is one of the collection in python. It is used for storing array of numbers.
    # It is ordered and changeable.
    # It also allows duplicate members.
    inputData   = []
    outputData  = []
    
    # Why do we need training Data?
    # Training data is used to train the machine learning algorithms 
    # It also helps in increasing the accuracy.
    for i in range(start,end):
        inputData.append(i)
        outputData.append(fizzbuzz(i))
    
    # Why Dataframe?
    # Dataframes are two dimensional data structure. The data is aligned in a tabular fashion in rows and columns.
    # Here, inputData of range 101-1000 is stored into `input` column. The outputData which can have values of Fizz, Buzz, 
    # FizzBuzz and Other is stored in `label` column.
    dataset = {}
    dataset["input"]  = inputData
    dataset["label"] = outputData
    
    # Writing to csv
    # This function is used to write into csv files. This function is used in creating training and testing.csv files.
    pd.DataFrame(dataset).to_csv(filename)
    
    print(filename, "Created!")


# ## Processing Input and Label Data

# In[47]:


def processData(dataset):
    
    # Why do we have to process?
    # Processing is necessary to have uniformity in input and the output values or normalizing the data. Also, computers operate in binaries.
    # Proper formatting of data helps in achieving better results and will led to much accurate predictions.
    # encodeData is representing a number in its corresponding bit representation. To increase the number of features, the inputs were
    # converted to a binary mode.
    data   = dataset['input'].values
    labels = dataset['label'].values
    
    processedData  = encodeData(data)
    processedLabel = encodeLabel(labels)
    
    return processedData, processedLabel


# In[48]:


def encodeData(data):
    
    processedData = []
    
    for dataInstance in data:
        
        # Why do we have number 10?
        # We have to process 1 to 1000 numbers. For representing 1000, atleast 10 bits are necessary.
        processedData.append([dataInstance >> d & 1 for d in range(10)])
    
    return np.array(processedData)


# In[49]:


from keras.utils import np_utils

def encodeLabel(labels):
    
    processedLabel = []
    
    for labelInstance in labels:
        if(labelInstance == "FizzBuzz"):
            # Fizzbuzz
            # If label is `FizzBuzz`, then store a value 3.
            processedLabel.append([3])
        elif(labelInstance == "Fizz"):
            # Fizz
            # If label is `Fizz`, then store a value 1.
            processedLabel.append([1])
        elif(labelInstance == "Buzz"):
            # Buzz
            # If label is `Buzz`, then store a value 2.
            processedLabel.append([2])
        else:
            # Other
            # If label is `Other`, then store a value 0.
            processedLabel.append([0])
            
    # This is used to convert array of labeled data to one-hot vector. 
    return np_utils.to_categorical(np.array(processedLabel),4)


# ## Model Definition

# In[50]:


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard

import numpy as np

# This is the input neurons. The size is 10 as we have to train out data on 1000 numbers and this can be achieved using 10 bits.
input_size = 10

# Dropout is randomly setting a fraction rate of input units to 0 at each update during training time.
# It helps is preventing overfitting.
drop_out = 0.05

# This is a hidden layer.
first_dense_layer_nodes  = 290

# Added this for testing the accuracy prediction.
# third_dense_layer_nodes  = 256

# This is the output layer. The value is 4 as we have 4 different categories.
second_dense_layer_nodes = 4

def get_model():
    
     # Why do we need a model?
    # model is the artifact that is created once a training process ends.
    # model here is a type of sequential layer. A model definition also includes defining the input and output layer.
    
    # Why use Dense layer and then activation?
    # Dense layer is a 2D hidden layer. Dropouts cannot be applied to dense layers.
    # Activation takes the input values from connecting nodes and gives an output based on activation function used. 
    # If we use activation at first then there are chances of data losses  ar data being modified.
    
    # Why use sequential model with layers?
    # Sequential model is a linear stack of layers.
    # It has just one unique input and output layer respectively.
    # The first layer in a Sequential model needs to receive information about its input shape
    model = Sequential()
    
    # add() is used to add a new layer to out neural network.
    # Model needs to know the input shape it should expect.input_dim specifies that the input size is of 10 neurons.
    # Dense specifies that each neuron is connected with every other neuron.
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    
    # Added this for testing the accuracy prediction.
    # model.add(Dense(third_dense_layer_nodes))
    
    # ReLU ranges from 0 to 1, i.e for negative inputs the value is considered as 0 and for positive values the graph
    # increases exponentially.
    # ReLu is less computationally expensive.
    model.add(Activation('relu'))
    
    # Why dropout?
    # Dropout is only applied on hidden layers.
    # Dropout helps in regularization effect.
    # Dropout reduces overfitting.
    model.add(Dropout(drop_out))
    
     
    # This again adds a layer of 4 neurons corresponding to 4 outputs in the network.
    model.add(Dense(second_dense_layer_nodes))
    
    # Why Softmax?
    # Softmax function helps in approximating a probability distribution.
    # Softmax is used when we have more than 2 classes to categorize the output.
    model.add(Activation('softmax'))
    
    # This prints a summary representation of the model.
    model.summary()
    
    # Why use categorical_crossentropy?
    # This is a loss function.
    # This is a parameter required to compile a model.
    # An optimizer is one of the two arguments required for compiling a Keras model for example, rmsprop and Adagrad etc. 
    # An optimization algorithms helps us to minimize (or maximize) an Objective function
    # A metrics is a function which is used to judge the performance of your model for example, binary_accuracy and 
    # categorical_accuracy etc.
    model.compile(optimizer='AdaDelta',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model


# # <font color='blue'>Creating Training and Testing Datafiles</font>

# In[51]:


# Create datafiles
createInputCSV(101,1001,'training.csv')
createInputCSV(1,101,'testing.csv')


# # <font color='blue'>Creating Model</font>

# In[52]:


model = get_model()


# # <font color = blue>Run Model</font>

# In[53]:


# This defines the way how a training data is splitted into training and validation sets. Here 0.2 specifies 80% 
# of the data will be used as training and rest 20% as validation set.
validation_data_split = 0.3

# This is the number of times the  input data is fed into the model.
num_epochs = 10000

# This is the size of input data which is fed at a time.
model_batch_size = 20

# This is size of batch of inputs to feed to the network for histograms computation.
tb_batch_size = 32

#This is the number of epochs with no improvement after which training will be stopped.
early_patience = 100

# Tensorboard is a visualization tool for visualizing dynamic graphs of of training and test metrics.
tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)

# This is stopping the training when a monitored quantity has stopped improving.
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')

# Read Dataset
dataset = pd.read_csv('training.csv')

# Process Dataset
processedData, processedLabel = processData(dataset)
history = model.fit(processedData
                    , processedLabel
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb,earlystopping_cb]
                   )


# # <font color = blue>Training and Validation Graphs</font>

# In[54]:


get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.DataFrame(history.history)
df.plot(subplots=True, grid=True, figsize=(10,15))


# # <font color = blue>Testing Accuracy [Software 2.0]</font>

# In[55]:


def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return "Other"
    elif encodedLabel == 1:
        return "Fizz"
    elif encodedLabel == 2:
        return "Buzz"
    elif encodedLabel == 3:
        return "FizzBuzz"


# In[56]:


wrong   = 0
right   = 0

testData = pd.read_csv('testing.csv')

processedTestData  = encodeData(testData['input'].values)
processedTestLabel = encodeLabel(testData['label'].values)
predictedTestLabel = []

for i,j in zip(processedTestData,processedTestLabel):
    y = model.predict(np.array(i).reshape(-1,10))
    predictedTestLabel.append(decodeLabel(y.argmax()))
    
    if j.argmax() == y.argmax():
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Testing Accuracy: " + str(right/(right+wrong)*100))

# Please input your UBID and personNumber 
testDataInput = testData['input'].tolist()
testDataLabel = testData['label'].tolist()

testDataInput.insert(0, "UBID")
testDataLabel.insert(0, "mikipadh")

testDataInput.insert(1, "personNumber")
testDataLabel.insert(1, "50286289")

predictedTestLabel.insert(0, "")
predictedTestLabel.insert(1, "")

output = {}
output["input"] = testDataInput
output["label"] = testDataLabel

output["predicted_label"] = predictedTestLabel

opdf = pd.DataFrame(output)
opdf.to_csv('output.csv')

