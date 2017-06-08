"""
    Program loads caltech101_silhouettes_28 data set
    and runs multiclass neural network with SGD/softmax to classify images
"""


import Dataset
import Network

Dataset.pca("train_data")
train, valid, test = Dataset.load_data()
net = Network.Network([784,442,101], activationFunc=Network.sigmoid)
#net.SGD(train,30,10,0.5,5,test)
#just using sigmoid, training data with 30 epochs minibatch size 10, learning rate 0.5, regularization param 5, test data:
#really we're supposed to do net.SGD(train,30,10,0.5,5,valid), then feedforward the test data. woops.
#Anyways the result for this was accurancy: 1468/2307. theres some overfitting since we're using that as our evaluation data

#TRIAL 0:
#net.SGD(train,30,10,0.5,5,valid)
#print net.accuracy(test)
#the way you're supposed to do it. 61.64% accuracy on test data. Had 61.97% accuracy on validation data, 82% accuracy on training data
#matrices: (training cost, training acc, eval cost, eval acc): http://i.imgur.com/wLlRcPY.png. in the future i'll probably save it to a file

#TRIAL 1: (reduce learning rate to 0.3)
#net.SGD(train,30,10,0.3,5,valid)
#print net.accuracy(test)
#Decreasing learning rate to 0.3. Had 98% accuracy on training data, like 66% accuracy on validation data.
#1498 / 2307: 64.93% on test data, 66% on validation data   http://i.imgur.com/OHxnRtS.png
#slightly better but severe overfitting seems to be occurring

#TRIAL 2: (reduce learning rate to 0.3, increase regularization param to 10)
#net.SGD(train,30,10,0.3,10,valid)
#print net.accuracy(test)
#1500/2307, 4076/4100 acc on training, 1501/2264 for validation

#TRIAL 3:
#validCost,validAcc,trainCost,trainAcc = net.SGD(train,30,100,0.3,5,valid)
#thefile=open('resBatch100LrnRate0pt3Reg5.txt','w') #specify a new file for each trial.
#4097/4100 acc on training, 1510/2264 on validation. didn't get test accuracy.

#TRIAL 4:
#validCost,validAcc,trainCost,trainAcc = net.SGD(train,30,10,1,5,valid)
#didn't let it complete because it was obviously bad. cost at 0.9285, wasn't changing

#TRIAL 5:
#validCost,validAcc,trainCost,trainAcc = net.SGD(train,30,10,.2,5,valid)
#thefile=open('resBatch10LrnRate0pt2Reg5.txt','w')
#3952/4100 acc on training, 1543/2264 on validation. didnt get test accuracy. seems like slightly less overfitting than 0.3 case.

#TRIAL 6:
#validCost,validAcc,trainCost,trainAcc = net.SGD(train,30,10,.1,5,valid)
#thefile=open('resBatch10LrnRate0pt1Reg5.txt','w')
#3737/4100 acc on training, 1539/2264 on validation (probably needs more epochs though). 1530 / 2307 on test data, which is 66%. less overfitting tho.

#TRIAL 7:
#validCost,validAcc,trainCost,trainAcc = net.SGD(train,50,10,.1,5,valid)
#thefile=open('resBatch10LrnRate0pt1Reg5Epochs50.txt','w')
#With 50 epochs:
#4014/4100 accon training, 1540/2264 on validation (ok more epochs was not the problem). 1546/2307 on test data.

#TRIAL 8:
#validCost,validAcc,trainCost,trainAcc = net.SGD(train,30,10,.5,5,valid)
#thefile=open('resBatch10LrnRate0pt5Reg5v2.txt','w')
#1540/2307 for test data (slight improvement i guess), 98% on training data, 69% on validation.

#TRIAL 9: ACTIVATION FUNCTION RELU
#net=Network.Network([784,442,101], activationFunc=Network.relu)
#validCost,validAcc,trainCost,trainAcc = net.SGD(train,30,10,.1,5,valid)
#NOTE: at learning rate 0.5 it oscillated with error around 100
#thefile=open('resBatch10LrnRate0pt1Reg5activRELU.txt','w')
#1544/2264 validation, 4017/4100 on training. 1535 on test data


#TRIAL 10: ACTIVATION FUNCTION TANH
#net=Network.Network([784,442,101], activationFunc=Network.tanh)
#NOTE: validCost,validAcc,trainCost,trainAcc = net.SGD(train,30,10,.5,5,valid) had it oscillate with error about 100
#validCost,validAcc,trainCost,trainAcc = net.SGD(train,30,10,.1,5,valid)
#thefile=open('resBatch10LrnRate0pt1Reg5activTANH.txt','w')
#1540 /2264 on validation, 3995/4100 on training. 1546 on test data

#TRIAL 11: ACTIVATION FUNCTION RELU, NO REGULARIZATION
#net=Network.Network([784,442,101], activationFunc=Network.relu)
#validCost,validAcc,trainCost,trainAcc = net.SGD(train,30,10,.1,0,valid)
#thefile=open('resBatch10LrnRate0pt1Reg0activRELU.txt','w')
#1540/2264 on validation, 4014/4100 on training

#TRIAL 12: ACTIVATION FUNCTION RELU, LOTS OF REGULARIZATION
#net=network3.Network([784,442,101], activationFunc=network3.relu)
#validCost,validAcc,trainCost,trainAcc = net.SGD(train,30,10,.1,30,valid)
#thefile=open('resBatch10LrnRate0pt1Reg30activRELU.txt','w').
#1544/2264 on validatoin, 4016 on training.

#TRIAL 13: REDOING TRIAL 5
#validCost,validAcc,trainCost,trainAcc = net.SGD(train,30,10,.2,5,valid)
#thefile=open('resBatch10LrnRate0pt2Reg5v2.txt','w')
#1570/2264 for validation, 3929 for training. 1563 on test data

#TRIAL 14: REDOIGN TRIAL 5, HIGH REGULARIZATION
#validCost,validAcc,trainCost,trainAcc = net.SGD(train,30,10,.2,50,valid)
#thefile=open('resBatch10LrnRate0pt2Reg50.txt','w')
#1544/2264 on validatoin, 4013 on training.

#TRIAL 15: REDOING TRIAL 5, SLIGHTLY LOWER LEARNING RATE, MORE EPOCHS
#validCost,validAcc,trainCost,trainAcc = net.SGD(train,40,10,.15,5,valid)
#thefile=open('resBatch10LrnRate0pt15Reg5v2.txt','w')

#TRIAL 16: REDOING TRIAL 5, HIGH BATCH SIZE
#validCost,validAcc,trainCost,trainAcc = net.SGD(train,30,200,.2,5,valid)
#thefile=open('resBatch200LrnRate0pt2Reg5.txt','w')
#Very slow, but also avoids overfitting

#TRIAL 17: REDOING TRIAL 5, HIGH LEARNING RATE
#validCost,validAcc,trainCost,trainAcc = net.SGD(train,30,10,1,5,valid)
#thefile=open('resBatch10LrnRate1pt0Reg5.txt','w')
#doesnt work

#TRIAL 18: REDOING TRIAL 5, NAN-TO-NUM
validCost,validAcc,trainCost,trainAcc = net.SGD(train,30,10,.2,5,valid)
thefile=open('resBatch10LrnRate0pt2Reg5v3.txt','w')

for item in trainCost:
	thefile.write(str(item) + ",")
thefile.write("\n")
for item in trainAcc:
	thefile.write(str(item) + ",") #train accuracy should be divided by 4100
thefile.write("\n")
for item in validCost:
	thefile.write(str(item) + ",")
thefile.write("\n")
for item in validAcc:
	thefile.write(str(item) + ",") #validation accuracy should be divided by 2264
thefile.write("\n")
thefile.write(str(net.accuracy(test)))

#lines are printed in order of train cost, train acc, validcost, validacc

for item in trainCost:
	thefile.write(str(item) + ",")
thefile.write("\n")
for item in trainAcc:
	thefile.write(str(item) + ",") #train accuracy should be divided by 4100
thefile.write("\n")
for item in validCost:
	thefile.write(str(item) + ",")
thefile.write("\n")
for item in validAcc:
	thefile.write(str(item) + ",") #validation accuracy should be divided by 2264
thefile.write("\n")
thefile.write(str(net.accuracy(test)))

#lines are printed in order of train cost, train acc, validcost, validacc
