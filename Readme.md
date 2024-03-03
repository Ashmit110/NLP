# OPTIMIZING MOE and Comparing Performance

###### NOTE-

i was unable to import the model from hugging face as i have never worked with sequenced data so for this assignment i have used MOE on mnist dataset classification task.I know this going of track from the provided assignment but exploring the use of MOE for image classification i guess wont cause any harm to the society and probably help me gain some more insight.

## Aim

To study the effects of number of experts and experts size

## Goal

to achieve maximum accuracy in 10 epochs

## FineTuning

#### Meaning of different parameters

hidden_size_moe= refers to number of features in the hidden layer of the expert

num_experts =refers to the expert network initialized

top_k=number of experts that will be used to evaluate the result.

### Run 1

DRY RUN

'''moe hyperparameters'''

hidden_size_moe=256

num_experts=20

top_k=2#top k gates pass

![1709465870202](image/Readme/1709465870202.png)

for 9 epoch average loss is 0.017321780404927267
Accuracy on training set: 99.485847
Accuracy on test set: 99.14


observation-accuracy is pretty good but can be improved .

### Run 2

For this run


'''moe hyperparameters'''

hidden_size_moe=256

num_experts=10

top_k=2#top k gates pass

![1709467538769](image/Readme/1709467538769.png)

for 9 epoch average loss is 0.01558819169531489
Accuracy on training set: 99.564308
Accuracy on test set: 99.35

### Run 3
