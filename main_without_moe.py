
# Imports
import torch
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader   # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For a nice progress bar!
from moe import MoE
import matplotlib.pyplot as plt
torch.manual_seed(1)


'''--------------------------Defining function so that i can save progress-----------------------------------'''
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer=None):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer!=None:
        optimizer.load_state_dict(checkpoint["optimizer"])

'''-----------------------------------------------------------------------------------------------------------'''


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
input_size = 28
hidden_size = 256
num_layers = 1
num_classes = 10
sequence_length = 28
learning_rate = 5e-3
batch_size = 64
num_epochs = 10
load_params=False
save_params=False



# Recurrent neural network (many-to-one)

# Recurrent neural network with LSTM (many-to-one)
class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm0 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        self.lstm1 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        h1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm0( x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        
        
        out, _ = self.lstm1(out.reshape(batch_size,sequence_length,self.hidden_size), (h1, c1))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out


# Load Data
train_dataset = datasets.MNIST(
    root="C:\python learning\SAIDL\COMPUTER VISION\dataset", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root="C:\python learning\SAIDL\COMPUTER VISION\dataset", train=False, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

# Initialize network (try out just using simple RNN, or GRU, and then compare with LSTM)
model = RNN_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,num_epochs)

'''loading last saved parameters'''
if load_params==True:
    load_checkpoint(torch.load(r"C:\python learning\SAIDL\NLP\checkpoint_model_without_moe"),model,optimizer)
    load_checkpoint(torch.load(r"C:\python learning\SAIDL\NLP\checkpoint_scheduler_without_moe"),scheduler)

# Train Network
epoch_no=[]
av_loss=[]
for epoch in range(num_epochs):
    epoch_no.append(epoch)
    loss_total=0
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        # scores = model(data)
        loss = criterion(scores, targets)
        loss_total+=loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent update step/adam step
        optimizer.step()
    scheduler.step()
    print(f"for {epoch} epoch average loss is {loss_total/batch_size}")
    av_loss.append(loss_total/batch_size)



# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    # Toggle model back to train
    model.train()
    return num_correct / num_samples


print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")

plt.plot(epoch_no, av_loss)

# Add labels and title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Line Graph Example')

# Display the plot
plt.show()

'''-----------------------------saving updated parameters------------------------------------'''
if save_params==True:
    checkpoint_model={
                "state_dict":model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
    save_checkpoint(checkpoint_model,r"C:\python learning\SAIDL\NLP\checkpoint_model_without_moe")

    checkpoint_scheduler={
                "state_dict":scheduler.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
    save_checkpoint(checkpoint_scheduler,r"C:\python learning\SAIDL\NLP\checkpoint_scheduler_without_moe")
