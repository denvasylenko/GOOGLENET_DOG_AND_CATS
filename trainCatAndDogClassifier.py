import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from customCSV import CatsAndDogsCSV
from resizeCatsAndDogsImages import ResizedCatsAndDogsDataset
from customDataset import CatsAndDogsDataset

num_classes = 2
learning_rate = 3e-4
batch_size = 32
num_epochs = 10
load_modal = False


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ResizedCatsAndDogsDataset(root_dir='train', resized_dir='cats_dogs_resized', resize_shape=(224, 224))
CatsAndDogsCSV(csv_file='cats_dogs.csv', root_dir='train')
dataset = CatsAndDogsDataset(csv_file='cats_dogs.csv', root_dir='cats_dogs_resized',
                             transform=transforms.ToTensor())

train_set, test_set = torch.utils.data.random_split(dataset, [20000, 5000])
train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=batch_size)



# Model
model = torchvision.models.googlenet(weights="DEFAULT")

# freeze all layers, change final linear layer with num_classes
for param in model.parameters():
    param.requires_grad = False

# final layer is not frozen
model.fc = nn.Linear(in_features=1024, out_features=num_classes)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1, verbose=True)


def save_model_parameters(state, filename="my_model_parameters.pth.tar"):
    print("save to file model parameters")
    torch.save(state, filename)


def load_model_parameters(parameters):
    print("save to file model parameters")
    model.load_state_dict(parameters["state_dict"])
    optimizer.load_state_dict(parameters["optimizer"])


if load_modal:
    load_model_parameters(torch.load("my_model_parameters.pth.tar"))

# Train Network
for epoch in range(num_epochs):
    losses = []

    if epoch % 3:
        parameters = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        save_model_parameters(parameters)

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    scheduler.step()

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses)}")

# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()


print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on Test Set")
check_accuracy(test_loader, model)
