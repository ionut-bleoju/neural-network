"""Train a simple neural network on the MNIST dataset"""
from torch.utils.data import DataLoader
from torch import nn
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
import pandas as pd

# Constants
NUM_GENERATIONS = 10
NUM_INPUT_NEURONS = 784
NUM_HIDDEN_NEURONS1 = 256
NUM_HIDDEN_NEURONS2 = 126
NUM_OUTPUT_NEURONS = 10
BATCH_SIZE = 50
RANDOM_SEED = 23
DECAY_FACTOR = 0.001

torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Sequential(
    nn.Linear(NUM_INPUT_NEURONS, NUM_HIDDEN_NEURONS1),
    nn.BatchNorm1d(NUM_HIDDEN_NEURONS1),
    nn.LeakyReLU(),
    nn.Linear(NUM_HIDDEN_NEURONS1, NUM_HIDDEN_NEURONS2),
    nn.BatchNorm1d(NUM_HIDDEN_NEURONS2),
    nn.LeakyReLU(),
    nn.Dropout(0.1),
    nn.Linear(NUM_HIDDEN_NEURONS2, NUM_OUTPUT_NEURONS),
)

model = model.to(device)


def flatten_tensor(x):
    """Flatten the tensor"""
    return x.view(-1)


train_transforms = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(scale=(0.5, 1.5), degrees=10),
    transforms.ToTensor(),
    transforms.Lambda(flatten_tensor)
])
train_dataset = MNIST(root='./data', train=True,
                      download=True, transform=train_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=10,
)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(flatten_tensor)
])
test_dataset = MNIST(root='./data', train=False,
                     download=True, transform=test_transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=500,
    num_workers=10,
)

optimizer = torch.optim.Adam(model.parameters(), lr=DECAY_FACTOR, weight_decay=DECAY_FACTOR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
)

ce_loss = nn.CrossEntropyLoss()


def train_model():
    """Train the model"""

    model.train()
    mean_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = ce_loss(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        mean_loss += loss.item()

    mean_loss /= len(train_loader)
    return mean_loss


def test_model():
    """Test the model"""
    model.eval()

    test_lost = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = ce_loss(outputs, labels)
            test_lost += loss.item()

            values, predictions = torch.max(outputs, 1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    test_lost /= len(test_loader)
    accuracy = correct / total
    return test_lost, accuracy


def generate_csv():
    """Generate the CSV file for submission"""
    model.eval()

    results = {
        "ID": [],
        "target": []
    }

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            values, predictions = torch.max(outputs, 1)

            results["ID"].extend(range(i * labels.size(0), (i+1) * labels.size(0)))
            results["target"].extend(predictions.tolist())

    pd_dataframe = pd.DataFrame(results)
    pd_dataframe.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    for epoch in range(NUM_GENERATIONS):
        train_loss = train_model()
        test_loss, test_accuracy = test_model()
        scheduler.step(test_loss)
        print(
            f"Accuracy: {test_accuracy*100:.3f}")
    generate_csv()
