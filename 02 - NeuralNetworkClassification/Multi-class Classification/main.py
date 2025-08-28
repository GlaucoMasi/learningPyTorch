import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary

NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

X_blob, y_blob = make_blobs(n_samples=1000,
                            n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.5,
                            random_state=RANDOM_SEED)

X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob, y_blob,
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)

plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)


############################
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()

        # Is non-linearity needed?
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )
    
    def forward(self, x):
        return self.linear_layer_stack(x)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_true))*100
    return acc
############################


model = BlobModel(input_features=NUM_FEATURES,
                  output_features=NUM_CLASSES,
                  hidden_units=8)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# For every feature the model outputs one value for each class
# By using the softmax activation function we can remap this values to a probability distribution
# So that for each feature we have a probability assigned to each class so that the total sum is 1
print(torch.softmax(model(X_blob_test), dim=1)[:5])
print(torch.argmax(torch.softmax(model(X_blob_test), dim=1), dim=1)[:5])


torch.manual_seed(42)
epochs = 100

for epoch in range(epochs):
    model.train()

    y_logits = model(X_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, y_blob_train)
    acc = accuracy_fn(y_true=y_blob_train, y_pred=y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch%10 == 0:
        model.eval()
        y_logits = model(X_blob_test)
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

        test_loss = loss_fn(y_logits, y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test, y_pred=y_pred)

        print(f"Epoch {epoch} | Loss {loss:.5f}/{test_loss:.5f} | Accuracy {acc:.2f}/{test_acc:.2f}%")

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.title("Train")
        plot_decision_boundary(model, X_blob_train, y_blob_train)
        plt.subplot(1, 2, 2)
        plt.title("Test")
        plot_decision_boundary(model, X_blob_test, y_blob_test)

# There are other evaluation metrics besides accuracy
from torchmetrics import Accuracy
torchmetrics_accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES)

model.eval()
y_logits = model(X_blob_test)
y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
print(torchmetrics_accuracy(y_pred, y_blob_test))

from sklearn.metrics import classification_report
print(classification_report(y_true=y_blob_test, y_pred=y_pred))