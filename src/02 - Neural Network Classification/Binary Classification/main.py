import torch
from data_loader import load_circles, load_line
from torch import nn
from model import CircleModelV0, CircleModelV1, CircleModelV2
import config as cfg
import matplotlib.pyplot as plt

# Helper functions from the Learn PyTorch for Deep Learning repo (https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py)
from helper_functions import plot_predictions, plot_decision_boundary

def main():
    X_train, X_test, y_train, y_test = load_circles()

    model_2 = CircleModelV2().to(cfg.device)
    # It's easier with nn.Sequential, but it only runs in sequential order
    # model = nn.Sequential(nn.Linear(...), nn.Linear(...)).to(cfg.device)

    # Binary Cross Entropy loss function: Hp(q) = -1/N * sum(yi*log(p(i)) + (1-yi)*log(1-p(i)))
    # We have two classes, class 0 and 1; p(i) is the probability that a feature belongs to class 1
    # This implementation has already a built-in sigmoid layer and is more stable than torch.nn.BCELoss()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)

    for epoch in range(cfg.epochs):
        model_2.train()

        y_logits = model_2(X_train.to(cfg.device)).squeeze()
        y_pred = normalize(y_logits)

        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model_2.eval()
        with torch.inference_mode():
            test_logits = model_2(X_test.to(cfg.device)).squeeze()
            test_pred = normalize(test_logits)

            test_loss = loss_fn(test_pred, y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

            if epoch%100 == 0:
                print(f"Epoch: {epoch} | Loss/Test loss: {loss:.5f}/{test_loss:.5f} | Accuracy/Test accuracy: {acc:.2f}%/{test_acc:.2f}%")
                # Run code in interactive window to see plotted data
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.title("Train")
                plot_decision_boundary(model_2, X_train, y_train)
                plt.subplot(1, 2, 2)
                plt.title("Test")
                plot_decision_boundary(model_2, X_test, y_test)
                plt.show()



def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_true))*100
    return acc


# The raw outputs of the forward pass are called logits, they are unnormalised predictions of a model
# To make sense of them, it can be useful to use a sigmoid activation function, that will map it to a probability from 0 to 1
# At this point it's enough to round them, because if the prediction is >= 0.5, then our model thinks that the sample belongs to class 1
def normalize(y_pred):
    y_pred_probs = torch.sigmoid(y_pred)
    return torch.round(y_pred_probs)


if __name__ == "__main__":
    main()
