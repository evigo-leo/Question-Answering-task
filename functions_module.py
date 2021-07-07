from typing import List
import torch
from torch import nn
import numpy as np
from sklearn.metrics import accuracy_score
from IPython.display import clear_output
import time, math
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline


def transform_logits(predictions: List[torch.tensor]):
    return np.argmax(predictions.detach().cpu().numpy(), axis=1).flatten()


def transform_target(target_labels: List[torch.tensor]):
    return target_labels.to('cpu').numpy().flatten()


def plot(history=None, train_history=None, dev_history=None, score=None):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))

    clear_output(True)
    if history is not None:
        ax[0].plot(history, linewidth=2.5)
        ax[0].set_title('Train loss', fontsize=18)
        ax[0].set_xlabel('Batch')
        ax[0].set_ylabel('Loss')
        ax[0].xaxis.label.set_size(15)
        ax[0].yaxis.label.set_size(15)

    if train_history is not None:
        ax[1].plot(list(range(1, len(train_history) + 1)), train_history, label='train history', linewidth=2.5)
        ax[1].set_title('Train/Dev history', fontsize=18)
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Loss')
        ax[1].xaxis.label.set_size(15)
        ax[1].yaxis.label.set_size(15)

    if dev_history is not None:
        ax[1].plot(list(range(1, len(dev_history) + 1)), dev_history, label='dev history', linewidth=2.5)

    if score is not None:
        ax[2].plot(list(range(1, len(score) + 1)), score, marker='o', c='red', linewidth=2.5, markersize=8)
        ax[2].set_title('Accuracy dev', fontsize=18)
        ax[2].set_xlabel('Epoch')
        ax[2].set_ylabel('Accuracy score')
        ax[2].xaxis.label.set_size(15)
        ax[2].yaxis.label.set_size(15)
    ax[1].legend(prop={"size": 15})
    plt.show()


def train(model, iterator, optimizer, clip, plot, train_history, dev_history, score):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()

    epoch_loss = 0
    history = []

    for i, batch in enumerate(iterator):

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)

        optimizer.zero_grad()
        loss = model.forward(input_ids, attention_mask, token_type_ids, labels)[1]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

        history.append(loss.cpu().data.numpy())
        if (i + 1) % 10 == 0:
            plot(history, train_history, dev_history, score)

    return epoch_loss / len(iterator), history


def evaluate(model, iterator):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    epoch_loss = 0

    gold_labels = []
    predict_labels = []

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            logits, loss = model.forward(input_ids, attention_mask, token_type_ids, labels)
            # print('loss\n', logits, 'logits\n', logits)
            epoch_loss += loss.item()

            gold_labels.append(transform_target(labels))
            predict_labels.append(transform_logits(logits))
            # print(gold_labels, predict_labels)
            epoch_loss += loss.item()

        acc_score = accuracy_score(np.concatenate(gold_labels), np.concatenate(predict_labels))

    return epoch_loss / len(iterator), acc_score


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_eval_loop(model, train, tr_dataloader, eval, dev_dataloader, optimizer, n_epochs, clip, plot, path):
    train_history = []
    dev_history = []
    score = []

    best_dev_loss = float('inf')

    for epoch in range(n_epochs):
        
        start_time = time.time()
        
        train_loss, history = train(model, tr_dataloader, optimizer, clip, plot, train_history, dev_history, score)
        dev_loss, acc_score = evaluate(model, dev_dataloader)
        #scheduler.step()    

        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), 'best-dev-model.pt')
        
        train_history.append(train_loss)
        dev_history.append(dev_loss)
        score.append(acc_score)

        plot(history, train_history, dev_history, score) 
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Dev. Loss: {dev_loss:.3f} |  Dev. PPL: {math.exp(dev_loss):7.3f}')
        print(f'\t MAX SCORE: {max(score)}')
    return score