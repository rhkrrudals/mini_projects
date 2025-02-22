import torch
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt


def train(model, datasets, criterion, optimizer, device, interval):
    model.train()
    train_losses = list()
    train_corrects = list()

    for step, (input_ids, labels) in enumerate(datasets):
        input_ids = input_ids.to(device)
        labels = labels.to(device).float()
        logits = model(input_ids)
        logits = logits.squeeze(dim=1)

        loss = criterion(logits, labels)
        #print(f'loss : {loss}')
        train_losses.append(loss.item())
        
        yhat = torch.sigmoid(logits)  # [batch_size, 1] -> [batch_size]
        yhat = (yhat > 0.5).float()  # 0.5 기준으로 이진 분류

        train_corrects.extend((yhat == labels).cpu().tolist())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % interval == 0:
            avg_loss = np.mean(train_losses)
            avg_accuracy = np.mean(train_corrects)
            print(f"Train {step} - Loss: {avg_loss:.4f}         Accuracy: {avg_accuracy:.4f}")
            
    avg_loss = np.mean(train_losses)
    avg_accuracy = np.mean(train_corrects)        
    
    return avg_accuracy, avg_loss

def test(model, datasets, criterion, device, scheduler):
    model.eval()
    test_losses = list()
    test_corrects = list()
    all_yhat = list()
    all_labels = list()
   
    for step, (input_ids, labels) in enumerate(datasets):
        input_ids = input_ids.to(device)
        labels = labels.to(device).float()

        logits = model(input_ids)
        # print(f'Before Squeeze : {logits}')
        logits = logits.squeeze(dim=1)
        # print(f'After Squeeze: {logits}')
        
        loss = criterion(logits, labels)
        test_losses.append(loss.item())

        yhat = torch.sigmoid(logits)  # [batch_size, 1] -> [batch_size]
        yhat = (yhat > 0.5).float()  # 0.5 기준으로 이진 분류

        test_corrects.extend((yhat == labels).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_yhat.extend(yhat.cpu().tolist())
        
    avg_loss = np.mean(test_losses)
    avg_accuracy = np.mean(test_corrects)
    
    print(f"Test -  Loss: {avg_loss:.4f}    Accuracy: {avg_accuracy:.4f}")
    print('-'*100)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_yhat))
    
    scheduler.step(avg_accuracy)
    
    return avg_accuracy, avg_loss

def drawGraph(train_losses,train_accuracies,test_losses,test_accuracies,FILE_PATH):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.title('Loss')
    plt.plot(train_losses,label='Train Loss', alpha=0.7)
    plt.plot(test_losses,label='Test Loss', alpha=0.7)
    plt.xlabel('Steps'); plt.ylabel('Loss')
    plt.grid()
    plt.legend()

    plt.subplot(1,2,2)
    plt.title('Accuracy')
    plt.plot(train_accuracies,label='Train Accuracy', alpha=0.7)
    plt.plot(test_accuracies,label='Test Accuracy', alpha=0.7)
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(f'{FILE_PATH}/loss_accuracy')
    plt.show()

