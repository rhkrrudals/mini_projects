import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, dataloader

import pandas as pd
import numpy as np
from konlpy.tag import Okt
from collections import Counter 
import re
import json

from model import *
from playlist import reText, pad_sequences

def test(model):
    model.eval()
    happy = ''
    sad = ''
    test_id = test_ids

    logits = model(test_id)
    logits = logits.squeeze(dim=1)

    yhat = torch.sigmoid(logits)  # [batch_size, 1] -> [batch_size]s
    yhat = (yhat > 0.5).float()  # 0.5 기준으로 이진 분류
    
    happy_count = (yhat==1).sum().item()
    total_count = yhat.numel()
    
    if happy_count > total_count/2:
        happy = '해피 플리'
        print(happy)
    else:
        sad = '우울 플리'
        print(sad)

    return happy, sad

if __name__ == '__main__':
    device = 'cpu'
    test_tokens = input('지금 기분이 어떠신가요?: ')
    # test_tokens = '하 인생 쓰다'
    # json 파일로 불러오기
    with open('./playlist/train_vocab.json', 'r') as f:
        vocab = json.load(f)
        
    # vocab 담을 딕셔너리 생성
    vocab_dict = {}

    for idx, word in enumerate(vocab):
        vocab_dict[word] = idx
        
    ukd_id = vocab_dict['<unk>']
    test_ids = [[vocab_dict.get(token, ukd_id)for token in text ]for text in test_tokens]
    max_length = 35

    pad_id = vocab_dict['<pad>']
    test_ids = pad_sequences(test_ids, max_length=max_length, pad_value=pad_id)
    test_ids = torch.tensor(test_ids)

    # 예측하기
    n_vocab = len(vocab_dict)
    hidden_dim = 64
    embedding_dim = 128
    n_layer = 4
    n_classes = 1

    classifier = SentenceClassifier(n_vocab=n_vocab,hidden_dim=hidden_dim,embedding_dim=embedding_dim,
                                    n_layers=n_layer,n_classes=n_classes).to(device=device)
    criterion = torch.nn.BCEWithLogitsLoss().to(device=device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    model = SentenceClassifier(n_vocab=n_vocab,hidden_dim=hidden_dim,embedding_dim=embedding_dim,
                               n_layers=n_layer,n_classes=n_classes).to(device=device)

    # 저장된 state_dict 로드
    state_dict = torch.load('./playlist/pth/model_acuuracy(0.7775)_loss(0.6342).pth', weights_only=True)
    # state_dict 로드
    model.load_state_dict(state_dict, strict=False)
    
    test(model)

