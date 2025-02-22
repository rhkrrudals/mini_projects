# 데이터 텐서화 모듈 
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim.lr_scheduler as lr_schedular
from dataset import *
from train import * 
from model import *

# 데이터 전처리 모듈
import os 
import re 
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from konlpy.tag import Okt
from collections import Counter

def makeDir(FILE_PATH, name):
    if os.path.exists(f'{FILE_PATH}{name}'): print('파일이 이미 존재합니다.')
    else : os.makedirs(f'{FILE_PATH}{name}')

def makeDf(FILE_PATH, ToCSV=False):
    '''
        xlsx --> DataFrame
    '''
    df_dict = list()
    file_list = os.listdir(FILE_PATH)
    for file in file_list:
        filename, ext = os.path.splitext(file)
        if ext == '.xlsx':
            df = pd.read_excel(FILE_PATH+file, engine='openpyxl')
            df_dict.append(df)
    df =  pd.concat(df_dict).drop(columns='Unnamed: 0').rename(columns={'댓글 내용':'text'})
    # print(tabulate(df.head(),headers='keys',tablefmt='github'))
    
    if ToCSV: 
        makeDir(FILE_PATH,'df_data')
        df.to_csv(f'{FILE_PATH}df_data/{filename}')
    return df

def reText(text):
    '''
    정규 표현식 불용어 제거 
    '''
    text = re.sub(r'[^\w가-힇]','', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r"[ぁ-んァ-ン一-龥]", "", text)
    # 공백 제거 한 text 반환
    return text.strip()

def getTextsLabels(df, texts, labels):
    '''
    데이터 프레임 내 댓글 내용 정제 후 리스트에 담아주기
    ''' 
    for text, label in df[['text', 'label']].values:
        if type(text) == str and len(text) >=2:
            # 130자만 들고오기 
            text = text[:130]
            text = reText(text)
            texts.append(text)
            labels.append(label)

def makeStopwords(WORD_PATH):
    with open(WORD_PATH, 'r',encoding='utf-8') as f:
        stopwords = f.read().splitlines()
    return set(stopwords)


def tokenizeCorpus(train_df, test_df, tokenizer, WORD_PATH, checkLength=False, checkWord=False):
    '''
    corpus(말뭉치) 토큰화하기
    '''
    stopwords = makeStopwords(WORD_PATH)
    
    train_tokens =[
        [   
            token
            for token in tokenizer.morphs(corpus)
            if token not in stopwords and len(token) >=2 
        ]
        for corpus in train_df.text ]
    
    test_tokens =[
        [
            token
            for token in tokenizer.morphs(corpus)
            if token not in stopwords and len(token) >=2
        ]
        for corpus in test_df.text ]
    
        
    if checkLength:
        token_len = [len(token) for token in train_tokens] 
        # print(f'token_length: {token_len}')
        plt.hist(token_len, bins=5)
        plt.show()
    
    if checkWord:
        check_tokens = [token for tokens in train_tokens for token in tokens]
        token_count = Counter(check_tokens)
        
        # 빈도 많은 단어 --> 불용어 보이면 stopword에 추가 
        print(f'The most common words (50개): {token_count.most_common(50)}')
        
    return train_tokens, test_tokens, token_len

def build_vocab(corpus, n_vocab,  special_tokens=["<pad>", "<unk>"]):
    counter = Counter()
    for tokens in corpus:
        counter.update(tokens)
    vocab = special_tokens
    for token, cnt in counter.most_common(n_vocab):
        vocab.append(token)
    return vocab

def makeVocab(corpus, n_vocab, special_tokens=['<pad>','<unk>'], Tojson=False):

    vocab = build_vocab(corpus, n_vocab,special_tokens)
    
    # 토큰 이름 : 토큰 라벨
    token_to_id = {token:idx for idx,token in enumerate(vocab)}
    # 토큰 라벨 : 토큰 이름
    id_to_token = {idx:token for idx, token in enumerate(vocab)}
    
    if Tojson: 
        with open('./playlist/train_vocab.json', 'w') as f: json.dump(vocab,f)
    
    return token_to_id, id_to_token
    
def pad_sequences(sequences, max_length, pad_value):
    result = list()
    for sequence in sequences:
        sequence = sequence[:max_length]
        pad_length = max_length - len(sequence)
        padded_sequence = sequence + [pad_value] * pad_length
        result.append(padded_sequence)
    
    return np.asarray(result)

def padding(token_to_id,  train_tokens, test_tokens, max_length):
    ukd_id = token_to_id['<unk>']
    pad_id = token_to_id['<pad>']
    
    train_ids = [[token_to_id.get(token,ukd_id) for token in corpus]for corpus in train_tokens]
    test_ids = [[token_to_id.get(token, ukd_id) for token in corpus] for corpus in test_tokens]

    train_ids = pad_sequences(train_ids, max_length=max_length, pad_value=pad_id)
    test_ids = pad_sequences(test_ids, max_length=max_length, pad_value=pad_id)
    
    return train_ids, test_ids

if __name__ == '__main__':
    FILE_PATH = './playlist/'
    SAD_FILE_PATH = f'{FILE_PATH}data/0/'
    HAPPY_FILE_PATH = f'{FILE_PATH}data/1/'
    WORD_PATH = f'{FILE_PATH}kor_stopwords.txt'
    
    sad_df = makeDf(SAD_FILE_PATH)
    happy_df = makeDf(HAPPY_FILE_PATH)
    print('데이터 프레임 만들기 완료!')
    
    texts = []
    labels = []
    getTextsLabels(sad_df,texts,labels)
    getTextsLabels(happy_df,texts,labels)
    print('데이터 정제 완료!')
    
    total_df = pd.DataFrame({'text':texts, 'label':labels})
    train_df, test_df = splitDf(total_df)
    print('데이터셋 나누기 완료!')
    
    print('토큰화 시작!')
    tokenizer = Okt()
    train_tokens, test_tokens, token_len = tokenizeCorpus(train_df,test_df,tokenizer,WORD_PATH,checkLength=True)
    print('토큰화 완료!')
    token_to_id, id_to_token = makeVocab(train_tokens,500,Tojson=True)
    print('단어사전 생성 완료!')
    train_ids, test_ids = padding(token_to_id, train_tokens, test_tokens,max(token_len))
    print('패딩 완료!')
    trainDL, testDL = tensorData(train_ids,test_ids,train_df,test_df)
    print('데이터셋 만들기 완료!')

    # model 
    print('모델학습 시작!')
    device = 'mps:0' if torch.backends.mps.is_available() else 'cpu' 
    n_vocab = len(token_to_id)
    hidden_dim = 64
    embedding_dim = 128
    n_layers = 4
    n_classes = 1

    classifier = SentenceClassifier(n_vocab=n_vocab,hidden_dim=hidden_dim,embedding_dim=embedding_dim,
                                    n_layers=n_layers,n_classes=n_classes).to(device)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    scheduler = lr_schedular.ReduceLROnPlateau(optimizer, mode="max", patience=50)

    epochs = 50
    interval = 100
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    best_loss = float("inf")
    best_accuracy = 0  
    SAVE_PATH=f'{FILE_PATH}pth/'

    for epoch in range(epochs):
        print(f"EPOCH {epoch+1}{'-' * 90}")
        
        train_accuracy,train_loss  = train(classifier, trainDL, criterion, optimizer, device, interval)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        test_accuracy, test_loss  = test(classifier, testDL, criterion, device, scheduler)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        if test_loss < best_loss or test_accuracy > best_accuracy:
            best_loss = test_loss
            best_accuracy = test_accuracy
            
        if scheduler.num_bad_epochs >= scheduler.patience:
            SAVE_FILE = f'model_acuuracy({best_accuracy:.4f})_loss({test_loss:.4f}).pth'
            torch.save(classifier.state_dict(), SAVE_PATH+SAVE_FILE)
            print(f'최적 모델 저장: {SAVE_FILE}')
            print(f'{scheduler.patience} EPOCH 성능 개선 없어서 종료함!')
            break
        
        if epoch == epochs-1:
            SAVE_FILE = f'model_acuuracy({best_accuracy:.4f})_loss({test_loss:.4f}).pth'
            torch.save(classifier.state_dict(), SAVE_PATH+SAVE_FILE)
            print(f'최적 모델 저장: {SAVE_FILE}')
            print('학습 종료!')
    
    drawGraph(train_losses,train_accuracies,test_losses,test_accuracies,FILE_PATH)
    

        
    