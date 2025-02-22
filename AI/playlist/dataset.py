import torch
from torch.utils.data import TensorDataset, DataLoader
from tabulate import tabulate

def splitDf(total_df):
    traindf = total_df.sample(frac=0.9, random_state=80)
    testdf = total_df.drop(traindf.index)

    print('triandf 테이블')
    print(tabulate(traindf.head(), headers='keys', tablefmt='github'))
    print('testdf 테이블')
    print(tabulate(testdf.head(), headers='keys', tablefmt='github'))
    
    print("Train Dataset Size", traindf.shape[0])
    print("Test Dataset Size", testdf.shape[0])
    
    return traindf, testdf

def tensorData(train_ids, test_ids, traindf, testdf):
    '''
    데이터 텐서화 후 데이터셋 나누기
    '''
    train_ids = torch.tensor(train_ids)
    test_ids = torch.tensor(test_ids)
    print(f'train_ids: {train_ids}')
    
    train_labels = torch.tensor(traindf.label.values, dtype=torch.long)
    test_labels = torch.tensor(testdf.label.values, dtype=torch.long)
    print(f'train_lables : {train_labels}')
    
    trainDS = TensorDataset(train_ids, train_labels)
    testDS = TensorDataset(test_ids, test_labels)
    
    trainDL = DataLoader(trainDS, batch_size=50, shuffle=True)
    testDL = DataLoader(testDS, batch_size=50, shuffle=False)
    
    return trainDL, testDL