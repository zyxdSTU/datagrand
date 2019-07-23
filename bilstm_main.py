import yaml
from dataLoader import tagDict
from dataLoader import NERDataset
from dataLoader import pad
from torch.utils import data
from bilstm import BiLSTM
from tqdm import tgrange
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch
import sys
from seqeval.metrics import f1_score, accuracy_score, classification_report
from dataLoader import int2tag
from dataLoader import tag2int
from util import generateSubmit

def main(config):
    trainDataPath = config['data']['trainDataPath']
    validDataPath = config['data']['validDataPath']
    submitDataPath = config['data']['submitDataPath']
    submitPrePath = config['data']['submitPrePath']
    submitResultPath = config['data']['submitResultPath']

    batchSize = config['model']['batchSize']
    epochNum = config['model']['epochNum']
    earlyStop = config['model']['earlyStop']
    learningRate = config['model']['learningRate']
    modelSavePath = config['model']['modelSavePath']
    
    #GPU/CPU
    DEVICE = config['DEVICE']

    trianDataset = NERDataset(trainDataPath, config)
    validDataset = NERDataset(validDataPath, config)
    submitDataset = NERDataset(submitDataPath, config)
    
    trainIter = data.DataLoader(dataset = trianDataset,
                                 batch_size = batchSize,
                                 shuffle = True,
                                 num_workers = 4,
                                 collate_fn = pad)

    validIter = data.DataLoader(dataset = validDataset,
                                 batch_size = batchSize,
                                 shuffle = False,
                                 num_workers = 4,
                                 collate_fn = pad)

    submitIter = data.DataLoader(dataset = submitDataset,
                                 batch_size = batchSize,
                                 shuffle = False,
                                 num_workers = 4,
                                 collate_fn = pad)

    net = BiLSTM(config)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    net = net.to(DEVICE)

    lossFunction = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=learningRate,betas=(0.9, 0.999), eps=1e-08)

    earlyNumber, beforeLoss, maxScore = 0, sys.maxsize, -1
    for epoch in range(epochNum):
        print ('第%d次迭代' % (epoch+1))
        train(net, trainIter, optimizer=optimizer, criterion=lossFunction, DEVICE=DEVICE)
        totalLoss, f1Score, _, _, _ = eval(net,validIter,criterion=lossFunction, DEVICE=DEVICE)
        if f1Score > maxScore:
            maxScore = f1Score
            torch.save(net.state_dict(), modelSavePath)
        print ('验证损失为:%f   f1Score:%f / %f' % (totalLoss, f1Score, maxScore))
        if f1Score < maxScore:
            earlyNumber += 1
            print('earyStop: %d/%d' % (earlyNumber, earlyStop))
        else:
            earlyNumber = 0
        if earlyNumber >= earlyStop: break
        print ('\n')
    
    #加载最优模型
    net.load_state_dict(torch.load(modelSavePath))
    totalLoss, f1Score, preTags, _, sentences = eval(net, submitIter, criterion=lossFunction, DEVICE=DEVICE)

    #生成提交结果
    submitPre = open(submitPrePath, 'w', encoding='utf-8', errors='ignore')
    for tag, sentence in zip(preTags, sentences):
        for element1, element2 in zip(sentence, tag):
            submitPre.write(str(element1) + '\t' + element2 + '\n')
        submitPre.write('\n')
    submitPre.close()
    generateSubmit(submitPrePath=submitPrePath, submitResultPath=submitResultPath)


def train(net, iterData, optimizer, criterion, DEVICE):
    net.train()
    totalLoss = 0
    for batchSentence, batchTag, lenList in tqdm(iterData):
        batchSentence = batchSentence.to(DEVICE)
        batchTag = batchTag.to(DEVICE)
        net.zero_grad()
        #print (batchSentence)
        tagScores  = net(batchSentence)

        loss = 0
        for index, element in enumerate(lenList):
            tagScore = tagScores[index][:element]
            tag = batchTag[index][:element]
            loss +=  criterion(tagScore, tag)

        loss.backward()
        optimizer.step()
        totalLoss += loss.item()
    print('训练损失为 %f' % totalLoss)

def eval(net, iterData, criterion, DEVICE):
    net.eval()
    totalLoss = 0
    yTrue, yPre, ySentence = [], [], []
    for batchSentence, batchTag, lenList in tqdm(iterData):
        batchSentence = batchSentence.to(DEVICE)
        batchTag = batchTag.to(DEVICE)
        net.zero_grad()
        tagScores  = net(batchSentence)

        loss = 0
        for index, element in enumerate(lenList):
            tagScore = tagScores[index][:element]
            tag = batchTag[index][:element]
            sentence = batchSentence[index][:element]
            loss +=  criterion(tagScore, tag)
            yTrue.append(tag.cpu().numpy().tolist())
            ySentence.append(sentence.cpu().numpy().tolist())
            yPre.append([element.argmax().item() for element in tagScore])

        totalLoss += loss.item()

    yTrue2tag = [[int2tag[element2] for element2 in element1] for element1 in yTrue]
    yPre2tag = [[int2tag[element2] for element2 in element1] for element1 in yPre]
    f1Score = f1_score(y_true=yTrue2tag, y_pred=yPre2tag)

    return totalLoss, f1Score, yPre2tag, yTrue2tag, ySentence

    
if __name__ == "__main__":
    f = open('./config.yml', encoding='utf-8', errors='ignore')
    config = yaml.load(f)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['DEVICE'] = DEVICE
    f.close()
    main(config)
    
