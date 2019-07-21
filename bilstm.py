from dataLoader import tagDict
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wordDictSize = config['data']['wordDictSize']
        self.embeddingSize = config['model']['embeddingSize']
        self.hiddenSize = config['model']['hiddenSize']

        self.wordEmbeddings = nn.Embedding(self.wordDictSize, self.embeddingSize)

        self.lstm = nn.LSTM(input_size=self.embeddingSize, hidden_size= self.hiddenSize // 2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hiddenSize, len(tagDict))
    
    def forward(self, batchSentence):
        if self.training:
            #获得词嵌入
            #print (batchSentence.shape)
            embeds = self.wordEmbeddings(batchSentence)
            lstmFeature,_ = self.lstm(embeds)
            tagFeature = self.fc(lstmFeature)
            tagScores = F.log_softmax(tagFeature, dim=2)
        else:
            with torch.no_grad():
                embeds = self.wordEmbeddings(batchSentence)
                lstmFeature, _ = self.lstm(embeds)    
                tagFeature = self.fc(lstmFeature)
                tagScores = F.log_softmax(tagFeature, dim=2)
        return tagScores
        