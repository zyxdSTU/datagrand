import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnn
import operator
from dataLoader import tagDict
from dataLoader import tag2int
from dataLoader import int2tag
from dataLoader import word2int
from dataLoader import int2word
from dataLoader import wordDict
from tqdm import tgrange
from tqdm import tqdm
from seqeval.metrics import f1_score, accuracy_score, classification_report

START_TAG = '<START>'
STOP_TAG = '<STOP>'

#计算log sum exp
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec-max_score_broadcast)))

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()

class BiLSTM_CRF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wordDictSize = len(wordDict)
        self.embeddingSize = config['model']['embeddingSize']
        self.hiddenSize = config['model']['hiddenSize']
        self.DEVICE = config['DEVICE']

        self.wordEmbeddings = nn.Embedding(self.wordDictSize, self.embeddingSize)

        self.tagDict = tagDict.copy();  self.tagDict.extend(['<START>', '<STOP>'])
        self.tag2int ={element:index for index, element in enumerate(self.tagDict)}
        self.int2tag ={index:element for index, element in enumerate(self.tagDict)}

        self.lstm = nn.LSTM(input_size=self.embeddingSize, hidden_size= self.hiddenSize // 2, batch_first=True, bidirectional=True, num_layers=2)
        self.fc = nn.Linear(self.hiddenSize, len(self.tagDict))

        #转移矩阵
        self.transitions = nn.Parameter(
            torch.randn(len(self.tagDict), len(self.tagDict)))

        #STOP_TAG转移, START_TAG转移到
        self.transitions.data[self.tag2int[STOP_TAG], :] = -100000
        self.transitions.data[:, self.tag2int[START_TAG]] = -100000

    #前向算法
    def _forward_alg(self, feats):
        #初始化size为 1xtagset_size,值为-10000的tensor
        alpha = torch.full((len(feats), len(self.tagDict)), -100000, dtype=torch.float32,device=self.DEVICE)

        #开始为START_TAG
        start_alpha = torch.full((1, len(self.tagDict)), -100000, dtype=torch.float32,device=self.DEVICE)
        start_alpha = torch.squeeze(start_alpha)
        start_alpha[self.tag2int[START_TAG]] = 0.

        for i in range(len(feats)):
            if i == 0: temp = start_alpha + self.transitions.t()
            else: temp = alpha[i-1] + self.transitions.t()
            alpha[i] = torch.cat([torch.unsqueeze(log_sum_exp(torch.unsqueeze(element, 0)), 0) for element in (temp + feats[i])]).to(self.DEVICE)
        
        stop_alpha = alpha[-1] + self.transitions[:, self.tag2int[STOP_TAG]]
        final_alpha = log_sum_exp(torch.unsqueeze(stop_alpha, 0))

        return final_alpha

    #得到发射概率·
    def _get_lstm_features(self, batchSentence):
        #对词进行嵌入
        embeds = self.wordEmbeddings(batchSentence)

        lstm_out, _ = self.lstm(embeds)

        lstm_feats = self.fc(lstm_out)

        return lstm_feats

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1, dtype=torch.float32, device=self.DEVICE)
        #score = torch.zeros(1)

        score = score + \
            self.transitions[self.tag2int[START_TAG], int(tags[0])] + feats[0][int(tags[0])]

        for i, feat in enumerate(feats):
            if i == len(feats) -1:
                score = score + self.transitions[int(tags[-1]), self.tag2int[STOP_TAG]]; continue

            score = score + \
                self.transitions[int(tags[i]), int(tags[i+1])] + feat[int(tags[i + 1])]

        return score

    def _viterbi_decode(self, feats):

        score = torch.full((len(feats), len(self.tagDict)), -100000, dtype=torch.float32, device=self.DEVICE)
        path = torch.zeros((len(feats), len(self.tagDict)), dtype=torch.int, device=self.DEVICE)

        start_score = torch.full((1, len(self.tagDict)), -100000, dtype=torch.float32, device=self.DEVICE)
        start_score = torch.squeeze(start_score)
        start_score[self.tag2int[START_TAG]] = 0

        for i in range(len(feats)):
            if i == 0:
                temp = start_score + self.transitions.t()
            else: temp = score[i-1] + self.transitions.t()
            path[i] = torch.argmax(temp, dim=1)

            for j, element in enumerate(temp):
                score[i][j] = element[int(path[i][j])] + feats[i][j]

        stop_score = score[-1] + self.transitions[:, self.tag2int[STOP_TAG]] 

        state = torch.full((1,len(feats)), 0, dtype=torch.float32, device=self.DEVICE)
        state = torch.squeeze(state)
        state[-1] = torch.argmax(stop_score)

        for index in range(0, len(feats)-1)[::-1]:
            state[index] = path[index+1][int(state[index+1])]

        assert path[0][1] == self.tag2int[START_TAG]

        #print (state)
        return state

    def neg_log_likelihood(self, batchSentence, batchTag, lenList):

        feats = self._get_lstm_features(batchSentence)

        #前向传播得到的分数
        forward_score = torch.cat([torch.unsqueeze(self._forward_alg(feat[:length]), 0) for feat, length in zip(feats, lenList)]).to(self.DEVICE)

        print ('forward_score: ', forward_score)


        #CRF计算特征函数得到的分数
        gold_score = torch.cat([self._score_sentence(feat[:length], tag[:length]) for feat, tag, length in zip(feats, batchTag, lenList)]).to(self.DEVICE)

        #gold_score1 = torch.cat([self._score_sentence_Test(feat[:length], tag[:length]) for feat, tag, length in zip(feats, batchTag, lenList)]).to(self.DEVICE)

        #print ('gold_score: ', gold_score)
        #print ('gold_score1: ', gold_score1)

        print ('forward_score', forward_score)
        print ('gold_score', gold_score)
        print ('transitions', self.transitions)

        return torch.mean(forward_score - gold_score)
        

    def forward(self, batchSentence, batchTag, lenList):  
        if self.training:
            loss = self.neg_log_likelihood(batchSentence, batchTag, lenList)
            return loss, None
        else:
            with torch.no_grad(): 
                loss = self.neg_log_likelihood(batchSentence, batchTag, lenList)
                feats = self._get_lstm_features(batchSentence)
                tag_seq = [self._viterbi_decode(feat[:length]) for feat, length in zip(feats, lenList)]
            return loss, tag_seq

def bilstmCRFTrain(net, iterData, optimizer, criterion, DEVICE):
    net.train()
    totalLoss = 0
    for batchSentence, batchTag, lenList in tqdm(iterData):
        batchSentence = batchSentence.to(DEVICE)
        batchTag = batchTag.to(DEVICE)
        net.zero_grad()
        loss, _ = net(batchSentence, batchTag, lenList)
        loss.backward()
        optimizer.step()

        totalLoss += loss.item()
    print (net.transitions)
    return totalLoss

def bilstmCRFEval(net, iterData, criterion, DEVICE):
    net.eval()
    totalLoss = 0
    yTrue, yPre, ySentence = [], [], []
    for batchSentence, batchTag, lenList in tqdm(iterData):
        batchSentence = batchSentence.to(DEVICE)
        batchTag = batchTag.to(DEVICE)
        
        loss, tagPre  = net(batchSentence, batchTag, lenList)
        tagPre = [element.cpu().numpy() for element in tagPre]
        tagTrue = [element[:length] for element, length in zip(batchTag.cpu().numpy(), lenList)]
        sentence = [element[:length] for element, length in zip(batchSentence.cpu().numpy(), lenList)]
        yTrue.extend(tagTrue); yPre.extend(tagPre); ySentence.extend(sentence)

        totalLoss += loss.item()

    yTrue2tag = [[int2tag[int(element2)] for element2 in element1] for element1 in yTrue]
    yPre2tag = [[int2tag[int(element2)] for element2 in element1] for element1 in yPre]

    f1Score = f1_score(y_true=yTrue2tag, y_pred=yPre2tag)

    return totalLoss, f1Score, yPre2tag, yTrue2tag, ySentence
  
