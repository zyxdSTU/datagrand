from dataLoader import tagDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnn
import operator
from dataLoader import tagDict

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
        self.wordDictSize = config['data']['wordDictSize']
        self.embeddingSize = config['model']['embeddingSize']
        self.hiddenSize = config['model']['hiddenSize']

        self.wordEmbeddings = nn.Embedding(self.wordDictSize, self.embeddingSize)

        self.tagDict = tagDict.copy; self.tagDict.extend(['<START>', '<STOP>'])
        self.tag2int ={element:index for index, element in enumerate(self.tagDict)}
        self.int2tag ={index:element for index, element in enumerate(self.tagDict)}

        self.lstm = nn.LSTM(input_size=self.embeddingSize, hidden_size= self.hiddenSize // 2, batch_first=True, bidirectional=True, num_layers=4)
        self.fc = nn.Linear(self.hiddenSize, len(self.tagDict))

        #转移矩阵
        self.transitions = nn.Parameter(
            torch.randn(len(self.tagDcit), len(self.tagDict)))

        #STOP_TAG转移, START_TAG转移到
        self.transitions.data[self.tag2int[STOP_TAG], :] = -10000
        self.transitions.data[:, self.tag2int[START_TAG]] = -10000

    #前向算法
    def _forward_alg(self, feats):
        #初始化size为 1xtagset_size,值为-10000的tensor

        alpha = torch.full((len(feats) + 2, len(self.tagDict)), -1000)

        #开始为START_TAG
        alpha[0][self.tag2int[START_TAG]] = 0

        for index, feat in enumerate(feats):
            alpha[index] = torch.cat([log_sum_exp(element) for element in (alpha[index-1] + self.transitions.t)] + feat)
        
        alpha[-1] = alpha[-2] + self.transitions[:self.tag2int[STOP_TAG]]

        return log_sum_exp(alpha[-1])


    #得到发射概率·
    def _get_lstm_features(self, batchSentence):
        #对词进行嵌入
        embeds = self.word_embeds(batchSentence)

        lstm_out, _ = self.lstm(embeds)

        lstm_feats = self.fc(lstm_out)

        return lstm_feats

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        for i, feat in enumerate(feats):
            if i == 0:
                score = score + \
                    self.transitions[self.transitions[self.tag2int[START_TAG]], tags[i+1]] + feat[tags[i + 1]]; continue

            score = score + \
                self.transitions[tags[i], tags[i+1]] + feat[tags[i + 1]]

        score = score + self.transitions[tags[-1], self.tag2int[STOP_TAG]]
        return score

    def _viterbi_decode(self, feats):

        score = torch.full((len(feats)+ 2, len(self.tagDict)), -1000)
        path = torch.zeors((len(feats)+ 2, len(self.tagDict)))

        score[0][self.tag2int[START_TAG]] = 0
        for index, feat in enumerate(feats):
            temp = score[index-1] + self.transitions.t
            path[index] = torch.argmax(temp, axis=1)
            score[index] = [element[int(path[i])] for i, element in enumerate(temp)] + feat
        
        score[-1] = score[-2] + self.transitions[self.tag2int[STOP_TAG]]

        state = torch.zeros(len(feats))
        state[-1] = argmax(score[-1])

        for index in range(len(feats)-1)[::-1]:
            state[index] = path[index+1][int(state[index+1])]

        return state

    def neg_log_likelihood(self, batchSentence, batchTag):

        feats = self._get_lstm_features(batchSentence)

        #前向传播得到的分数
        forward_score = torch.cat([self._forward_alg(feat) for feat in feats])
    
        #CRF计算特征函数得到的分数
        gold_score = torch.cat([self._score_sentence(feat, tag) for feat, tag in zip(feats, batchTag)])

        return sum(forward_score - gold_score)
        

    def forward(self, batchSentence):  
        # 获取发射概率
        feats = self._get_lstm_features(batchSentence)
        
        tag_seq = torch.cat([self._viterbi_decode(feat) for feat in feats])
        
        return tag_seq


