from dataLoader import tagDict
from dataLoader import tag2int
from dataLoader import int2tag
from tqdm import tgrange
from tqdm import tqdm
from seqeval.metrics import f1_score, accuracy_score, classification_report
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wordDictSize = config['data']['wordDictSize']
        self.embeddingSize = config['model']['embeddingSize']
        self.encoderNumber = config['model']['encoderNumber']
        self.headNumber = config['model']['headNumber']
        self.feedforwardSize = config['model']['feedforwardSize']
        self.dropout = config['model']['dropout']

        attn = MultiHeadedAttention(self.headNumber, self.embeddingSize)
        ff = PositionwiseFeedForward(self.embeddingSize, self.feedforwardSize, self.dropout)
        position  = PositionalEncoding(self.embeddingSize, self.dropout)
        embedding = nn.Sequential(Embeddings(self.embeddingSize, self.wordDictSize), position)
        self.encoder = Encoder(EncoderLayer(self.embeddingSize, attn, ff, self.dropout), self.encoderNumber, embedding)
        self.fc = nn.Linear(self.embeddingSize, len(tagDict))

    def forward(self, batchSentence):
        if self.training:
            #获得词嵌入
            mask = (batchSentence != 0).unsqueeze(-2)
            encoderFeature = self.encoder(batchSentence, mask)
            tagFeature = self.fc(encoderFeature)
            tagScores = F.log_softmax(tagFeature, dim=2)
        else:
            with torch.no_grad():
                mask = (batchSentence != 0).unsqueeze(-2)
                encoderFeature = self.encoder(batchSentence, mask)
                tagFeature = self.fc(encoderFeature)
                tagScores = F.log_softmax(tagFeature, dim=2)
        return tagScores

def transformerTrain(net, iterData, optimizer, criterion, DEVICE):
    net.train()
    totalLoss = 0
    for batchSentence, batchTag, lenList in tqdm(iterData):
        batchSentence = batchSentence.to(DEVICE)
        batchTag = batchTag.to(DEVICE)
        net.zero_grad()
        tagScores  = net(batchSentence)

        loss = 0
        for index, element in enumerate(lenList):
            tagScore = tagScores[index][:element]
            tag = batchTag[index][:element]
            loss +=  criterion(tagScore, tag)

        loss.backward()
        optimizer.step()
        totalLoss += loss.item()
    return totalLoss

def transformerEval(net, iterData, criterion, DEVICE):
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
  
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

#层正则
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N, embedding):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.embedding =  embedding
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

#多头注意力机制
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                    dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
                .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

#词嵌入
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                                -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                            requires_grad=False)
        return self.dropout(x)