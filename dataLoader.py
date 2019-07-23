from torch.utils import data
import torch

tagDict = ['O', 'B-a', 'I-a', 'E-a', 'S-a', 'B-b', 'I-b', 'E-b', 'S-b', 'B-c', 'I-c', 'E-c', 'S-c']
int2tag = {index:element for index, element in enumerate(tagDict)}
tag2int = {element:index for index, element in enumerate(tagDict)}

class NERDataset(data.Dataset):
    '''
    : path 语料路径
    '''
    def __init__(self, path, config):
        self.config = config
        f = open(path, 'r', encoding='utf-8', errors='ignore')
        sentenceList, tagList = [], []
        sentence, tag = [], []
        for line in f.readlines():
            #换行
            if len(line.strip()) == 0:
                if len(sentence) != 0 and len(tag) != 0: 
                    if len(sentence) == len(tag):
                        sentenceList.append(sentence); tagList.append(tag)
                sentence, tag = [], []
            else:
                line = line.strip()
                if len(line.split('\t')) < 1: continue
                #针对submit
                if len(line.split('\t')) == 1: 
                    sentence.append(line); tag.append(int2tag[0]); continue
                sentence.append(line.split()[0])
                tag.append(line.split()[1])
        f.close()
        if len(sentence) != 0 and len(tag) != 0: 
            if len(sentence) == len(tag):
                sentenceList.append(sentence); tagList.append(tag)

        self.sentenceList, self.tagList = sentenceList, tagList
    
    def __len__(self):
        return len(self.sentenceList)

    def __getitem__(self, index):
        sentence, tag = self.sentenceList[index], self.tagList[index]

        sentence = [int(element) for element in sentence]
        tag = [tag2int[element] for element in tag]

        return sentence, tag

'''
进行填充
'''
def pad(batch):
    #句子最大长度
    f = lambda x:[element[x] for element in batch]
    lenList = [len(element) for element in f(0)]
    maxLen = max(lenList)
    
    f = lambda x, maxLen:[element[x] + [0] * (maxLen - len(element[x])) for element in batch]

    return torch.LongTensor(f(0, maxLen)), torch.LongTensor(f(1, maxLen)), lenList



    

