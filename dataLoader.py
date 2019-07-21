tagDict = ['B-a', 'I-a', 'E-a', 'S-a', 'B-b', 'I-b', 'E-b', 'S-b', 'B-c', 'I-c', 'E-c', 'S-c', 'O']
int2tag = {index:element for index, element in enumerate(tagDict)}
tag2int = {element:index for index, element in enumerate(tagDict)}

def prepareData(inputPath):
    f = open(inputPath, encoding='utf-8', errors='ignore')
    sentenceList, tagList, sentence, tag = [], [], [], []
    for line in f.readlines():
        if len(line.strip()) == 0:
            if len(sentence) != 0 and len(tag) != 0:
                sentenceList.append(sentence)
                tagList.append(tag)
        else:
            sentence.append(int(line.strip().split('\t')[0]))
            tag.append(tag2int[line.strip().split('\t')[1]])
    
    if len(sentence) != 0 and len(tag) != 0:
        sentenceList.append(sentence)
        tagList.append(tag)
    f.close()
    return sentenceList, tagList