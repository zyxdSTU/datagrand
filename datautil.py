'''
转换为BIOES或BIO标注数据
'''
def changeBIO(inputPath, outputPath, label = 'BIOES'):
    input = open(inputPath, 'r', encoding='utf-8', errors='ignore')
    output = open(outputPath, 'w', encoding='utf-8', errors='ignore')
    for line in input.readlines():
        arr1 = line.strip('\n').split()
        for element1 in arr1:
            element1Class = element1[-1]
            element1 = element1[:-2]
            arr2 = element1.split('_')

            if element1Class == 'o':
                for element2 in arr2:
                    output.write(element2 + '\t' + 'O' + '\n')
                continue

            if label == 'BIOES':
                if len(arr2) == 1: output.write(arr2[0] + '\t' + 'S-' + element1Class + '\n')

                for index2, element2 in  enumerate(arr2):
                    if index2 == 0: output.write(element2 + '\t' + 'B-' + element1Class + '\n'); continue
                    if index2 == len(arr2) -1: output.write(element2 + '\t' + 'E-' + element1Class + '\n'); continue
                    output.write(element2 + '\t' + 'I-' + element1Class + '\n')

            if label == 'BIO':
                for index2, element2 in  enumerate(arr2):
                    if index2 == 0: output.write(element2 + '\t' + 'B-' + element1Class + '\n'); continue
                    else: output.write(element2 + '\t' + 'I-' + element1Class + '\n'); continue
        output.write('\n')
    input.close()
    output.close()

'''
切分为训练集、测试集、验证集
'''
#切分数据集
import random
def cutData(originPath, trainPath, validPath, testPath, portion1 = 0.75, portion2 = 0.1):
    origin = open(originPath, 'r', encoding='utf-8', errors='ignore')
    train = open(trainPath, 'w', encoding='utf-8', errors='ignore')
    valid = open(validPath, 'w', encoding='utf-8', errors='ignore')
    test = open(testPath, 'w', encoding='utf-8', errors='ignore')
    

    sentenceList, tagList = [], []
    sentence, tag = [], []
    for line in origin.readlines():
        #换行
        if len(line.strip()) == 0:
            if len(sentence) != 0 and len(tag) != 0: 
                if len(sentence) == len(tag):
                    sentenceList.append(sentence); tagList.append(tag)
            sentence, tag = [], []
        else:
            line = line.strip()
            if len(line.split('\t')) < 2: continue
            sentence.append(line.split('\t')[0])
            tag.append(line.split('\t')[1])
    if len(sentence) != 0 and len(tag) != 0: 
        if len(sentence) == len(tag):
            sentenceList.append(sentence); tagList.append(tag)

    order = list(range(len(sentenceList)))
    random.shuffle(order)

    trainNumber1 = int(len(sentenceList) * portion1)
    trainNumber2 = trainNumber1 + int(len(sentenceList) * portion2)

    trainSentenceList = [sentenceList[index] for index in order[:trainNumber1]]
    validSentenceList = [sentenceList[index] for index in order[trainNumber1:trainNumber2]]
    testSentenceList = [sentenceList[index] for index in order[trainNumber2:]]

    trainTagList = [tagList[index] for index in order[:trainNumber1]]
    validTagList = [tagList[index] for index in order[trainNumber1:trainNumber2]]
    testTagList = [tagList[index] for index in order[trainNumber2:]]


    for sentence, tag in zip(trainSentenceList, trainTagList):
        for elementS, elementT in zip(sentence, tag):
            train.write(elementS + '\t' + elementT + '\n')
        train.write('\n')


    for sentence, tag in zip(validSentenceList, validTagList):
        for elementS, elementT in zip(sentence, tag):
            valid.write(elementS + '\t' + elementT + '\n')
        valid.write('\n')

    for sentence, tag in zip(testSentenceList, testTagList):
        for elementS, elementT in zip(sentence, tag):
            test.write(elementS + '\t' + elementT + '\n')
        test.write('\n')

    origin.close(); train.close()

cutData('./datagrand/data.bioes', './datagrand/train.bioes', './datagrand/valid.bioes', './datagrand/test.bioes', portion1 = 0.75, portion2=0.1)