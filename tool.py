#将语料转成BIOES标注
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

#changeBIO('./datagrand/train.txt', './baseline/train.bio', label='BIO')
#changeBIO('./datagrand/test.txt', './baseline/test.bio', label='BIO')

#转换一下test语料
def changeTest1(inputPath, outputPath):
    input = open(inputPath, 'r', encoding='utf-8', errors='ignore')
    output = open(outputPath, 'w', encoding='utf-8', errors='ignore')
    for line in input.readlines():
        arr = line.strip('\n').split('_')
        for element in arr: output.write(element + '\n')
        output.write('\n')
    input.close(); output.close()

def changeTest2(inputPath, outputPath):
    input = open(inputPath, 'r', encoding='utf-8', errors='ignore')
    output = open(outputPath, 'w', encoding='utf-8', errors='ignore')
    for line in input.readlines():
        if len(line.strip()) == 0: output.write('\n'); continue
        word = line.strip('\n').split('\t')[0]
        output.write(word + '\n')
    input.close(); output.close()

def changeTest3(inputPath, outputPath):
    input = open(inputPath, 'r', encoding='utf-8', errors='ignore')
    output = open(outputPath, 'w', encoding='utf-8', errors='ignore')
    for line in input.readlines():
        if len(line.strip()) == 0: output.write('\n'); continue
        word = line.strip('\n').split('\t')[0]
        tag = line.strip('\n').split('\t')[1]
        if tag == 'O': output.write(line.strip() + '\n'); continue
        labelType, classType = tag.split('-')[0], tag.split('-')[1]
        if labelType == 'E': labelType = 'I'
        if labelType == 'S': labelType = 'B'
        output.write(word + '\t' + labelType + '-' + classType + '\n')
    input.close(); output.close()

#changeTest2('./datagrand/test.bioes', './datagrand/test.change')
#changeBIO('./datagrand/test.txt', './datagrand/submit.bioes')

changeTest3('./datagrand/train.bioes', './baseline/train.bio')
changeTest3('./datagrand/test.bioes', './baseline/test.bio')

#切分数据集
import random
def cutData(originPath, trainPath, testPath, portion1 = 0.85):
    origin = open(originPath, 'r', encoding='utf-8', errors='ignore')
    train = open(trainPath, 'w', encoding='utf-8', errors='ignore')
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

    trainNumber = int(len(sentenceList) * portion1)

    trainSentenceList = [sentenceList[index] for index in order[:trainNumber]]
    testSentenceList = [sentenceList[index] for index in order[trainNumber:]]

    trainTagList = [tagList[index] for index in order[:trainNumber]]
    testTagList = [tagList[index] for index in order[trainNumber:]]


    for sentence, tag in zip(trainSentenceList, trainTagList):
        for elementS, elementT in zip(sentence, tag):
            train.write(elementS + '\t' + elementT + '\n')
        train.write('\n')


    for sentence, tag in zip(testSentenceList, testTagList):
        for elementS, elementT in zip(sentence, tag):
            test.write(elementS + '\t' + elementT + '\n')
        test.write('\n')

    origin.close(); train.close()

#cutData('./datagrand/data.bioes', './datagrand/train.bioes', './datagrand/test.bioes', portion1 = 0.85)

#获得词表长度
def acquireWordLength1(inputPathArr):
    maxNumber = 0
    for inputPath in inputPathArr:
        f = open(inputPath, 'r', encoding='utf-8', errors='ignore')
        for line in f.readlines():
            if len(line.strip()) == 0: continue
            number = int(line.strip().split('\t')[0])
            if number > maxNumber: maxNumber = number
        f.close()
    return maxNumber + 1

def acquireWordLength2(inputPathArr):
    maxNumber = 0
    for inputPath in inputPathArr:
        f = open(inputPath, 'r', encoding='utf-8', errors='ignore')
        for line in f.readlines():
            if len(line.strip()) == 0: continue
            numberArr = [int(element) for element in line.strip().split('_')]
            for element in numberArr: 
                if element > maxNumber: maxNumber = element
        f.close()
    return maxNumber + 1


#获得实体
def acquireEntity(sentence, tag, method='BIOES'):
    def suitable(tagA, tagB, method='BIOES'):
        tagA1, tagA2 = tagA.split('-')
        tagB1, tagB2 = tagB.split('-')
        if tagA2 != tagB2: return False

        if method == 'BIOES':
            if tagA1 == 'B':
                if tagB1 == 'E' or tagB1 == 'I': return True
            if tagA1 == 'E':
                return False
            if tagA1 == 'I':
                if tagB1 == 'E' or tagB1 == 'I': return True
            if tagA1 == 'S': return False

        if method == 'BIO':
            if tagA1 == 'B':
                if tagB1 == 'I': return True
            if tagA1 == 'I':
                if tagB1 == 'I': return True
        
        return False

    i, entityList, entity = 0, [], []
    while i < len(sentence):

        if tag[i] == 'O': i += 1; continue

        if len(entity) == 0:
            if tag[i].split('-')[0] == 'B':
                entity.append((sentence[i],tag[i]))

        else:
            if suitable(entity[-1][1],tag[i]):
                entity.append((sentence[i], tag[i]))
            else:
                if method == 'BIOES':
                    if entity[-1][1].split('-')[0] == 'E' or entity[-1][1].split('-')[0] == 'S':
                        entityTemp = '_'.join([element[0] for element in entity])
                        entityClass = entity[0][1].split('-')[1]
                        entityList.append((entityTemp, entityClass))

                elif method == 'BIO':
                    if entity[-1][1].split('-')[0] == 'B' or entity[-1][1].split('-')[0] == 'I':
                        entityTemp = '_'.join([element[0] for element in entity])
                        entityClass = entity[0][1].split('-')[1]
                        entityList.append((entityTemp, entityClass))

                entity = []
                if tag[i].split('-')[0] == 'B':
                    entity.append((sentence[i],tag[i]))
        i += 1
    if len(entity) != 0:
        if method == 'BIOES':
            if entity[-1][1].split('-')[0] == 'E' or entity[-1][1].split('-')[0] == 'S':
                entityTemp = '_'.join([element[0] for element in entity])
                entityClass = entity[0][1].split('-')[1]
                entityList.append((entityTemp, entityClass))
        elif method == 'BIO':
            if entity[-1][1].split('-')[0] == 'B' or entity[-1][1].split('-')[0] == 'I':
                entityTemp = '_'.join([element[0] for element in entity])
                entityClass = entity[0][1].split('-')[1]
                entityList.append((entityTemp, entityClass))

    f = lambda index:[element[index] for element in entityList]
    return f(0), f(1)
