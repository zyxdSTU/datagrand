from seqeval.metrics import classification_report, f1_score
import codecs

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
#changeTest3('./datagrand/train.bioes', './baseline/train.bio')
#changeTest3('./datagrand/test.bioes', './baseline/test.bio')

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


#获得词表长度
def acquireWordLength1(inputPathArr):
    maxNumber = 0
    for inputPath in inputPathArr:
        f = open(inputPath, 'r', encoding='utf-8', errors='ignore')
        for line in f.readlines():
            if len(line.strip()) == 0: continue
            number = int(line.strip().split('\t')[0])
            if number == 0: print(True, line)
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
                if element == 0: print (True)
                if element > maxNumber: maxNumber = element
        f.close()
    return maxNumber + 1

#acquireWordLength1(['./datagrand/data.bioes'])
#acquireWordLength2(['./datagrand/test.txt'])


'''
输入预测数据、真实数据，输出F1-值
'''
def compF1Score(preDataPath, relDataPath):
    def getTags(dataPath):
        tags = []
        with codecs.open(dataPath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.read()
            lineArr = lines.split('\n\n')
            for line in lineArr:
                if len(line.strip()) == 0: continue
                tag = [element.split('\t')[1] for element in line.split('\n')]   
                tags.append(tag)
        return tags

    tagsPre = getTags(preDataPath)
    tagsRel = getTags(relDataPath)    
    f1Score = f1_score(y_true=tagsRel, y_pred=tagsPre)
    return f1Score

'''
生成submit代码
submitPrePath 对submit文本预测的效果
generatePath 符合提交格式的文本
'''
def generateSubmit(submitPrePath, submitResultPath):
    submit = open(submitResultPath, 'w', encoding='utf-8', errors='ignore')
    with codecs.open(submitPrePath, 'r', encoding='utf-8') as f:
        lines = f.read()
        lineArr = lines.split('\n\n')
        for line in lineArr:
            if len(line.strip()) == 0: continue
            sentence = [element.split('\t')[0] for element in line.split('\n')]
            tag = [element.split('\t')[1] for element in line.split('\n')]
            sentence1, tag1 = acquireEntity(sentence, tag)
            #没有识别出任何实体
            if len(sentence1) == 0 and len(tag1) == 0:
                submit.write('_'.join(sentence) + '/' + 'o' + '\n'); continue
            arr = [element1 + '/' + element2 for element1, element2 in zip(sentence1, tag1)]
            submit.write('  '.join(arr) + '\n')
    f.close()

#获取词典
def acquireWordDict(inputPathArr):
    wordDict = []
    for inputPath in inputPathArr:
        input = open(inputPath, 'r', encoding='utf-8', errors='ignore')
        for line in input.readlines():
            word = line.split('\t')[0]
            if word not in wordDict:
                wordDict.append(word)
        input.close()
    return wordDict