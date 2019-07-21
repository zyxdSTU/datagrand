import yaml
from dataLoader import prepareData
from dataLoader import tagDict
from HMM import HMM
from CRF import CRFModel

def main(config):
    trainPath = config['data']['trainPath']
    testPath = config['data']['testPath']
    wordLength = config['data']['wordLength']

    #训练集数据
    trainWordLists, trainTagLists = prepareData(trainPath)

    #测试集数据
    testWordLists, testTagLists = prepareData(testPath)

    # #HMM方法
    # print('-----------------------------------HMM-----------------------------')
    # hmm = HMM(wordLength, len(tagDict))

    # hmm.trainSup(trainWordLists, trainTagLists)
    # hmm.test(testWordLists, testTagLists)
    # print ('\n')

    #CRF方法
    print('-----------------------------------CRF-----------------------------')
    crf = CRFModel()
    crf.train(trainWordLists, trainTagLists)
    crf.test(testWordLists, testTagLists)
    print ('\n')

if __name__=='__main__':
    f = open('./config.yml', encoding='utf-8', errors='ignore')
    config = yaml.load(f)
    main(config)