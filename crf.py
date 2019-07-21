import codecs
import os
from tool import acquireEntity
from seqeval.metrics import classification_report
#crf train
crf_train = "crf_learn -f 3 ./datagrand/template.txt ./datagrand/train.bioes ./datagrand/crf_model"
os.system(crf_train)

#crf test
crf_test = "crf_test -m ./datagrand/crf_model ./datagrand/test.change -o ./datagrand/test.result"
os.system(crf_test)

tagsRel, tagsPre = [], []
with codecs.open('./datagrand/test.result', 'r', encoding='utf-8') as f:
    lines = f.read()
    lineArr = lines.split('\r\n\r\n')
    for line in lineArr:
        if len(line.strip()) == 0: continue
        tag = [element.split('\t')[1] for element in line.split('\r\n')]   
        tagsPre.append(tag)

with codecs.open('./datagrand/test.bioes', 'r', encoding='utf-8') as f:
    lines = f.read()
    lineArr = lines.split('\r\n\r\n')
    for line in lineArr:
        if len(line.strip()) == 0: continue
        tag = [element.split('\t')[1] for element in line.split('\r\n')]   
        tagsRel.append(tag)

print (classification_report(tagsRel, tagsPre, digits=6))
    

'''
生成submit代码
'''
# submit = open('./datagrand/submit.result', 'w', encoding='utf-8', errors='ignore')
# with codecs.open('./datagrand/test.result', 'r', encoding='utf-8') as f:
#     lines = f.read()
#     lineArr = lines.split('\r\n\r\n')
#     for line in lineArr:
#         if len(line.strip()) == 0: continue
#         sentence = [element.split('\t')[0] for element in line.split('\r\n')]
#         tag = [element.split('\t')[1] for element in line.split('\r\n')]
#         sentence1, tag1 = acquireEntity(sentence, tag)
#         #print (sentence, tag)
#         if len(sentence1) == 0 and len(tag1) == 0:
#             submit.write('_'.join(sentence) + '/' + 'o' + '\n'); continue
#         arr = [element1 + '/' + element2 for element1, element2 in zip(sentence1, tag1)]
#         submit.write('  '.join(arr) + '\n')
# f.close()
