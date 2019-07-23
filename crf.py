import codecs
import os
from util import acquireEntity
from util import compF1Score
from util import generateSubmit

# #crf train
# crf_train = "crf_learn -f 3 ./datagrand/template.txt ./datagrand/train.bioes ./datagrand/crf_model"
# os.system(crf_train)


# #获得test.change
# change = open('./datagrand/test.change', 'w', encoding = 'utf-8', errors='ignore')
# with codecs.open('./datagrand/test.bioes', 'r', encoding='utf-8') as f:
#     for line in f.readlines():
#         if len(line.strip()) == 0: change.write('\n'); continue
#         word = line.strip().split('\t')[0]
#         change.write(word + '\n')
# change.close()

# #crf test
# crf_test = "crf_test -m ./datagrand/crf_model ./datagrand/test.change -o ./datagrand/test.result"
# os.system(crf_test)

#计算F1值
f1Score = compF1Score(preDataPath='./datagrand/test.result', relDataPath='./datagrand/test.bioes')
print ('crf的F1值为 %f' % f1Score)

#crf test
crf_test = "crf_test -m ./datagrand/crf_model ./datagrand/submit.change -o ./datagrand/submit.result"
os.system(crf_test)


#获得提交数据
generateSubmit(submitPrePath='./datagrand/submit.result', submitResultPath='./datagrand/submit_crf.txt')

