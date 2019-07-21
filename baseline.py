import codecs
import os
from seqeval.metrics import classification_report

# 0 install crf++ https://taku910.github.io/crfpp/
# 1 train data in
# 2 test data in
# 3 crf train
# 4 crf test
# 5 submit test

#step 1 train data in
# with codecs.open('./baseline/train.txt', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     results = []
#     for line in lines:
#         features = []
#         tags = []
#         samples = line.strip().split('  ')
#         for sample in samples:
#             sample_list = sample[:-2].split('_')
#             tag = sample[-1]
#             features.extend(sample_list)
#             tags.extend(['O'] * len(sample_list)) if tag == 'o' else tags.extend(['B-' + tag] + ['I-' + tag] * (len(sample_list)-1))
#         results.append(dict({'features': features, 'tags': tags}))
#     train_write_list = []

#     with codecs.open('./baseline/dg_train.txt', 'w', encoding='utf-8') as f_out:
#         for result in results:
#             for i in range(len(result['tags'])):
#                 train_write_list.append(result['features'][i] + '\t' + result['tags'][i] + '\n')
#             train_write_list.append('\n')
#         f_out.writelines(train_write_list)

# # step 2 test data in
# with codecs.open('./baseline/test.txt', 'r', encoding='utf-8') as f:
#     lines = f.readlines()
#     results = []
#     for line in lines:
#         features = []
#         sample_list = line.split('_')
#         features.extend(sample_list)
#         results.append(dict({'features': features}))
#     test_write_list = []
#     with codecs.open('./baseline/dg_test.txt', 'w', encoding='utf-8') as f_out:
#         for result in results:
#             for i in range(len(result['features'])):
#                 test_write_list.append(result['features'][i] + '\n')
#             test_write_list.append('\n')
#         f_out.writelines(test_write_list)

# 3 crf train
crf_train = "crf_learn -f 3 ./baseline/template.txt ./baseline/train.bio ./baseline/crf_model"
os.system(crf_train)

# 4 crf test
crf_test = "crf_test -m ./baseline/crf_model ./baseline/test.change -o ./baseline/test.result"
os.system(crf_test)

tagsRel, tagsPre = [], []
with codecs.open('./baseline/test.result', 'r', encoding='utf-8') as f:
    lines = f.read()
    lineArr = lines.split('\r\n\r\n')
    for line in lineArr:
        if len(line.strip()) == 0: continue
        tag = [element.split('\t')[1] for element in line.split('\r\n')]   
        tagsPre.append(tag)

with codecs.open('./baseline/test.bio', 'r', encoding='utf-8') as f:
    lines = f.read()
    lineArr = lines.split('\r\n\r\n')
    for line in lineArr:
        if len(line.strip()) == 0: continue
        tag = [element.split('\t')[1] for element in line.split('\r\n')]   
        tagsRel.append(tag)

print (classification_report(tagsRel, tagsPre, digits=6))



#5 submit data
# f_write = codecs.open('./baseline/dg_submit.txt', 'w', encoding='utf-8') 
# with codecs.open('./baseline/dg_result.txt', 'r', encoding='utf-8') as f:
#     lines = f.read().split('\r\n\r\n')
#     for line in lines:
#         if line == '':
#             continue
#         tokens = line.split('\r\n')
#         features = []
#         tags = []
#         for token in tokens:
#             feature_tag = token.split()
#             features.append(feature_tag[0])
#             tags.append(feature_tag[-1])
#         samples = []
#         i = 0
#         while i < len(features):
#             sample = []
#             if tags[i] == 'O':
#                 sample.append(features[i])
#                 j = i + 1
#                 while j < len(features) and tags[j] == 'O':
#                     sample.append(features[j])
#                     j += 1
#                 samples.append('_'.join(sample) + '/o')
#             else:
#                 if tags[i][0] != 'B':
#                     print(tags[i][0] + ' error start')
#                     j = i + 1
#                 else:
#                     sample.append(features[i])
#                     j = i + 1
#                     while j < len(features) and tags[j][0] == 'I' and tags[j][-1] == tags[i][-1]:
#                         sample.append(features[j])
#                         j += 1
#                     samples.append('_'.join(sample) + '/' + tags[i][-1])
#             i = j
#         f_write.write('  '.join(samples) + '\n')
# f_write.close()
