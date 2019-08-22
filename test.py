from pytorch_pretrained_bert import BertModel, BertConfig

wordEmbeds = BertModel(config=BertConfig.from_json_file(args.bert_config_json))

wordEmbeds.load_state_dict(torch.load('./ckpts/bert_weight.bin'))

embeds, _ = wordEmbeds(sentence, attention_mask=attention_mask, output_all_encoded_layers=False)
