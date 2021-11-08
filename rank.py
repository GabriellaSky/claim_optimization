import pandas as pd
import numpy as np
import tensorflow as tf
import os
import transformers
from nltk.translate.bleu_score import sentence_bleu
from transformers import BertTokenizer,TFBertForSequenceClassification
import svmrank
from sklearn import preprocessing 
from scipy.stats import *
from sklearn.metrics import ndcg_score
import os
from sklearn.model_selection import train_test_split
import warnings
from sentence_transformers import SentenceTransformer, LoggingHandler, util, evaluation, models, InputExample
import torch
import argparse


def generateTfDataset(features):
        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="no_context, with_previous or with_topic depending on training setting", required=True, type=str)
    parser.add_argument("--datadir", help="folder contaning training data", required=True, type=str)
    parser.add_argument("--candidates", type=str, help="file containing generated candidates", required=True)
    parser.add_argument("--svmrank", type=str, help="svm reranking model file", required=True)
    parser.add_argument("--fluency", type=str, help="folder containing BERT models fine-tined for fluency estimation", required=True)
    parser.add_argument("--argquality", type=str, help="folder containing BERT models fine-tined for argument quality estimation", required=True)
    parser.add_argument("--sbertmodel", type=str, help="folder containing SBERT model for claim embeddings", required=True)
    parser.add_argument("--outfile", type=str, help="file containing generated candidates", required=True)

	args = parser.parse_args()


	strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
	MAX_SEQ_LEN=512
	BATCH=16

	datadir = args.datadir
	candidates = args.candidates
	mode = args.mode
	argdir = args.argquality
	fluency = args.fluency
	svm_model = args.svmrank
	sbert_model = args.sbertmodel
	outfile = args.outfile

	#read all inputs
	source = pd.read_csv(datadir+'/test.source', sep = '\t', header = None, names = ['source'])
	target = pd.read_csv(datadir+ '/test.target', sep = '\t', header = None, names = ['target'])
	hypo = pd.read_csv(candidates, sep = '\t', header = None, names = ['pred_1','pred_2',
	                                                                              'pred_3','pred_4',
	                                                                              'pred_5','pred_6',
	                                                                              'pred_7','pred_8',
	                                                                              'pred_9','pred_10'],
	                  index_col =False, quoting = 3)
	hypo['preds'] =  hypo.values.tolist()
	hypo = hypo.drop(['pred_1','pred_2',
	                  'pred_3','pred_4',
	                  'pred_5','pred_6',
	                  'pred_7','pred_8',
	                  'pred_9','pred_10'], axis = 1)
	hypo.preds = hypo.preds.apply(lambda x: list(x))

	#remove all control words where necessary
	if mode in ['with_previous','with_topic']:
	     source.source = source.source.apply(lambda x: x.split(' <V> ')[1])        
	        
	df = pd.concat([source, target, hypo], axis = 1)
	df = df.explode('preds').reset_index()
	df.columns = ['group_id','source','target','preds']

	df['bleu'] = df.apply(lambda x:  sentence_bleu( [x.target.split()],x.preds.split()) if x.preds==x.preds else 0,
	                       axis = 1) 

	examples = [transformers.InputExample(guid=index,
	                           text_a=str(row['source']),
	                           text_b=str(row['preds']),
	                           label = '1') for index, row in df.iterrows()]

	examples_single = [transformers.InputExample(guid=index,
	                           text_a=str(row['preds']),
	                           label = '1') for index, row in df.iterrows()]

	#read bert model for argument quality
	tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
	model = TFBertForSequenceClassification.from_pretrained(argdir)

	dataset = transformers.glue_convert_examples_to_features(examples, tokenizer, max_length=MAX_SEQ_LEN, task='mrpc')

	dataset_single = transformers.glue_convert_examples_to_features(examples_single, tokenizer,
	                                                                max_length=MAX_SEQ_LEN,
	                                                                task='mrpc')

	test_dataset = generateTfDataset(dataset)
	test_dataset = test_dataset.batch(BATCH)

	test_dataset_single = generateTfDataset(dataset_single)
	test_dataset_single = test_dataset_single.batch(BATCH)

	#infer and get softmax probs for class 1
	y_pred = model.predict(test_dataset)
	y_pred = y_pred.logits
	y = tf.nn.softmax(y_pred).numpy()

	df['arg_probs'] = y[:, 1]

	#fluency
	model = TFBertForSequenceClassification.from_pretrained(fluency)

	#infer and get softmax probs for class 1
	y_pred = model.predict(test_dataset_single)
	y_pred = y_pred.logits
	y = tf.nn.softmax(y_pred).numpy()

	df['fluency'] = y[:, 1]


	m = svmrank.Model({'-c': 3})
	m.read(svm_model)

	 #get ranking scores
	model = SentenceTransformer(sbert_model)
	sentence_embeddings = model.encode(df.preds)
	emb = pd.DataFrame(sentence_embeddings).apply(pd.Series)
	emb = emb.add_prefix('feature_')
	test = pd.concat([df,emb],axis = 1)
	filter_test = [col for col in test if col.startswith('feature')]
	le = preprocessing.LabelEncoder() 
	    
	test = test.sort_values(by=['group_id'])
	    
	test_xs = np.array(test[filter_test])
	test_ys = np.array(test.index)

	test_groups = np.array(le.fit_transform(test.group_id))

	test['svm_pred']= m.predict(test_xs, test_groups)

	#get similarity scores
	sbert = []
	model = SentenceTransformer('all-mpnet-base-v2')
	    

	#Sentences are encoded by calling model.encode()
	emb1 = model.encode(list(test.preds))
	emb2 = model.encode(list(test.source))
	test['sbert_score'] = np.diagonal(util.cos_sim(emb1, emb2))


	#get autoscore values
	test['auto_score'] = test.apply(lambda x: (0.43*x.fluency+ 0.01*x.sbert_score + 0.56*x.arg_probs), axis = 1)

	#get selected candidates by each strategy
	top1 = test[test.index % 10 == 0].reset_index(drop =True)
	top1['model'] = 'top1'
	rand = test.groupby('source').apply(lambda x: x.sample(1)).reset_index(drop=True).set_index('source').loc[top1.source].reset_index()
	rand['model'] = 'random'

	svm_idx = test.groupby('source')['svm_pred'].transform(max) == test['svm_pred']
    svmrank = test[svm_idx].drop_duplicates('source', keep = 'first').reset_index(drop=True)
    svmrank['model'] = 'svmrank'

    auto_score_idx = test.groupby('source')['auto_score'].transform(max) == test['auto_score']
    auto_score = test[auto_score_idx].drop_duplicates('source', keep = 'first').reset_index(drop=True)
    auto_score['model'] = 'autoscore'

    result = pd.concat([top1[['model','source','preds','target','bleu','arg_probs','fluency','sbert_score']],
    	rand[['model','source','preds','target','bleu','arg_probs','fluency','sbert_score']],
    	svmrank[['model','source','preds','target','bleu','arg_probs','fluency','sbert_score']],
    	auto_score[['model','source','preds','target','bleu','arg_probs','fluency','sbert_score']]
    	])

	result.to_csv(outfile)



