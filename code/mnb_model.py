import numpy as np
import joblib
import glob
import pandas as pd
import os
import pickle
import time

np.random.seed(1000)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import StratifiedKFold


if not os.path.exists('mnb_romney/'):
	os.makedirs('mnb_romney/')

if not os.path.exists('mnb_obama/'):
	os.makedirs('mnb_obama/')


train_obama_path = "data/obama_csv.csv"
train_romney_path = "data/romney_csv.csv"


def train_model(dataset = 'full', k_folds = 100, seed = 1000):
	print(dataset)
	data, labels = data_peparation(mode = "training", dataset = dataset)
	print(data.shape[0])
	skfolds = StratifiedKFold(k_folds, shuffle=True, random_state=seed)

	for i, (train_index, test_index) in enumerate(skfolds.split(data, labels)):
		x_train, y_train = data[train_index, :], labels[train_index]  # obtain train fold
		x_test, y_test = data[test_index, :], labels[test_index]  # obtain test fold

		print('\nTraining classifier %d' % (i + 1), 'Training samples : ', x_train.shape[0], 'Testing samples : ', x_test.shape[0])

		model  = MultinomialNB(alpha = 0.149)

		t1 = time.time()

		try:
			model.fit(x_train, y_train)  # attempt to fit the dataset using sparse numpy arrays
		except TypeError:
			# Model does not support sparce matrix input, convert to dense matrix input
			x_train = x_train.toarray()
			model.fit(x_train, y_train)

			# dense matrix input is very large, delete to preserve memory
			del (x_train)

		t2 = time.time()

		print('Classifier %d training time : %0.3f seconds.' % (i + 1, t2 - t1))

		print('Testing classifier %d' % (i + 1))

		t1 = time.time()
		try:
			preds = model.predict(x_test)  # attempt to obtain predictions using sparce numpy arrays
		except TypeError:
			# Model does not support sparce matrix input, convert to dense matrix input
			x_test = x_test.toarray()
			preds = model.predict(x_test)

			# dense matrix input is very large, delete to preserve memory
			del (x_test)

		t2 = time.time()

		print('Classifier %d finished predicting in %0.3f seconds.' % (i + 1, t2 - t1))

		if dataset == 'full':
			joblib.dump(model, 'mnb_romney/mnb_model-cv-%d.pkl' % (i + 1))  # serialize the trained model
		elif dataset == 'obama':
			joblib.dump(model, 'mnb_obama/mnb_model-cv-%d.pkl' % (i + 1))  # serialize the trained model

		del model  # delete the trained model from CPU memory


def data_peparation(mode='training', dataset='full', verbose=True):
	'''
	Utility function to load a given dataset with a certain mode

	Args:
		mode: can be 'train' or 'test'
		dataset: can be 'full', 'obama' or 'romney'
		verbose: set to True to obtain information of loaded dataset

	Returns:
		tokenized tf-idf transformed input text and corresponding labels
	'''
	assert dataset in ['full', 'obama', 'romney']

	if verbose: print('Loading %s data' % mode)

	print("Dataset: ",dataset)
	if dataset == 'full':
		texts, labels, label_map = load_both_datasets(mode)
	elif dataset == 'obama':
		texts, labels, label_map = load_obama_dataset(mode)
	else:
		texts, labels, label_map = load_romney_dataset(mode)

	if verbose: print('Tokenizing texts')
	x_counts = tokenize(texts)  # tokenize the loaded texts

	if verbose: print('Finished tokenizing texts')
	data = tfidf_computation(x_counts)  # transform the loaded texts

	if verbose:
		print('Finished computing TF-IDF')
		print('-' * 80)

	return data, labels


def load_both_datasets(mode='training'):
	'''
	Loads both Obama and Romney datasets for Joint Training

	Args:
		mode: decides whether to load train or test set

	Returns:
		raw text list, labels and label indices
	'''
	if mode == 'training':
		obama_df = pd.read_csv(train_obama_path, sep='\t', encoding='latin1')

	# Remove rows who have no class label attached, can hand label later
	obama_df = obama_df[pd.notnull(obama_df['label'])]
	obama_df = obama_df[obama_df['label']!='irrelevant']
	obama_df = obama_df[obama_df['label']!='irrevelant']
	obama_df['label'] = obama_df['label'].astype(np.int)

	if mode == 'training':
		romney_df = pd.read_csv(train_romney_path, sep='\t', encoding='latin1')

	# Remove rows who have no class label attached, can hand label later
	romney_df = romney_df[pd.notnull(romney_df['label'])]
	romney_df = romney_df[romney_df['label']!='!!!!']
	romney_df = romney_df[romney_df['label']!='IR']
	romney_df['label'] = romney_df['label'].astype(np.int)

	texts = []  # list of text samples
	labels_index = {-1: 0, 0: 1, 1: 2}  # dictionary mapping label name to numeric id
	labels = []  # list of label ids

	obama_df = obama_df[obama_df['label'] != 2]  # drop all rows with class = 2
	romney_df = romney_df[romney_df['label'] != 2]  # drop all rows with class = 2

	nb_rows = len(obama_df)
	for i in range(nb_rows):
		row = obama_df.iloc[i]
		texts.append(str(row['tweet']))
		labels.append(labels_index[int(row['label'])])

	nb_rows = len(romney_df)
	for i in range(nb_rows):
		row = romney_df.iloc[i]
		texts.append(str(row['tweet']))
		labels.append(labels_index[int(row['label'])])

	texts = np.asarray(texts)
	labels = np.asarray(labels)

	return texts, labels, labels_index


def tokenize(texts):
	'''
	For SKLearn models, use CountVectorizer to generate n-gram vectorized texts efficiently.

	Args:
		texts: input text sentences list

	Returns:
		the n-gram text
	'''
	if os.path.exists('data/vectorizer.pkl'):
		with open('data/vectorizer.pkl', 'rb') as f:
			vectorizer = pickle.load(f)
			x_counts = vectorizer.transform(texts)
	else:
		vectorizer = CountVectorizer(ngram_range=(1, 2))
		x_counts = vectorizer.fit_transform(texts)

		with open('data/vectorizer.pkl', 'wb') as f:
			pickle.dump(vectorizer, f)

	print('Shape of tokenizer counts : ', x_counts.shape)
	return x_counts



def tfidf_computation(x_counts):
	'''
	Perform TF-IDF transform to normalize the dataset

	Args:
		x_counts: the n-gram tokenized sentences

	Returns:
		the TF-IDF transformed dataset
	'''
	if os.path.exists('data/tfidf.pkl'):
		with open('data/tfidf.pkl', 'rb') as f:
			transformer = pickle.load(f)
			x_tfidf = transformer.transform(x_counts)
	else:
		transformer = TfidfTransformer()
		x_tfidf = transformer.fit_transform(x_counts)

		with open('data/tfidf.pkl', 'wb') as f:
			pickle.dump(transformer, f)

	return x_tfidf


def load_romney_dataset(mode='training'):
	'''
	Loads the Romney dataset

	Args:
		mode: decides whether to load train or test set

	Returns:
		raw text list, labels and label indices
	'''

	if mode == 'training':
		romney_df = pd.read_csv(train_romney_path, sep='\t', encoding='latin1')
	
	# Remove rows who have no class label attached, can hand label later
	romney_df = romney_df[pd.notnull(romney_df['label'])]
	romney_df = romney_df[romney_df['label']!='!!!!']
	romney_df = romney_df[romney_df['label']!='IR']
	romney_df['label'] = romney_df['label'].astype(np.int)

	texts = []  # list of text samples
	labels_index = {-1: 0, 0: 1, 1: 2}  # dictionary mapping label name to numeric id
	labels = []  # list of label ids

	romney_df = romney_df[romney_df['label'] != 2]  # drop all rows with class = 2

	nb_rows = len(romney_df)
	for i in range(nb_rows):
		row = romney_df.iloc[i]
		texts.append(str(row['tweet']))
		labels.append(labels_index[int(row['label'])])

	texts = np.asarray(texts)
	labels = np.asarray(labels)

	return texts, labels, labels_index

def load_obama_dataset(mode='training'):
	'''
	Loads the Obama dataset

	Args:
		mode: decides whether to load train or test set

	Returns:
		raw text list, labels and label indices
	'''
	if mode == 'training':
		obama_df = pd.read_csv(train_obama_path, sep='\t', encoding='latin1')
	
	# Remove rows who have no class label attached, can hand label later
	obama_df = obama_df[pd.notnull(obama_df['label'])]
	obama_df = obama_df[obama_df['label']!='irrelevant']
	obama_df = obama_df[obama_df['label']!='irrevelant']
	obama_df['label'] = obama_df['label'].astype(np.int)

	texts = []  # list of text samples
	labels_index = {-1: 0, 0: 1, 1: 2}  # dictionary mapping label name to numeric id
	labels = []  # list of label ids

	obama_df = obama_df[obama_df['label'] != 2]  # drop all rows with class = 2

	nb_rows = len(obama_df)
	for i in range(nb_rows):
		row = obama_df.iloc[i]
		texts.append(str(row['tweet']))
		labels.append(labels_index[int(row['label'])])

	texts = np.asarray(texts)
	labels = np.asarray(labels)

	return texts, labels, labels_index


if __name__ == '__main__':
	train_model(dataset = 'obama', k_folds=100)




