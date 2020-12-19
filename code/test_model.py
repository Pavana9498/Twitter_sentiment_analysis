import numpy as np
import pandas as pd
import glob
import joblib
import os
from sklearn.metrics import f1_score, classification_report, accuracy_score, confusion_matrix


from mnb_model import *

if not os.path.exists('results/'):
	os.makedirs('results/')

test_obama_path = "data/obama_test_csv.csv"
test_romney_path = "data/romney_test_csv.csv"

romney_result_path = "results/romney.txt"
obama_result_path = "results/obama.txt"

def evaluate_sklearn_model(dataset='full'):
	'''
	Utility function to evaluate the performance of SKLearn / XGBoost / Ensemble models

	Args:
		model_dir: path to the model directory
	'''
	data, label_map = data_preparation_test(mode='testing', dataset=dataset)

	if dataset == 'obama':
		basepath = 'mnb_obama/'
	elif dataset == 'romney':
		basepath = 'mnb_romney/'

	path = basepath + '*.pkl'  # path to model directory
	fns = glob.glob(path)  # get all fold names

	# create a buffer to store all of the predictions from each fold for each model
	temp_preds = np.zeros((len(fns), data.shape[0], 3))

	class_name = ""
	for j, fn in enumerate(fns):
		model = joblib.load(fn)  # deserialize the model
		class_name = model.__class__.__name__  # get the model name

		if not 'xgb' in fn:  # if SKLearn model
			preds = model.predict_proba(data)

		temp_preds[j, :, :] = preds  # for the jth fold, preserve the predictions

	preds = temp_preds.mean(axis=0)  # calculate the average prediction over all folds
	preds = np.argmax(preds, axis=1)  # compute the maximum probability class

	print("Finished predicting classes\n")

	inverse_label_map = {v: k for k, v in label_map.items()}
	if dataset == 'romney':
		with open(romney_result_path, 'w') as f:
			f.write("676352041\n")
			for i in range(len(preds)):
				f.write("%d;;%d\n" %(i + 1, inverse_label_map[preds[i]]))
	elif dataset == 'obama':
		with open(obama_result_path, 'w') as f:
			f.write("676352041\n")
			for i in range(len(preds)):
				f.write("%d;;%d\n" %(i + 1, inverse_label_map[preds[i]]))
	print("Finished writing output files.\n")
	print('-' *80)


def data_preparation_test(mode = 'testing', dataset='full', verbose = True):
	assert dataset in ['full', 'obama', 'romney']

	if verbose: print('Loading %s data' % mode)

	if dataset == 'full':
		texts, label_map = load_both_datasets_test(mode)
	elif dataset == 'obama':
		texts, label_map = load_obama_dataset_test(mode)
	else:
		texts, label_map = load_romney_dataset_test(mode)

	if verbose: print('Tokenizing texts')
	x_counts = tokenize(texts)  # tokenize the loaded texts
	if verbose: print('Finished tokenizing texts')
	data = tfidf_computation(x_counts)  # transform the loaded texts

	if verbose:
		print('Finished computing TF-IDF')
		print('-' * 80)

	return data, label_map


def load_both_datasets_test(mode = 'testing'):
	if mode == 'testing':
		obama_df = pd.read_csv(test_obama_path, sep='\t', encoding='latin1')
		romney_df = pd.read_csv(test_romney_path, sep='\t', encoding = 'latin1')

	texts = []
	labels_index = {-1: 0, 0: 1, 1: 2}

	nb_rows = len(obama_df)
	for i in range(nb_rows):
		row = obama_df.iloc[i]
		texts.append(str(row['tweet']))

	nb_rows = len(romney_df)
	for i in range(nb_rows):
		row = romney_df.iloc[i]
		texts.append(str(row['tweet']))

	texts = np.asarray(texts)

	return texts, labels_index

def load_romney_dataset_test(mode='testing'):
	'''
	Loads the Romney dataset

	Args:
		mode: decides whether to load train or test set

	Returns:
		raw text list, labels and label indices
	'''

	if mode == 'testing':
		romney_df = pd.read_csv(test_romney_path, sep='\t', encoding='latin1')

	texts = []  # list of text samples
	labels_index = {-1: 0, 0: 1, 1: 2}  # dictionary mapping label name to numeric id

	nb_rows = len(romney_df)
	for i in range(nb_rows):
		row = romney_df.iloc[i]
		texts.append(str(row['tweet']))

	texts = np.asarray(texts)

	return texts, labels_index

def load_obama_dataset_test(mode='testing'):
	'''
	Loads the Obama dataset

	Args:
		mode: decides whether to load train or test set

	Returns:
		raw text list, labels and label indices
	'''
	if mode == 'testing':
		obama_df = pd.read_csv(test_obama_path, sep='\t', encoding='latin1')


	texts = []  # list of text samples
	labels_index = {-1: 0, 0: 1, 1: 2}  # dictionary mapping label name to numeric id

	nb_rows = len(obama_df)
	for i in range(nb_rows):
		row = obama_df.iloc[i]
		texts.append(str(row['tweet']))

	texts = np.asarray(texts)

	return texts, labels_index


if __name__ == '__main__':
	evaluate_sklearn_model(dataset = 'obama')




