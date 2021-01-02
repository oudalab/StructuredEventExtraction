class Config(object):
	apr_dir = '../model/'
	data_dir = '../data/'
	model_name = 'model_1.pt'
	epoch = 5
	bert_model = 'bert-base-cased'
	lr = 5e-5
	eps = 1e-8
	batch_size = 16
	mode = 'prediction' # for prediction mode = "prediction"
	training_data = 'train_mavendata.txt'
	val_data = 'dev_ref_mavendata.txt'
	test_data = 'test_mavendata.txt'
	test_out = 'test_prediction.csv'
	raw_prediction_output = 'bert_prediction.csv'