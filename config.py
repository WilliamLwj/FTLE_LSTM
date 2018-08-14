params = {}

# Encoder Network parameters
params['encoder'] = [
	[True, 2,  16, [3, 2, 1], 'relu', True],
	[True, 16, 32, [3, 2, 1], 'relu', True],
	[True, 32, 64, [3, 2, 1], 'relu', True],
	[True, 64, 64, [3, 2, 1], 'relu', True],
	[True, 64, 64, [3, 2, 1], 'relu', True]
]
# LSTM Network parameters
params['input_dim'] = [2, 200, 200]
params['seq_len'] = 21
params['batch_first'] = True

# Decoder Network parameters
params['decoder'] = [
    [False, 64, 64, [3, 2, 1, 0], 'relu', True],
	[False, 64, 64, [3, 2, 1, 0], 'relu', True],
	[False, 64, 32, [3, 2, 1, 1], 'relu', True],
	[False, 32, 16, [3, 2, 1, 1], 'relu', True],
	[False, 16, 1,  [3, 2, 1, 1], 'none', False]
]

# Training parameters
params['lr'] = 1e-3
params['warm_up'] = 5
params['num_epoch'] = 50
params['num_steps'] = 200
params['batch_size'] = 4
params['device'] = 1