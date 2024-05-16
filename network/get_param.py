import argparse

def str2bool(v):
	"""
	'boolean type variable' for add_argument
	"""
	if v.lower() in ('yes','true','t','y','1'):
		return True
	elif v.lower() in ('no','false','f','n','0'):
		return False
	else:
		raise argparse.ArgumentTypeError('boolean value expected.')

def get_params():
	"""
	return parameters for training / testing / plotting of models
	:return: parameter-Namespace
	"""
	parser = argparse.ArgumentParser(description='train / test a pytorch model to simulate cloth')

	# Network parameters
	parser.add_argument('--net', default="SMP", type=str, help='network to train (default: SMP)', choices=["SMP","SMP_param","SMP_param_a","SMP_param_a_gated","SMP_param_a_gated2","SMP_param_a_gated3","SMP_param_a_gated5","UNet","UNet_param_a"])
	parser.add_argument('--SMP_model_type', default="Unet", type=str, help='model type used for SMP segmentation nets')
	parser.add_argument('--SMP_encoder_name', default="resnet34", type=str, help='encoder name used for SMP segmentation nets')
	parser.add_argument('--hidden_size', default=20, type=int, help='hidden size of network (default: 20)')
	
	# Training parameters
	parser.add_argument('--n_epochs', default=101, type=int, help='number of epochs (after each epoch, the model gets saved)')
	parser.add_argument('--n_batches_per_epoch', default=501, type=int, help='number of batches per epoch (default: 5000)')
	parser.add_argument('--batch_size', default=100, type=int, help='batch size (default: 100)')
	parser.add_argument('--average_sequence_length', default=1000, type=int, help='average sequence length in dataset (default: 1000)')
	parser.add_argument('--dataset_size', default=500, type=int, help='size of dataset (default: 500)')
	parser.add_argument('--cuda', default=True, type=str2bool, help='use GPU')
	parser.add_argument('--ema_beta', default=0.995, type=float, help='ema beta (default: 0.995)')
	parser.add_argument('--ema_update_after_step', default=None, type=int, help='only after this number of .update() calls will it start updating EMA (default: 100)')
	parser.add_argument('--ema_update_every', default=1, type=int, help='how often to actually update EMA, to save on compute (default: 1)')
	parser.add_argument('--lr', default=0.001, type=float, help='learning rate of optimizer (default: 0.001)')
	parser.add_argument('--clip_grad_norm', default=None, type=float, help='gradient norm clipping (default: None)')
	parser.add_argument('--clip_grad_value', default=None, type=float, help='gradient value clipping (default: None)')
	parser.add_argument('--plot', default=False, type=str2bool, help='plot during training')
	parser.add_argument('--log', default=True, type=str2bool, help='log models / metrics during training (turn off for debugging)')

	# Setup parameters
	parser.add_argument('--height', default=32, type=int, help='cloth height')
	parser.add_argument('--width', default=32, type=int, help='cloth width')
	
	# Cloth parameters
	parser.add_argument('--max_stretching', default=5000, type=float, help='max stretching parameter of cloth')
	parser.add_argument('--min_stretching', default=100, type=float, help='min stretching parameter of cloth')
	parser.add_argument('--max_shearing', default=20, type=float, help='max shearing parameter of cloth')
	parser.add_argument('--min_shearing', default=0.05, type=float, help='min shearing parameter of cloth')
	parser.add_argument('--max_bending', default=1, type=float, help='max bending parameter of cloth')
	parser.add_argument('--min_bending', default=0.001, type=float, help='min bending parameter of cloth')
	parser.add_argument('--a_ext', default=1, type=float, help='gravitational constant (external acceleration)')
	parser.add_argument('--min_a_ext', default=None, type=float, help='min gravitational constant (default: same as a_ext)')
	parser.add_argument('--g', default=1, type=float, help='gravitational constant (deprecated: use a_ext instead!)')
	parser.add_argument('--L_0', default=1, type=float, help='rest length of cloth grid edges')
	parser.add_argument('--dt', default=1, type=float, help='timestep of cloth simulation integrator')
	
	# Load parameters
	parser.add_argument('--l_stretching', default=None, type=float, help='load stretching parameter of cloth')
	parser.add_argument('--l_shearing', default=None, type=float, help='load shearing parameter of cloth')
	parser.add_argument('--l_bending', default=None, type=float, help='load bending parameter of cloth')
	parser.add_argument('--l_g', default=None, type=float, help='load gravitational constant')
	parser.add_argument('--l_L_0', default=None, type=float, help='load rest length of cloth grid edges')
	parser.add_argument('--l_dt', default=1, type=float, help='load timestep of cloth simulation integrator')
	parser.add_argument('--load_date_time', default=None, type=str, help='date_time of run to load (default: None)')
	parser.add_argument('--load_index', default=None, type=str, help='index of run to load (default: None)')
	#parser.add_argument('--load_index', default=None, type=int, help='index of run to load (default: None)')
	parser.add_argument('--load_optimizer', default=False, type=str2bool, help='load state of optimizer (default: True)')
	parser.add_argument('--load_latest', default=False, type=str2bool, help='load latest version for training (if True: leave load_date_time and load_index None. default: False)')
	
	# parse parameters
	params = parser.parse_args()
	
	params.min_stretching = params.max_stretching if params.min_stretching is None else params.min_stretching
	params.min_shearing = params.max_shearing if params.min_shearing is None else params.min_shearing
	params.min_bending = params.max_bending if params.min_bending is None else params.min_bending
	params.min_a_ext = params.a_ext if params.min_a_ext is None else params.min_a_ext
	params.stretching_range = [params.min_stretching,params.max_stretching]
	params.shearing_range = [params.min_shearing,params.max_shearing]
	params.bending_range = [params.min_bending,params.max_bending]
	params.a_ext_range = [params.min_a_ext,params.a_ext]
	
	params.l_stretching = params.max_stretching if params.l_stretching is None else params.l_stretching
	params.l_shearing = params.max_shearing if params.l_shearing is None else params.l_shearing
	params.l_bending = params.max_bending if params.l_bending is None else params.l_bending
	params.l_g = params.g if params.l_g is None else params.l_g
	params.l_L_0 = params.L_0 if params.l_L_0 is None else params.l_L_0
	params.l_dt = params.dt if params.l_dt is None else params.l_dt
	
	params.ema_update_after_step = params.n_batches_per_epoch*10 if params.ema_update_after_step is None else params.ema_update_after_step
	
	return params

params = get_params()

def get_hyperparam(params):
	if params.net=="SMP_param" or params.net=="SMP_param_a" or params.net=="SMP_param_a_gated" or params.net=="SMP_param_a_gated2" or params.net=="SMP_param_a_gated3":
		return f"net {params.net}; type {params.SMP_model_type}; enc {params.SMP_encoder_name}; dt {params.dt};"
	if params.net=="SMP":
		return f"net {params.net}; type {params.SMP_model_type}; enc {params.SMP_encoder_name}; stiff {params.stretching}; shear {params.shearing}; bend {params.bending}; dt {params.dt};"
	if params.net=="UNet_param_a":
		return f"net {params.net}; hs {params.hidden_size}; dt {params.dt};"
	return f"net {params.net}; hs {params.hidden_size}; stiff {params.stretching}; shear {params.shearing}; bend {params.bending}; dt {params.dt};"

def get_load_hyperparam(params):
	if params.net=="SMP_param" or params.net=="SMP_param_a" or params.net=="SMP_param_a_gated" or params.net=="SMP_param_a_gated2" or params.net=="SMP_param_a_gated3":
		return f"net {params.net}; type {params.SMP_model_type}; enc {params.SMP_encoder_name}; dt {params.l_dt};"
	if params.net=="SMP":
		return f"net {params.net}; type {params.SMP_model_type}; enc {params.SMP_encoder_name}; stiff {params.l_stretching}; shear {params.l_shearing}; bend {params.l_bending}; dt {params.l_dt};"
	if params.net=="UNet_param_a":
		return f"net {params.net}; hs {params.hidden_size}; dt {params.l_dt};"
	return f"net {params.net}; hs {params.hidden_size}; stiff {params.l_stretching}; shear {params.l_shearing}; bend {params.l_bending}; dt {params.l_dt};"

def toCuda(x):
	if type(x) is tuple or type(x) is list:
		return [xi.cuda() if params.cuda else xi for xi in x]
	return x.cuda() if params.cuda else x

def toCpu(x):
	if type(x) is tuple or type(x) is list:
		return [xi.detach().cpu() for xi in x]
	return x.detach().cpu()
