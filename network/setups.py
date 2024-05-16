import torch
import numpy as np

"""
ask-tell interface:
ask(): ask for batch of x(t),v(t), constraints(x,v)
tell(x,v): tell results for x(t+1),v(t+1) of batch
"""
#Attention: x/y are swapped (x-dimension=1; y-dimension=0)

# CODO: spatially varying stretching / shearing / bending parameters

def rotation_matrix(dyaw=0.0,dpitch=0.0,droll=0.0,device=None):
	"""
	:return: matrix to rotate by dpitch/dyaw/droll
	"""
	def tensor(x):
		if type(x) is not torch.Tensor:
			return torch.tensor(x,device=device)
		return x
	dpitch,dyaw,droll = tensor(dpitch),tensor(dyaw),tensor(droll)
	trafo_pitch_matrix = torch.eye(3,device=device)
	if dpitch != 0 or dpitch.requires_grad:
		trafo_pitch_matrix[1,1] = torch.cos(dpitch)
		trafo_pitch_matrix[1,2] = -torch.sin(dpitch)
		trafo_pitch_matrix[2,1] = torch.sin(dpitch)
		trafo_pitch_matrix[2,2] = torch.cos(dpitch)
	trafo_yaw_matrix = torch.eye(3,device=device)
	if dyaw != 0 or dyaw.requires_grad:
		trafo_yaw_matrix[0,0] = torch.cos(dyaw)
		trafo_yaw_matrix[0,2] = torch.sin(dyaw)
		trafo_yaw_matrix[2,0] = -torch.sin(dyaw)
		trafo_yaw_matrix[2,2] = torch.cos(dyaw)
	trafo_roll_matrix = torch.eye(3,device=device)
	if droll != 0 or droll.requires_grad:
		trafo_roll_matrix[0,0] = torch.cos(droll)
		trafo_roll_matrix[0,1] = -torch.sin(droll)
		trafo_roll_matrix[1,0] = torch.sin(droll)
		trafo_roll_matrix[1,1] = torch.cos(droll)
	trafo_matrix = torch.matmul(torch.matmul(trafo_yaw_matrix,trafo_pitch_matrix),trafo_roll_matrix)
	return trafo_matrix


class Dataset:
	def __init__(self,h,w,batch_size=100,dataset_size=1000,average_sequence_length=5000,interactive=False,dt=1,L_0=1,stretching_range=None,shearing_range=None,bending_range=None,a_ext_range=None):
		self.h,self.w = h,w
		self.batch_size = batch_size
		self.dataset_size = dataset_size
		self.average_sequence_length = average_sequence_length
		self.interactive = interactive
		self.dt = dt
		self.step = 0
		self.L_0 = L_0
		self.reset_i = 0
		def log_range_params(range_params,default_param=1):# useful to sample parameters from "exponential distribution"
			range_params = default_param if range_params is None else range_params
			range_params = [range_params,range_params] if type(range_params) is not list else range_params
			range_params = np.log(range_params)
			return range_params[0],range_params[1]-range_params[0]
		def range_params(r_params,default_param=1):# useful to sample parameters from "exponential distribution"
			r_params = default_param if r_params is None else r_params
			r_params = [r_params,r_params] if type(r_params) is not list else r_params
			return r_params[0],r_params[1]-r_params[0]
		
		self.stretching_range = log_range_params(stretching_range)
		self.shearing_range = log_range_params(shearing_range)
		self.bending_range = log_range_params(bending_range)
		self.g_vect = torch.tensor([0,0,-1.]).unsqueeze(0).repeat(self.dataset_size,1).unsqueeze(2).unsqueeze(3) # gravity vector. CODO: radnom directions / strengths of gravity
		self.a_ext_range = range_params(a_ext_range)
		self.a_exts = torch.ones(self.dataset_size,3,self.h,self.w)*self.g_vect# external forces
		self.a_exts_damping = 0.999
		self.da_exts_dt = torch.zeros(self.dataset_size,3,self.h,self.w)# derivatives of external forces
		self.da_exts_dt_damping = 0.95
		
		x_space = torch.linspace(0,L_0*(w-1),w)
		y_space = torch.linspace(-L_0*(h-1)/2,L_0*(h-1)/2,h)
		y_grid,x_grid = torch.meshgrid(y_space,x_space,indexing="ij")
		self.y_mesh,self.x_mesh = torch.meshgrid([torch.arange(0,self.h),torch.arange(0,self.w)])
		
		x_0 = torch.cat([x_grid.unsqueeze(0),y_grid.unsqueeze(0),torch.zeros(1,h,w)],dim=0)
		v_0 = torch.zeros(3,h,w)
		self.x_v_0 = torch.cat([x_0,v_0],dim=0)
		self.x_v = torch.zeros(self.dataset_size,6,self.h,self.w)
		self.T = torch.zeros(self.dataset_size,1) # timestep
		self.rot_speed = torch.zeros(self.dataset_size,3,3) # delta rotation matrix that is recurrently multiplied onto rotations
		self.translation_freq = torch.zeros(self.dataset_size,3) # delta rotation matrix that is recurrently multiplied onto rotations
		self.pinch_freq = torch.zeros(self.dataset_size,1) # delta rotation matrix that is recurrently multiplied onto rotations
		self.translation_amp = torch.zeros(self.dataset_size,3) # delta rotation matrix that is recurrently multiplied onto rotations
		self.rotations = torch.zeros(self.dataset_size,3,3)
		self.conditions = torch.zeros(self.dataset_size,2,3) # left / right point & (x,y,z) coordinates
		self.stretchings = torch.zeros(self.dataset_size)
		self.shearings = torch.zeros(self.dataset_size)
		self.bendings = torch.zeros(self.dataset_size)
		#self.a_exts = torch.ones(self.dataset_size,3,self.h,self.w)*torch.tensor([0,0,-1]).unsqueeze(0).unsqueeze(2).unsqueeze(3) # init with gravity. CODO: radnom directions / strengths of gravity
		
		for i in range(self.dataset_size):
			self.reset_env(i)
		
		self.M = torch.ones(1,1,h,w)
		self.M[:,:,0] = self.M[:,:,-1] = self.M[:,:,:,0] = self.M[:,:,:,-1] = 0.5
		self.M[:,:,0,0] = self.M[:,:,0,-1] = self.M[:,:,-1,0] = self.M[:,:,-1,-1] = 0.25
		
	def reset_env(self,index):
		self.stretchings[index] = torch.exp(self.stretching_range[0]+torch.rand(1)*self.stretching_range[1])
		self.shearings[index] = torch.exp(self.shearing_range[0]+torch.rand(1)*self.shearing_range[1])
		self.bendings[index] = torch.exp(self.bending_range[0]+torch.rand(1)*self.bending_range[1])
		#self.a_exts[index] = torch.exp(self.a_ext_range[0]+torch.rand(1)*self.a_ext_range[1]) # TODO: init with gravity
		g_scale = self.a_ext_range[0]+torch.rand(1)*self.a_ext_range[1]
		
		self.x_v[index] = self.x_v_0.clone()
		self.conditions[index,0] = self.x_v[index,:3,0,0]
		self.conditions[index,1] = self.x_v[index,:3,-1,0]
		self.T[index] = 0
		yaw = (torch.rand(1)-0.5)*2*2*3.14#0#
		pitch = (torch.rand(1)-0.5)*2*2*3.14#0#
		roll = (torch.rand(1)-0.5)*2*2*3.14#0#
		dyaw = (torch.rand(1)-0.5)*2*2*3.14*0.01
		dpitch = (torch.rand(1)-0.5)*2*2*3.14*0.01
		droll = (torch.rand(1)-0.5)*2*2*3.14*0.01 # keep only roll for rotation
		self.rot_speed[index] = rotation_matrix(dyaw,dpitch,droll)
		self.rotations[index] = rotation_matrix(yaw,pitch,roll)
		self.translation_freq[index] = (torch.rand(3)-0.5)*2*0.2
		self.translation_amp[index] = (torch.rand(3)-0.5)*2*10
		self.pinch_freq[index] = (torch.rand(1)-0.5)*2*0.2
		self.conditions[index] = torch.einsum("ab,cb->ca",self.rotations[index],self.conditions[index])
		self.x_v[index,:3] = torch.einsum("ab,bcd->acd",self.rotations[index],self.x_v[index,:3])
		#print(f"reset {index}")
		
		self.g_vect[index,:,0,0] = torch.einsum("ab,b->a",
												rotation_matrix((torch.rand(1)-0.5)*2*2*3.14,
																(torch.rand(1)-0.5)*2*2*3.14,
																(torch.rand(1)-0.5)*2*2*3.14),
												torch.tensor([0,0,-1.0])*g_scale)
		self.a_exts[index,:,:,:] = self.g_vect[index]
		#print(f"rot mat: {rotation_matrix((torch.rand(1)-0.5)*2*2*3.14,(torch.rand(1)-0.5)*2*2*3.14,(torch.rand(1)-0.5)*2*2*3.14)}")
		#print(f"g_vect: {self.g_vect[index,:,0,0]}")
		self.da_exts_dt[index,:,:,:] = 0
	
	def ask(self):
		"""
		:return:
			:x_v: tensor containing (x/y/z) positions and velocities of cloth of shape (batch_size x 6 x h x w)
			:M: mass tensor of shape (1 x 1 x h x w)
			:BoundaryConditions: function that maps x_v back onto x_v with enforced boundaries (this could probably be implemented more elegantly)
		"""
		self.indices = np.random.choice(self.dataset_size,self.batch_size)
		self.T[self.indices] = self.T[self.indices]+1
		self.conditions[self.indices] = torch.einsum("dab,dcb->dca",self.rot_speed[self.indices],self.conditions[self.indices])
		
		
		# update external forces (CODO: clip min/max forces) ...not very efficient (slows down test_cv2_interactive by approx 10%)
		self.a_exts[self.indices,:,:,:] = self.a_exts_damping*self.a_exts[self.indices,:,:,:]+(1-self.a_exts_damping)*self.g_vect[self.indices]+0.01*self.da_exts_dt[self.indices,:,:,:] # TODO: add g_vect
		if torch.rand(1)<0.2:
			gaussian_w = torch.rand(1)*5+0.1
			gaussian = torch.exp(-((self.x_mesh-torch.rand(1,1)*self.w)**2+(self.y_mesh-torch.rand(1,1)*self.h)**2)/gaussian_w).unsqueeze(0).unsqueeze(1)
			gaussian = gaussian*torch.randn(1,3,1,1)
		else:
			gaussian = 0
		self.da_exts_dt[self.indices,:,:,:] = (self.da_exts_dt_damping*self.da_exts_dt[self.indices,:,:,:]
										      +0.1*torch.randn(1,3,1,1)
										      +0.05*torch.randn(1,3,self.h,self.w)
											  +gaussian)
		
		
		def BoundaryConditions(x,v):
			f = self.rot_speed[self.indices]
			
			x[:,:3,0,0] = self.conditions[self.indices,0,:]*(torch.cos(self.T[self.indices]*self.pinch_freq[self.indices])*0.4+0.6) + torch.sin(self.T[self.indices]*self.translation_freq[self.indices])*self.translation_amp[self.indices]
			x[:,:3,-1,0] = self.conditions[self.indices,1,:]*(torch.cos(self.T[self.indices]*self.pinch_freq[self.indices])*0.4+0.6) + torch.sin(self.T[self.indices]*self.translation_freq[self.indices])*self.translation_amp[self.indices]
			v[:,:,0,0] = v[:,:,-1,0] = 0
			return x,v # CODO: be more careful with BC...
		
		# TODO: add random noise to a_exts
		
		return self.x_v[self.indices], self.stretchings[self.indices], self.shearings[self.indices], self.bendings[self.indices], self.a_exts[self.indices], self.M, BoundaryConditions
	
	def tell(self,x_v_new):
		self.x_v[self.indices,:,:,:] = x_v_new.detach()
		
		self.step += 1
		if self.step % (self.average_sequence_length/self.batch_size) == 0:#ca x*batch_size steps until env gets reset
			self.reset_env(int(self.reset_i))
			self.reset_i = (self.reset_i+1)%self.dataset_size
