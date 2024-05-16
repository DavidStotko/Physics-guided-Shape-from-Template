import segmentation_models_pytorch as smp 
import torch
from torch import nn
from get_param import params

def get_Net(params):
	if params.net == "UNet":
		net = Cloth_Unet(params.hidden_size)
	elif params.net == "UNet_param_a":
		net = Cloth_Unet_param_a(params.hidden_size)
	elif params.net == "SMP":
		net = Cloth_net(params.SMP_model_type,params.SMP_encoder_name)
	elif params.net == "SMP_param":
		net = Cloth_net_param(params.SMP_model_type,params.SMP_encoder_name)
	elif params.net == "SMP_param_a":
		net = Cloth_net_param_a(params.SMP_model_type,params.SMP_encoder_name)
	elif params.net == "SMP_param_a_gated":
		net = Cloth_net_param_a_gated(params.SMP_model_type,params.SMP_encoder_name)
	elif params.net == "SMP_param_a_gated2":
		net = Cloth_net_param_a_gated2(params.SMP_model_type,params.SMP_encoder_name)
	elif params.net == "SMP_param_a_gated3":
		net = Cloth_net_param_a_gated3(params.SMP_model_type,params.SMP_encoder_name)
	elif params.net == "SMP_param_a_gated5":
		net = Cloth_net_param_a_gated5(params.SMP_model_type,params.SMP_encoder_name)
	return net

class Cloth_net(nn.Module):
	
	def __init__(self, model_type, encoder_name):
		"""
		:model_type: ... see diverse models below
		:encoder_name: e.g. tu-mobilevitv2_100 or resnet34 ... more examples: https://github.com/qubvel/segmentation_models.pytorch
		"""
		
		super(Cloth_net, self).__init__()
		if model_type=="Unet":
			SMP_model_type = smp.Unet
		elif model_type=="UnetPlusPlus":
			SMP_model_type = smp.UnetPlusPlus
		elif model_type=="MAnet":
			SMP_model_type = smp.MAnet
		elif model_type=="Linknet":
			SMP_model_type = smp.Linknet
		elif model_type=="FPN":
			SMP_model_type = smp.FPN
		elif model_type=="PSPNet":
			SMP_model_type = smp.PSPNet
		elif model_type=="PAN":
			SMP_model_type = smp.PAN
		elif model_type=="DeepLabV3":
			SMP_model_type = smp.DeepLabV3
		elif model_type=="DeepLabV3Plus":
			SMP_model_type = smp.DeepLabV3Plus
		else:
			raise Exception("invalid SMP_model_type!")
		self.model = SMP_model_type(encoder_name=encoder_name,in_channels=12,classes=3)
	
	def forward(self, x_v, stiffnesses=None, shearings=None, bendings=None, a=None):
		bs,c,h,w = x_v.shape
		device = x_v.device
		di = torch.cat([x_v[:,:,1:]-x_v[:,:,:-1],torch.zeros(bs,c,1,w,device=device)],dim=2)
		dj = torch.cat([x_v[:,:,:,1:]-x_v[:,:,:,:-1],torch.zeros(bs,c,h,1,device=device)],dim=3)
		x = torch.cat([di,dj],dim=1)
		x = self.model(x)
		return 10*torch.tanh(x/10)

class Cloth_net_param(nn.Module):
	
	def __init__(self, model_type, encoder_name):
		"""
		same as Cloth_net but makes use of additional parameters for stiffness, shearing and bending
		:model_type: ... see diverse models below
		:encoder_name: e.g. tu-mobilevitv2_100 or resnet34 ... more examples: https://github.com/qubvel/segmentation_models.pytorch
		"""
		
		super(Cloth_net_param, self).__init__()
		if model_type=="Unet":
			SMP_model_type = smp.Unet
		elif model_type=="UnetPlusPlus":
			SMP_model_type = smp.UnetPlusPlus
		elif model_type=="MAnet":
			SMP_model_type = smp.MAnet
		elif model_type=="Linknet":
			SMP_model_type = smp.Linknet
		elif model_type=="FPN":
			SMP_model_type = smp.FPN
		elif model_type=="PSPNet":
			SMP_model_type = smp.PSPNet
		elif model_type=="PAN":
			SMP_model_type = smp.PAN
		elif model_type=="DeepLabV3":
			SMP_model_type = smp.DeepLabV3
		elif model_type=="DeepLabV3Plus":
			SMP_model_type = smp.DeepLabV3Plus
		else:
			raise Exception("invalid SMP_model_type!")
		self.model = SMP_model_type(encoder_name=encoder_name,in_channels=12+3,classes=3)
	
	def forward(self, x_v, stiffnesses, shearings, bendings, a=None):
		bs,c,h,w = x_v.shape
		device = x_v.device
		ones = torch.ones(1,1,h,w,device=device)
		di = torch.cat([x_v[:,:,1:]-x_v[:,:,:-1],torch.zeros(bs,c,1,w,device=device)],dim=2)
		dj = torch.cat([x_v[:,:,:,1:]-x_v[:,:,:,:-1],torch.zeros(bs,c,h,1,device=device)],dim=3)
		x = torch.cat([di,dj,torch.log(stiffnesses).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,torch.log(shearings).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,torch.log(bendings).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones],dim=1)
		x = self.model(x)
		return 10*torch.tanh(x/10)

class Cloth_net_param_a(nn.Module):
	
	def __init__(self, model_type, encoder_name):
		"""
		same as Cloth_net_param but takes additional parameter for external forces / accelerations
		:model_type: ... see diverse models below
		:encoder_name: e.g. tu-mobilevitv2_100 or resnet34 ... more examples: https://github.com/qubvel/segmentation_models.pytorch
		"""
		
		super(Cloth_net_param_a, self).__init__()
		if model_type=="Unet":
			SMP_model_type = smp.Unet
		elif model_type=="UnetPlusPlus":
			SMP_model_type = smp.UnetPlusPlus
		elif model_type=="MAnet":
			SMP_model_type = smp.MAnet
		elif model_type=="Linknet":
			SMP_model_type = smp.Linknet
		elif model_type=="FPN":
			SMP_model_type = smp.FPN
		elif model_type=="PSPNet":
			SMP_model_type = smp.PSPNet
		elif model_type=="PAN":
			SMP_model_type = smp.PAN
		elif model_type=="DeepLabV3":
			SMP_model_type = smp.DeepLabV3
		elif model_type=="DeepLabV3Plus":
			SMP_model_type = smp.DeepLabV3Plus
		else:
			raise Exception("invalid SMP_model_type!")
		self.model = SMP_model_type(encoder_name=encoder_name,in_channels=12+3+3,classes=3)
	
	def forward(self, x_v, stiffnesses, shearings, bendings, a):
		bs,c,h,w = x_v.shape
		device = x_v.device
		ones = torch.ones(1,1,h,w,device=device)
		di = torch.cat([x_v[:,:,1:]-x_v[:,:,:-1],torch.zeros(bs,c,1,w,device=device)],dim=2)
		dj = torch.cat([x_v[:,:,:,1:]-x_v[:,:,:,:-1],torch.zeros(bs,c,h,1,device=device)],dim=3)
		# CODO: normalize a to have 0-mean
		x = torch.cat([di,dj,torch.log(stiffnesses).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,torch.log(shearings).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,torch.log(bendings).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,a],dim=1)
		x = self.model(x)
		return 10*torch.tanh(x/10)
		
class Cloth_net_param_a_gated(nn.Module):
	
	def __init__(self, model_type, encoder_name):
		"""
		same as Cloth_net_param_a but allows to pass external accelerations through gating mechanism
		:model_type: ... see diverse models below
		:encoder_name: e.g. tu-mobilevitv2_100 or resnet34 ... more examples: https://github.com/qubvel/segmentation_models.pytorch
		"""
		
		super(Cloth_net_param_a_gated, self).__init__()
		if model_type=="Unet":
			SMP_model_type = smp.Unet
		elif model_type=="UnetPlusPlus":
			SMP_model_type = smp.UnetPlusPlus
		elif model_type=="MAnet":
			SMP_model_type = smp.MAnet
		elif model_type=="Linknet":
			SMP_model_type = smp.Linknet
		elif model_type=="FPN":
			SMP_model_type = smp.FPN
		elif model_type=="PSPNet":
			SMP_model_type = smp.PSPNet
		elif model_type=="PAN":
			SMP_model_type = smp.PAN
		elif model_type=="DeepLabV3":
			SMP_model_type = smp.DeepLabV3
		elif model_type=="DeepLabV3Plus":
			SMP_model_type = smp.DeepLabV3Plus
		else:
			raise Exception("invalid SMP_model_type!")
		self.model = SMP_model_type(encoder_name=encoder_name,in_channels=12+3+3,classes=3+2)
	
	def forward(self, x_v, stiffnesses, shearings, bendings, a):
		bs,c,h,w = x_v.shape
		device = x_v.device
		ones = torch.ones(1,1,h,w,device=device)
		di = torch.cat([x_v[:,:,1:]-x_v[:,:,:-1],torch.zeros(bs,c,1,w,device=device)],dim=2)
		dj = torch.cat([x_v[:,:,:,1:]-x_v[:,:,:,:-1],torch.zeros(bs,c,h,1,device=device)],dim=3)
		# CODO: normalize a to have 0-mean
		x = torch.cat([di,dj,torch.log(stiffnesses).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,torch.log(shearings).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,torch.log(bendings).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,a],dim=1)
		x = self.model(x)
		return 10*torch.tanh(x[:,0:3]/10)*torch.sigmoid(x[:,3:4])+a*torch.sigmoid(x[:,4:5])

class Cloth_net_param_a_gated2(nn.Module):
	
	def __init__(self, model_type, encoder_name):
		"""
		same as Cloth_net_param_a_gated but passes additionally normalized external forces (improves performance for small external forces)
		:model_type: ... see diverse models below
		:encoder_name: e.g. tu-mobilevitv2_100 or resnet34 ... more examples: https://github.com/qubvel/segmentation_models.pytorch
		"""
		
		super(Cloth_net_param_a_gated2, self).__init__()
		if model_type=="Unet":
			SMP_model_type = smp.Unet
		elif model_type=="UnetPlusPlus":
			SMP_model_type = smp.UnetPlusPlus
		elif model_type=="MAnet":
			SMP_model_type = smp.MAnet
		elif model_type=="Linknet":
			SMP_model_type = smp.Linknet
		elif model_type=="FPN":
			SMP_model_type = smp.FPN
		elif model_type=="PSPNet":
			SMP_model_type = smp.PSPNet
		elif model_type=="PAN":
			SMP_model_type = smp.PAN
		elif model_type=="DeepLabV3":
			SMP_model_type = smp.DeepLabV3
		elif model_type=="DeepLabV3Plus":
			SMP_model_type = smp.DeepLabV3Plus
		else:
			raise Exception("invalid SMP_model_type!")
		self.model = SMP_model_type(encoder_name=encoder_name,in_channels=12+3+3+3,classes=3+2)
	
	def forward(self, x_v, stiffnesses, shearings, bendings, a):
		bs,c,h,w = x_v.shape
		device = x_v.device
		ones = torch.ones(1,1,h,w,device=device)
		di = torch.cat([x_v[:,:,1:]-x_v[:,:,:-1],torch.zeros(bs,c,1,w,device=device)],dim=2)
		dj = torch.cat([x_v[:,:,:,1:]-x_v[:,:,:,:-1],torch.zeros(bs,c,h,1,device=device)],dim=3)
		# CODO: normalize a to have 0-mean
		a_norm = torch.nn.functional.normalize(a,1)
		x = torch.cat([di,dj,torch.log(stiffnesses).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,torch.log(shearings).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,torch.log(bendings).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,a,a_norm],dim=1)
		x = self.model(x)
		return 10*torch.tanh(x[:,0:3]/10)*torch.sigmoid(x[:,3:4])+a*torch.sigmoid(x[:,4:5])
		
class Cloth_net_param_a_gated3(nn.Module):
	
	def __init__(self, model_type, encoder_name):
		"""
		same as Cloth_net_param_a_gated2 but doesn't pass dx but dx-L. This way only the deviations from the resting length are passed through the network => hopefully, this helps the network to learn better dynamics.
		:model_type: ... see diverse models below
		:encoder_name: e.g. tu-mobilevitv2_100 or resnet34 ... more examples: https://github.com/qubvel/segmentation_models.pytorch
		"""
		
		super(Cloth_net_param_a_gated3, self).__init__()
		if model_type=="Unet":
			SMP_model_type = smp.Unet
		elif model_type=="UnetPlusPlus":
			SMP_model_type = smp.UnetPlusPlus
		elif model_type=="MAnet":
			SMP_model_type = smp.MAnet
		elif model_type=="Linknet":
			SMP_model_type = smp.Linknet
		elif model_type=="FPN":
			SMP_model_type = smp.FPN
		elif model_type=="PSPNet":
			SMP_model_type = smp.PSPNet
		elif model_type=="PAN":
			SMP_model_type = smp.PAN
		elif model_type=="DeepLabV3":
			SMP_model_type = smp.DeepLabV3
		elif model_type=="DeepLabV3Plus":
			SMP_model_type = smp.DeepLabV3Plus
		else:
			raise Exception("invalid SMP_model_type!")
		self.model = SMP_model_type(encoder_name=encoder_name,in_channels=12+3+3+3+6,classes=3+2)
	
	def forward(self, x_v, stiffnesses, shearings, bendings, a):
		bs,c,h,w = x_v.shape
		device = x_v.device
		ones = torch.ones(1,1,h,w,device=device)
		di = torch.cat([x_v[:,:,1:]-x_v[:,:,:-1],torch.zeros(bs,c,1,w,device=device)],dim=2)
		dj = torch.cat([x_v[:,:,:,1:]-x_v[:,:,:,:-1],torch.zeros(bs,c,h,1,device=device)],dim=3)
		dxi = di[:,:3]
		dxi = dxi - params.L_0*torch.nn.functional.normalize(dxi,1)
		dxj = dj[:,:3]
		dxj = dxj - params.L_0*torch.nn.functional.normalize(dxj,1)
		
		a_norm = torch.nn.functional.normalize(a,1)
		x = torch.cat([dxi,di,dxj,dj,
				       torch.log(stiffnesses).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,
					   torch.log(shearings).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,
					   torch.log(bendings).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,
					   a,a_norm
					  ],dim=1)
		x = self.model(x)
		return 10*torch.tanh(x[:,0:3]/10)*torch.sigmoid(x[:,3:4])+a*torch.sigmoid(x[:,4:5])

normalize = torch.nn.functional.normalize
def cross_prod(v1,v2,dim=1):
	return torch.linalg.cross(v1,v2,dim=dim)
class Cloth_net_param_a_gated5(nn.Module):
	
	def __init__(self, model_type, encoder_name):
		"""
		same as Cloth_net_param_a_gated4 but with basis transormations in order to achieve rotational equivariance
		:model_type: ... see diverse models below
		:encoder_name: e.g. tu-mobilevitv2_100 or resnet34 ... more examples: https://github.com/qubvel/segmentation_models.pytorch
		"""
		
		super(Cloth_net_param_a_gated5, self).__init__()
		if model_type=="Unet":
			SMP_model_type = smp.Unet
		elif model_type=="UnetPlusPlus":
			SMP_model_type = smp.UnetPlusPlus
		elif model_type=="MAnet":
			SMP_model_type = smp.MAnet
		elif model_type=="Linknet":
			SMP_model_type = smp.Linknet
		elif model_type=="FPN":
			SMP_model_type = smp.FPN
		elif model_type=="PSPNet":
			SMP_model_type = smp.PSPNet
		elif model_type=="PAN":
			SMP_model_type = smp.PAN
		elif model_type=="DeepLabV3":
			SMP_model_type = smp.DeepLabV3
		elif model_type=="DeepLabV3Plus":
			SMP_model_type = smp.DeepLabV3Plus
		else:
			raise Exception("invalid SMP_model_type!")
		self.model = SMP_model_type(encoder_name=encoder_name,in_channels=(3*4+1)*3,classes=3)#+2)
		self.input_norm = torch.nn.BatchNorm2d((3*4+1)*3)
	
	def forward(self, x_v, stiffnesses, shearings, bendings, a):
		bs,c,h,w = x_v.shape
		device = x_v.device
		
		# create "extended" version of x with extra padding (to compute finite differences)
		x_v_ext = torch.cat([x_v[:,:,0:1],x_v,x_v[:,:,-1:]],2)
		x_v_ext = torch.cat([x_v_ext[:,:,:,0:1],x_v_ext,x_v_ext[:,:,:,-1:]],3)
		x_ext = x_v_ext[:,:3]
		v_ext = x_v_ext[:,3:]
		
		# compute "basis" vectors (don't confuse with velocity v!)
		v1 = normalize(x_ext[:,:,2:,1:-1]-x_ext[:,:,:-2,1:-1],p=2,dim=1)
		v2 = normalize(x_ext[:,:,1:-1,2:]-x_ext[:,:,1:-1,:-2],p=2,dim=1)
		v3 = normalize(cross_prod(v1,v2),p=2,dim=1)
		v4 = normalize(v1+v2,p=2,dim=1)
		v5 = normalize(v1-v2,p=2,dim=1)
		
		V = torch.cat([v.unsqueeze(2) for v in [v4,v5,v3]],dim=2) # basis matrix
		
		# compute dxi
		dx1 = x_ext[:,:, 2:,1:-1]-x_ext[:,:,1:-1,1:-1]
		dx2 = x_ext[:,:,1:-1, 2:]-x_ext[:,:,1:-1,1:-1]
		dx3 = x_ext[:,:,:-2,1:-1]-x_ext[:,:,1:-1,1:-1]
		dx4 = x_ext[:,:,1:-1,:-2]-x_ext[:,:,1:-1,1:-1]
		
		# directions to neighboring nodes
		dn1 = normalize(dx1,p=2,dim=1)
		dn2 = normalize(dx2,p=2,dim=1)
		dn3 = normalize(dx3,p=2,dim=1)
		dn4 = normalize(dx4,p=2,dim=1)
		
		# distances from resting lengths
		dx1 = dx1 - params.L_0*normalize(dx1,p=2,dim=1)
		dx2 = dx2 - params.L_0*normalize(dx2,p=2,dim=1)
		dx3 = dx3 - params.L_0*normalize(dx3,p=2,dim=1)
		dx4 = dx4 - params.L_0*normalize(dx4,p=2,dim=1)
		
		# compute dvi
		dv1 = v_ext[:,:, 2:,1:-1]-v_ext[:,:,1:-1,1:-1]
		dv2 = v_ext[:,:,1:-1, 2:]-v_ext[:,:,1:-1,1:-1]
		dv3 = v_ext[:,:,:-2,1:-1]-v_ext[:,:,1:-1,1:-1]
		dv4 = v_ext[:,:,1:-1,:-2]-v_ext[:,:,1:-1,1:-1]
		
		# transform dxi / v / a into new basis => könnte man mit torch.cat in eine operation verwandeln (wäre evtl etwas effizienter)
		Vdn1 = torch.einsum("abcde,abde->acde",V,dn1)
		Vdn2 = torch.einsum("abcde,abde->acde",V,dn2)
		Vdn3 = torch.einsum("abcde,abde->acde",V,dn3)
		Vdn4 = torch.einsum("abcde,abde->acde",V,dn4)
		Vdx1 = torch.einsum("abcde,abde->acde",V,dx1)
		Vdx2 = torch.einsum("abcde,abde->acde",V,dx2)
		Vdx3 = torch.einsum("abcde,abde->acde",V,dx3)
		Vdx4 = torch.einsum("abcde,abde->acde",V,dx4)
		Vdv1 = torch.einsum("abcde,abde->acde",V,dv1)
		Vdv2 = torch.einsum("abcde,abde->acde",V,dv2)
		Vdv3 = torch.einsum("abcde,abde->acde",V,dv3)
		Vdv4 = torch.einsum("abcde,abde->acde",V,dv4)
		Va = torch.einsum("abcde,abde->acde",V,a)
		
		# create input batch
		# batchnorm at the beginning => scheint gar keine so schlechte Idee zu sein: https://stackoverflow.com/questions/46771939/batch-normalization-instead-of-input-normalization
		batch_in = self.input_norm(torch.cat([Vdn1,Vdn2,Vdn3,Vdn4,Vdx1,Vdx2,Vdx3,Vdx4,Vdv1,Vdv2,Vdv3,Vdv4,Va],dim=1))
		
		# apply neural network
		batch_out = self.model(batch_in)
		
		batch_out = 0.3*torch.tanh(batch_out/3)
		
		# transform output back to xyz
		a_out = torch.einsum("abcde,acde->abde",V,batch_out[:,:3])
		
		return a_out

class Cloth_Unet(nn.Module):
	
	def __init__(self, hidden_size):
		"""
		:hidden_size: hidden_size of UNet
		"""
		
		super(Cloth_Unet, self).__init__()
		self.hidden_size = hidden_size
		self.conv1 = nn.Conv2d( 12, self.hidden_size,kernel_size=[3,3],padding=[1,1])
		self.conv2 = nn.Conv2d( self.hidden_size, self.hidden_size,kernel_size=5,padding=0,stride=2)
		self.conv3 = nn.Conv2d( self.hidden_size, self.hidden_size,kernel_size=5,padding=0,stride=2)
		self.conv4 = nn.Conv2d( self.hidden_size, self.hidden_size,kernel_size=5,padding=0,stride=2)
		self.conv5 = nn.Conv2d( self.hidden_size, self.hidden_size,kernel_size=5,padding=0,stride=2)
		self.deconv1 = nn.ConvTranspose2d( self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.deconv2 = nn.ConvTranspose2d( 2*self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.deconv3 = nn.ConvTranspose2d( 2*self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.deconv4 = nn.ConvTranspose2d( 2*self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.deconv5 = nn.ConvTranspose2d( 2*self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.conv6 = nn.Conv2d( 2*self.hidden_size,3,kernel_size=[3,3],padding=[1,1])
	
	def forward(self,x_v, stiffnesses=None, shearings=None, bendings=None, a=None):
		bs,c,h,w = x_v.shape
		device = x_v.device
		di = torch.cat([x_v[:,:,1:]-x_v[:,:,:-1],torch.zeros(bs,c,1,w,device=device)],dim=2)
		dj = torch.cat([x_v[:,:,:,1:]-x_v[:,:,:,:-1],torch.zeros(bs,c,h,1,device=device)],dim=3)
		x = torch.cat([di,dj],dim=1)
		
		x1 = torch.sigmoid(self.conv1(x))
		x2 = torch.sigmoid(self.conv2(x1))
		x3 = torch.sigmoid(self.conv3(x2))
		x4 = torch.sigmoid(self.conv4(x3))
		x = torch.sigmoid(self.conv5(x4))
		x = torch.sigmoid(self.deconv1(x, output_size = [x4.shape[2],x4.shape[3]]))
		x = torch.cat([x,x4],dim=1)
		x = torch.sigmoid(self.deconv2(x, output_size = [x3.shape[2],x3.shape[3]]))
		x = torch.cat([x,x3],dim=1)
		x = torch.sigmoid(self.deconv4(x, output_size = [x2.shape[2],x2.shape[3]]))
		x = torch.cat([x,x2],dim=1)
		x = torch.sigmoid(self.deconv5(x, output_size = [x1.shape[2],x1.shape[3]]))
		x = torch.cat([x,x1],dim=1)
		x = self.conv6(x)
		
		return 10*torch.tanh(x/10)

class Cloth_Unet_param_a(nn.Module):
	
	def __init__(self, hidden_size):
		"""
		:hidden_size: hidden_size of UNet
		"""
		
		super(Cloth_Unet_param_a, self).__init__()
		self.hidden_size = hidden_size
		self.conv1 = nn.Conv2d( 12+3+3, self.hidden_size,kernel_size=[3,3],padding=[1,1])
		self.conv2 = nn.Conv2d( self.hidden_size, self.hidden_size,kernel_size=5,padding=0,stride=2)
		self.conv3 = nn.Conv2d( self.hidden_size, self.hidden_size,kernel_size=5,padding=0,stride=2)
		self.conv4 = nn.Conv2d( self.hidden_size, self.hidden_size,kernel_size=5,padding=0,stride=2)
		self.conv5 = nn.Conv2d( self.hidden_size, self.hidden_size,kernel_size=5,padding=0,stride=2)
		self.deconv1 = nn.ConvTranspose2d( self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.deconv2 = nn.ConvTranspose2d( 2*self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.deconv3 = nn.ConvTranspose2d( 2*self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.deconv4 = nn.ConvTranspose2d( 2*self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.deconv5 = nn.ConvTranspose2d( 2*self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.conv6 = nn.Conv2d( 2*self.hidden_size,3,kernel_size=[3,3],padding=[1,1])
	
	def forward(self,x_v, stiffnesses, shearings, bendings, a):
		bs,c,h,w = x_v.shape
		device = x_v.device
		ones = torch.ones(1,1,h,w,device=device)
		di = torch.cat([x_v[:,:,1:]-x_v[:,:,:-1],torch.zeros(bs,c,1,w,device=device)],dim=2)
		dj = torch.cat([x_v[:,:,:,1:]-x_v[:,:,:,:-1],torch.zeros(bs,c,h,1,device=device)],dim=3)
		x = torch.cat([di,dj,torch.log(stiffnesses).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,torch.log(shearings).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,torch.log(bendings).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,a],dim=1)
		
		x1 = torch.sigmoid(self.conv1(x))
		x2 = torch.sigmoid(self.conv2(x1))
		x3 = torch.sigmoid(self.conv3(x2))
		x4 = torch.sigmoid(self.conv4(x3))
		x = torch.sigmoid(self.conv5(x4))
		x = torch.sigmoid(self.deconv1(x, output_size = [x4.shape[2],x4.shape[3]]))
		x = torch.cat([x,x4],dim=1)
		x = torch.sigmoid(self.deconv2(x, output_size = [x3.shape[2],x3.shape[3]]))
		x = torch.cat([x,x3],dim=1)
		x = torch.sigmoid(self.deconv4(x, output_size = [x2.shape[2],x2.shape[3]]))
		x = torch.cat([x,x2],dim=1)
		x = torch.sigmoid(self.deconv5(x, output_size = [x1.shape[2],x1.shape[3]]))
		x = torch.cat([x,x1],dim=1)
		x = self.conv6(x)
		
		return 10*torch.tanh(x/10)
