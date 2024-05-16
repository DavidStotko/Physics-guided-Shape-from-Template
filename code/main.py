# run this file from the "Code" directory using
# python .\main.py --json=[scenes/R1.json,scenes/R2.json,scenes/R3.json,scenes/R4.json,scenes/R5.json,scenes/R6.json,scenes/R7.json,scenes/R8.json,scenes/R9.json] --net=SMP_param_a_gated3 --load_index=network

import numpy as np
import torch
from ema_pytorch import EMA
from torchvision.transforms.functional import gaussian_blur
import nvdiffrast.torch as dr
import json
import time
import openmesh as om
from PIL import Image

from cloth_net import get_Net
from logger import Logger
from get_param import params,toCuda,get_hyperparam
import evaluation

def render(
    context,
    vertices,
    triangles,
    vertex_attributes,
    texture,
    camera_transforms,
    resolution,
    antialias = True,
    crop = True,
):
    device = vertices.device
    n = camera_transforms.shape[0]
    v = vertices.shape[0]

    vertices_hom = torch.cat([vertices, torch.ones([v, 1], device = device, dtype = torch.float32)], axis=1)
    vertices_pixel = torch.matmul(vertices_hom.expand(n, v, -1), torch.transpose(camera_transforms, -2, -1))
    
    rast, diff_rast = dr.rasterize(context, vertices_pixel, triangles, resolution = resolution)
    image_attributes, _ = dr.interpolate(vertex_attributes, rast, triangles, rast_db = diff_rast, diff_attrs = None)
    color = dr.texture(texture, uv = image_attributes[...,[0, 1]], filter_mode="linear")
    if antialias:
        color = dr.antialias(color, rast, vertices_pixel, triangles)

    image_properties = torch.cat([color, torch.ones(*color.shape[:-1], 1, device = device)], dim = -1)

    if crop:
        image_properties = torch.where(rast[..., 3:] > 0, image_properties, torch.tensor([0.0], device = device))
    
    return image_properties

def opencv_projection(
    image_size,     # width, height
    optical_center, # c in pixel
    focal_lengths,  # f in pixel
    z_near,
    z_far,
):
    optical_shifts = (image_size - 2 * optical_center) / image_size
    optical_shifts[0] *= -1
    relative_focal_lengths = -2 * focal_lengths / image_size

    projection = torch.zeros([4, 4], dtype = torch.float32, device = "cuda")
    projection[0, 0] = relative_focal_lengths[0]
    projection[1, 1] = relative_focal_lengths[1]
    projection[0, 2] = optical_shifts[0]
    projection[1, 2] = optical_shifts[1]
    projection[2, 2] = -(z_far + z_near) / (z_far - z_near)
    projection[2, 3] = -2 * z_far * z_near / (z_far - z_near)
    projection[3, 2] = -1.0

    return projection


class Optimization():
    def __init__(self):
        pass
    
    def initializeParameters(self, scene, evaluate):
        self.scene_parameters = scene

        self.n_x = 32
        self.n_y = 32
        self.frame_counter = 0
        self.frames_per_epoch = min(10, self.scene_parameters["n_images"])
        self.new_frame_period = 5
        self.epoch_counter = 0
        self.simulation_frames = self.scene_parameters["n_images"]
        self.evaluate = evaluate

        self.time_conversion = 1/0.02                                           # 1s = 50 [NN-t]
        self.length_conversion = (self.n_x - 1)/self.scene_parameters["mesh_size"] # 1m = 31 [NN-m]

    def initializeMesh(self):
        mesh = om.read_trimesh(self.scene_parameters["mesh_file"], vertex_tex_coord = True)
        self.rest_positions = torch.zeros([len(mesh.vertices()), 3], device = "cuda")
        self.faces = torch.zeros([len(mesh.faces()), 3], dtype = torch.int32, device = "cuda")
        self.uv = torch.zeros([len(mesh.vertices()), 2], device = "cuda")

        vertex_i = 0
        for vertex in mesh.vertices():
            self.rest_positions[vertex_i] = torch.from_numpy(mesh.point(vertex))
            self.uv[vertex_i] = torch.from_numpy(mesh.texcoord2D(vertex))
            vertex_i += 1

        face_i = 0
        for face in mesh.faces():
            vertex_i = 0
            for vh in mesh.fv(face):
                self.faces[face_i, vertex_i] = vh.idx()
                vertex_i += 1
            face_i += 1

        self.positions = self.rest_positions.clone()
        self.rest_positions = self.rest_positions * self.length_conversion

        self.rest_positions = self.rest_positions.clone().requires_grad_(True)
        self.original_uv = self.uv.clone()
        self.uv = self.uv.clone().requires_grad_(True)

    def initializeNetwork(self):
        network = toCuda(get_Net(params))
        ema_net = EMA(
            network,
            beta = params.ema_beta,                              # exponential moving average factor
            update_after_step = params.ema_update_after_step,    # only after this number of .update() calls will it start updating
            update_every = params.ema_update_every,              # how often to actually update, to save on compute (updates every 10th .update() call)
            power = 3.0/4.0,
            include_online_model = True
        )
        
        logger = Logger(get_hyperparam(params),use_csv=False,use_tensorboard=False)
        date_time,index = logger.load_state(ema_net,None,datetime=params.load_date_time,index=params.load_index)
        print(f"loaded: {date_time}, {index}")
        self.cloth_net = ema_net
        self.cloth_net.eval()

        positions_net = self.rest_positions.transpose(0, 1).view(3, self.n_x, self.n_y).type(torch.float32)
        velocities_net = torch.zeros(3, self.n_x, self.n_y, device = "cuda")
        self.x_v = torch.cat([positions_net, velocities_net]).unsqueeze(0)

    def ComputeViewMatrix(self, camera_position, camera_forward, camera_up):
        right = np.cross(camera_forward, camera_up)
        right /= np.linalg.norm(right)
        
        direction_matrix = np.array([[right[0], camera_up[0], - camera_forward[0], 0.0],
                                     [right[1], camera_up[1], - camera_forward[1], 0.0],
                                     [right[2], camera_up[2], - camera_forward[2], 0.0],
                                     [0.0, 0.0, 0.0, 1.0]],
                                     dtype = np.float32)
        position_matrix = np.array([[1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [- camera_position[0], - camera_position[1], - camera_position[2], 1.0]],
                                     dtype = np.float32)
        
        return position_matrix @ direction_matrix

    def initializeCameraMatrix(self):
        self.crop_size = self.scene_parameters["upper_right_corner"] - self.scene_parameters["lower_left_corner"]
        self.diffrast_size = self.scene_parameters["image_size"]
        
        self.projection = opencv_projection(image_size = self.diffrast_size[::-1],
                                            optical_center = self.scene_parameters["optical_center"],
                                            focal_lengths = self.scene_parameters["focal_length"],
                                            z_near = 0.01,
                                            z_far = 1000.0).unsqueeze_(0)
        self.optimizer_modelview = self.projection @ torch.from_numpy(self.ComputeViewMatrix(self.scene_parameters["camera_position"], self.scene_parameters["camera_forward"], self.scene_parameters["camera_up"]).T).to("cuda:0")

    def initializeOptimization(self):
        # initialize cloth parameters
        self.stretching_stiffness = torch.tensor([3000], device="cuda:0", dtype=torch.float32, requires_grad=True)
        self.shearing_stiffness   = torch.tensor([8],    device="cuda:0", dtype=torch.float32, requires_grad=True)
        self.bending_stiffness    = torch.tensor([0.5],  device="cuda:0", dtype=torch.float32, requires_grad=True)
        self.external_forces      = torch.tensor([ 0   *self.length_conversion/(self.time_conversion*self.time_conversion),
                                                  -9.81*self.length_conversion/(self.time_conversion*self.time_conversion),
                                                   0   *self.length_conversion/(self.time_conversion*self.time_conversion)],
                                                  device="cuda:0",
                                                  dtype=torch.float32).unsqueeze(0).unsqueeze(2).unsqueeze(3).requires_grad_(True)
        self.vertex_forces        = torch.zeros((1, self.simulation_frames, 3, self.n_x, self.n_y), device="cuda:0", dtype=torch.float32, requires_grad=True)

        self.parameters = [self.stretching_stiffness,
                           self.shearing_stiffness,
                           self.bending_stiffness,
                           self.external_forces,
                           self.vertex_forces,
                           self.uv
        ]
        self.optimizer = [None] * len(self.parameters)
        learning_rates = [5e1, 1e-1, 1e-2, 5e-2*self.length_conversion/(self.time_conversion*self.time_conversion), 1e-3, 2e-4]
        for i in range(len(self.parameters)):
            self.optimizer[i] = torch.optim.Adam([self.parameters[i]], lr=learning_rates[i])
        
        self.loss = torch.tensor([0.], dtype=torch.float32).to("cuda:0")

    def loadGroundTruth(self):
        self.gt_images = torch.ones((self.simulation_frames + 1, 
                                     self.crop_size[0], 
                                     self.crop_size[1],
                                     4), 
                                    dtype=torch.float32).to("cuda:0")
    
        for i in range(self.simulation_frames + 1):
            image = torch.from_numpy(np.array(Image.open(self.scene_parameters["image_files"] + str(i).zfill(3) + ".png"), dtype = np.float32)).to("cuda:0")
            self.gt_images[i,:,:,:3] = image[self.scene_parameters["lower_left_corner"][0]:self.scene_parameters["upper_right_corner"][0],
                                             self.scene_parameters["lower_left_corner"][1]:self.scene_parameters["upper_right_corner"][1],
                                             :3]

            mask = torch.from_numpy(np.array(Image.open(self.scene_parameters["mask_files"] + str(i).zfill(3) + ".png"), dtype = np.float32)).to("cuda:0")
            if mask.dim() == 2:
                mask = mask.unsqueeze(2)
            self.gt_images[i,:,:,3] = torch.mean(mask[self.scene_parameters["lower_left_corner"][0]:self.scene_parameters["upper_right_corner"][0],
                                                      self.scene_parameters["lower_left_corner"][1]:self.scene_parameters["upper_right_corner"][1],
                                                      :3],
                                                 dim = 2)

        self.gt_images = torch.flip(self.gt_images, dims = [1]) / 255.0

        self.blurred_gt_images = self.gt_images.clone()
        self.blurred_gt_images = torch.permute(self.blurred_gt_images, (0, 3, 1, 2))
        self.blurred_gt_images = gaussian_blur(self.blurred_gt_images, 21, 7)
        self.blurred_gt_images = torch.permute(self.blurred_gt_images, (0, 2, 3, 1))


    def updateMesh(self):
        # apply cloth net
        a_ext = torch.ones(1,3,self.n_x,self.n_y, device = "cuda") * self.external_forces
        a_ext = a_ext + self.vertex_forces[:, self.frame_counter - 1]
        a = self.cloth_net(self.x_v, self.stretching_stiffness, self.shearing_stiffness, self.bending_stiffness, a_ext)
        # integrate accelerations
        v_new = self.x_v[:,3:] + a
        # apply boundary conditions
        v_new[:,:,0,0] = self.vertex_forces[:,self.frame_counter - 1,:,0,0]
        v_new[:,:,-1,0] = self.vertex_forces[:,self.frame_counter - 1,:,-1,0]
        x_new = self.x_v[:,:3] + v_new
        # update x_v
        self.x_v = torch.cat([x_new,v_new],dim=1)

        with torch.no_grad():
            self.positions[:] = x_new.view(3, -1).transpose(0, 1) / self.length_conversion

        return x_new

    def renderDiffrast(self, x_new):
        vertices = x_new.view(3, -1).transpose(0, 1).contiguous() / (self.length_conversion)
        diffrast_attributes = torch.concat([self.uv], axis=-1).unsqueeze(0)
        
        return render(self.context, 
                      vertices,
                      self.faces, 
                      diffrast_attributes,
                      self.texture_image,
                      self.optimizer_modelview,
                      self.diffrast_size)

    def processImages(self, image):
        image = torch.flip(image, dims = [1,2])
        image = image.squeeze(0)
        # crop image
        image = image[self.scene_parameters["lower_left_corner"][0]:self.scene_parameters["upper_right_corner"][0],
                      self.scene_parameters["lower_left_corner"][1]:self.scene_parameters["upper_right_corner"][1]]
        blurred_image = torch.permute(image, (2, 0, 1))
        blurred_image = gaussian_blur(blurred_image, 21, 7)
        blurred_image = torch.permute(blurred_image, (1, 2, 0))

        image_diff         = image         - self.gt_images[self.frame_counter]
        blurred_image_diff = blurred_image - self.blurred_gt_images[self.frame_counter]

        return image_diff, blurred_image_diff

    def printQuantities(self):
        t = time.perf_counter()
        if self.epoch_counter % 50 == 0:
            print(f"{self.epoch_counter:5d} | {t - self.time:.2f} s | {t - self.time_start:7.2f} s | "
                f"Loss: {self.loss.item() * self.frames_per_epoch:.2e} {self.loss.item():.2e} | "
                f"{self.stretching_stiffness.item():.3e} "
                f"{self.shearing_stiffness.item():.3e} "
                f"{self.bending_stiffness.item():.3e} "
                f"{(self.external_forces[0, 0, 0, 0].item() / self.length_conversion * self.time_conversion * self.time_conversion): .2e} "
                f"{(self.external_forces[0, 1, 0, 0].item() / self.length_conversion * self.time_conversion * self.time_conversion): .2e} "
                f"{(self.external_forces[0, 2, 0, 0].item() / self.length_conversion * self.time_conversion * self.time_conversion): .2e}   "
                f"{self.vertex_shift_loss:.2e}    "
                f"{self.chamfer_distance.item():.2e}")
        self.time = t

    def resetState(self):
        positions_net = self.rest_positions.transpose(0, 1).view(3, self.n_x, self.n_y).type(torch.float32)
        velocities_net = torch.zeros(3, self.n_x, self.n_y, device = "cuda")
        self.x_v = torch.cat([positions_net, velocities_net]).unsqueeze(0)
        with torch.no_grad():
            self.positions[:] = self.rest_positions / self.length_conversion
            self.loss = torch.tensor([0.], dtype=torch.float32, device = "cuda")
            # remove temporally and spacially constant part that could be modeled by wind
            self.vertex_forces -= torch.mean(self.vertex_forces, dim = [1, 3, 4]).unsqueeze_(1).unsqueeze_(3).unsqueeze_(4)

    def clampOptimization(self):
        with torch.no_grad():
            self.stretching_stiffness[0] = max(10, self.stretching_stiffness[0])
            self.shearing_stiffness[0] = max(1e-2, self.shearing_stiffness[0])
            self.bending_stiffness[0] = max(1e-5, self.bending_stiffness[0])
            # self.external_forces[0][0][0][0] = 0*self.length_conversion/(self.time_conversion*self.time_conversion)
            self.external_forces[0][1][0][0] = -9.81*self.length_conversion/(self.time_conversion*self.time_conversion)
            # self.external_forces[0][2][0][0] = 0*self.length_conversion/(self.time_conversion*self.time_conversion)

    def initialize(self, scene, max_epochs, evaluate):
        print("Start: Initialization")
        t_start = time.perf_counter()

        self.initializeParameters(scene, evaluate)
        self.initializeMesh()
        self.initializeNetwork()
        self.initializeCameraMatrix()
        self.initializeOptimization()

        # For NvDiffRast
        self.context = dr.RasterizeCudaContext()
        # self.context = dr.RasterizeGLContext()
        self.texture_image = Image.open(self.scene_parameters["texture_file"]).resize((1024,1024), Image.Resampling.BICUBIC)
        self.texture_image = torch.from_numpy(np.array(self.texture_image, dtype = np.float32)).to("cuda:0").unsqueeze(0) / 255.0
        self.texture_image = torch.flip(self.texture_image, dims = [1])

        self.loadGroundTruth()

        if evaluate:
            ground_truth_point_clouds, point_clouds_lengths = evaluation.loadGroundTruth(self.scene_parameters)
            our_point_clouds = torch.zeros_like(ground_truth_point_clouds)
            self.point_clouds = {"ground_truth" : ground_truth_point_clouds, "ours" : our_point_clouds, "lengths" : point_clouds_lengths}
            self.chamfer_distance = torch.tensor([0.0])
            self.chamfer_distances_epochs = torch.zeros((max_epochs), device="cuda")
        
        t_end = time.perf_counter()
        print(f"Done:  Initialization in {t_end - t_start:.3f} s\n")
        print("Epoch |  Time  |  Total t  | Loss:   Full  per Frame |  Stretch    Shear     Bend      Wind x    Wind y    Wind z    Vertex F  Chamfer dist")
        print("------+--------+-----------+-------------------------+-------------------------------------------------------------------------------------")
        
        self.time = time.perf_counter()
        self.time_start = time.perf_counter()


    def step(self):
        if self.frame_counter == 0 and self.evaluate:
            with torch.no_grad():
                self.point_clouds["ours"][self.frame_counter, :self.point_clouds["lengths"][self.frame_counter]] = evaluation.sampleMesh(self.point_clouds["lengths"][self.frame_counter].item(), self.positions, self.faces)
                if self.epoch_counter == 250:
                    torch.save(self.point_clouds["ours"][self.frame_counter, :self.point_clouds["lengths"][self.frame_counter]].clone(), "../evaluation/" + self.scene_parameters["scene"] + "/ours/point_cloud_" + str(self.frame_counter).zfill(3) + ".pt")

        self.frame_counter += 1

        ### PHYSICS
        x_new = self.updateMesh()
        
        if self.evaluate:
            with torch.no_grad():
                self.point_clouds["ours"][self.frame_counter, :self.point_clouds["lengths"][self.frame_counter]] = evaluation.sampleMesh(self.point_clouds["lengths"][self.frame_counter].item(), self.positions, self.faces)
                if self.epoch_counter == 250:
                    torch.save(self.point_clouds["ours"][self.frame_counter, :self.point_clouds["lengths"][self.frame_counter]].clone(), self.scene_parameters["result_point_cloud_files"] + str(self.frame_counter).zfill(3) + ".pt")

        ### DIFFRAST
        image = self.renderDiffrast(x_new)
        image_diff, blurred_image_diff = self.processImages(image)
        
        ### OPTIMIZATION
        self.loss += torch.mean(torch.abs(image_diff[..., :3])) # IMAGE LOSS
        self.loss += 1.0 * torch.mean(torch.abs(blurred_image_diff[..., 3])) # SILHOUETTE LOSS

        if self.frame_counter == self.frames_per_epoch:
            self.chamfer_distance = torch.tensor([0.0])
            with torch.no_grad():
                if self.evaluate or self.epoch_counter == 250:
                    self.chamfer_distance = evaluation.computeChamferDistance(self.point_clouds["ground_truth"], self.point_clouds["ours"], 0, self.frame_counter + 1, self.point_clouds["lengths"])
                    self.chamfer_distances_epochs[self.epoch_counter] = self.chamfer_distance
            
            # mean over all frames
            self.loss = self.loss / self.frames_per_epoch

            # VERTEX SHIFT REGULARIZATION
            self.vertex_shift_loss = 0

            exponent = 2
            self.vertex_shift_loss = self.vertex_shift_loss + 1e-2 * torch.mean(torch.linalg.norm(  self.vertex_forces[:, :self.frames_per_epoch], dim = 2)**exponent)
            self.vertex_shift_loss = self.vertex_shift_loss + 1e-2 * torch.mean(torch.linalg.norm(  self.vertex_forces[:, 1:self.frames_per_epoch]
                                                                                        - self.vertex_forces[:, :self.frames_per_epoch-1], dim = 2)**exponent)
            self.vertex_shift_loss = self.vertex_shift_loss + 1e-3 * torch.mean(torch.linalg.norm(  self.vertex_forces[:, :self.frames_per_epoch, :, 1:]
                                                                                        - self.vertex_forces[:, :self.frames_per_epoch, :, :-1], dim = 2)**exponent)
            self.vertex_shift_loss = self.vertex_shift_loss + 1e-3 * torch.mean(torch.linalg.norm(  self.vertex_forces[:, :self.frames_per_epoch, :, :, 1:]
                                                                                        - self.vertex_forces[:, :self.frames_per_epoch, :, :, :-1], dim = 2)**exponent)
            self.loss += self.vertex_shift_loss
            
            self.printQuantities()

            loss_norm = self.loss.detach() + 1e-3
            self.loss = self.loss / loss_norm
            self.loss.backward()

            for i in range(len(self.parameters)):
                self.optimizer[i].step()
                self.optimizer[i].zero_grad()
            
            self.frame_counter = 0
            self.epoch_counter += 1

            self.resetState()
            self.clampOptimization()

            # successively add frames
            if self.frames_per_epoch < self.simulation_frames and self.epoch_counter % self.new_frame_period == 0:
                self.frames_per_epoch += 1
                if self.frames_per_epoch == self.simulation_frames:
                    print("Reached max frames")

def loadJson(file_name):
    with open(file_name) as json_file:
        dictionary = json.load(json_file)
        for key in ["camera_position", "camera_forward", "camera_up", "optical_center", "focal_length", "image_size", "lower_left_corner", "upper_right_corner"]:
            dictionary[key] = np.array(dictionary[key])
    return dictionary

def main():
    scene_list = list(map(str, params.json.strip('[]').split(',')))
    evaluate = params.evaluate
    for file_name in scene_list:
        scene = loadJson(file_name)
        time1 = time.perf_counter()
        opt = Optimization()

        max_epochs = 251

        opt.initialize(scene = scene, max_epochs = max_epochs, evaluate = evaluate)
        while(opt.epoch_counter < max_epochs):
            opt.step()
        print("------+--------+-----------+-------------------------+-------------------------------------------------------------------------------------")
        opt.printQuantities()
        
        if evaluate:
            torch.save(opt.chamfer_distances_epochs, scene["result_chamfer_file"])
        
        time2 = time.perf_counter()
        print(f"Done in {(time2 - time1): .2f} s")


main()