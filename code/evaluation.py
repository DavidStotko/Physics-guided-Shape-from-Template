import pathlib
import torch
import pytorch3d.loss

def loadGroundTruth(scene_parameters):
    pathlist = [str(p) for p in sorted(pathlib.Path(scene_parameters["ground_truth_dir"]).rglob("*.pt"))]

    max_n_points = 0
    for path in pathlist:
        temp = torch.load(path).to("cuda:0")
        if temp.shape[0] > max_n_points:
            max_n_points = temp.shape[0]

    point_clouds = torch.zeros((len(pathlist), max_n_points, temp.shape[1]), device="cuda:0")
    point_clouds_lengths = torch.zeros(len(pathlist), dtype=torch.int64, device="cuda:0")
    counter = 0
    for path in pathlist:
        temp = torch.load(path).to("cuda:0")
        point_clouds[counter, :temp.shape[0]] = temp
        point_clouds_lengths[counter] = temp.shape[0]
        counter += 1

    return point_clouds, point_clouds_lengths


def sampleMesh(number, vertices, triangles):
    triangle_vertices = vertices[triangles]
    edges = triangle_vertices[:, 1:] - triangle_vertices[:, 0].unsqueeze(1)
    areas = torch.linalg.norm(torch.linalg.cross(edges[:, 0], edges[:, 1], dim = 1), dim = 1) # double of areas
    triangle_samples = torch.multinomial(input = areas, num_samples = number, replacement = True)

    uv = torch.rand([number, 2], device = "cuda")
    u = 1 - torch.sqrt(uv[:, 0])
    v = torch.sqrt(uv[:, 0]) * (1 - uv[:, 1])
    w = uv[:, 1] * torch.sqrt(uv[:, 0])
    uvw = torch.stack([u, v, w], dim = 1).unsqueeze_(2)

    points = torch.sum(triangle_vertices[triangle_samples] * uvw, dim = 1)
    return points


def computeChamferDistance(ground_truth_point_clouds, our_point_clouds, min_index, max_index, point_clouds_lengths):
    our_point_clouds[..., 0] = -our_point_clouds[..., 0]
    our_point_clouds[..., 1] = -our_point_clouds[..., 1]
    chamfer_distances = 1e4 * pytorch3d.loss.chamfer_distance(
                                  our_point_clouds[min_index:max_index],
                                  ground_truth_point_clouds[min_index:max_index],
                                  x_lengths=point_clouds_lengths[min_index:max_index],
                                  y_lengths=point_clouds_lengths[min_index:max_index],
                                  batch_reduction=None,
                              )[0]

    return torch.mean(chamfer_distances)
