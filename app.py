import cv2
import numpy as np
from scipy.interpolate import griddata

def parse_obj(filename):
    vertices = []
    normals = []
    uvs = []
    faces = []
    colors = []

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                data = [float(x) for x in line[2:].split()]
                vertices.append(data[:3])
                colors.append(data[3:])
            elif line.startswith('vn '):
                normals.append([float(x) for x in line[3:].split()])
            elif line.startswith('vt '):
                uvs.append([float(x) for x in line[3:].split()])
            elif line.startswith('f '):
                triangle = [x.split('/') for x in line.split()[1:]]
                triangle = [[int(x) for x in vertex] for vertex in triangle]
                faces.append(triangle)

    data = {}
    data['vertices']    = np.array(vertices)
    data['normals']     = np.array(normals)
    data['uvs']         = np.array(uvs)
    data['faces']       = np.array(faces)
    data['colors']      = np.array(colors)

    return data

# Generate position image map to store vertices position info of a 3D model
def gen_pos_map(w, h, x_max, y_max, z_max):
    obj_name     = '20230702185753.obj'
    mask_name    = '20230702185753_mask.png'
    pos_map_name = '20230702185753.png'

    # load 3D geometry
    data = parse_obj(obj_name)

    # vertices point normalize & add a channel for opacity (p: rgba 0~1)
    p    = data['vertices']
    p   /= np.array([x_max, y_max, z_max])
    p    = np.concatenate([p, np.ones((p.shape[0], 1))], axis=-1)

    # fit uv points into the grid structure (w, h)
    u = (data['uvs'][:, 0] * w).astype(int)
    v = (data['uvs'][:, 1] * h).astype(int)
    v = h - v
    uv = np.column_stack((u, v))
    grid_u, grid_v = np.meshgrid(np.arange(0, w), np.arange(0, h))

    # generate the position map (w, h, rgba)
    position_map = griddata(uv, p, (grid_u, grid_v), method='nearest', fill_value=0)
    position_map = np.clip(position_map, 0, 1)

    # remove the part outside the mask (opacity 0)
    mask_img = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED)
    mask_img = mask_img.astype(np.float32) / 255
    mask_img = cv2.resize(mask_img, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
    mask = mask_img < 0.01
    position_map[mask, -1] = 0

    # save position map with unit16 resolution (also rgba -> bgra for opencv)
    position_map = (position_map * 65535).astype(np.uint16)
    position_map = position_map[:, :, [2, 1, 0, 3]]
    cv2.imwrite(pos_map_name, position_map, [cv2.IMWRITE_PNG_COMPRESSION, 9])

# Generate multiple sub mask (clustering)
def clustering(x_max, y_max, z_max):
    center_name  = 'scroll1_center.obj'
    pos_map_name = '20230702185753.png'

    # load scroll center into (used for cutting)
    data    = parse_obj(center_name)
    center  = data['vertices']
    center /= np.array([x_max, y_max, z_max])

    # load position map (rgba 0~1)
    position_map = cv2.imread(pos_map_name, cv2.IMREAD_UNCHANGED)
    position_map = position_map[:, :, [2, 1, 0, 3]]
    position_map = position_map.astype(np.float32) / 65535

    # cutting along the scroll center (edge: opacity 0)
    edge_x = np.interp(position_map[:, :, 2], center[:, 2], center[:, 0])
    # mask = abs(position_map[:, :, 0] - edge_x) < 0.01
    mask = abs(position_map[:, :, 0] - edge_x) < 0.003
    position_map[mask, -1] = 0

    # position_map = (position_map * 65535).astype(np.uint16)
    # position_map = position_map[:, :, [2, 1, 0, 3]]
    # cv2.imwrite('ok.png', position_map, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    # aa = (position_map * 255).astype(np.uint8)
    # aa = aa[:, :, [2, 1, 0, 3]]
    # cv2.imwrite('ok.png', aa)

    alpha_channel = position_map[:, :, 3]
    binary_alpha = (alpha_channel > 0.95).astype(np.uint8)

    contours, _ = cv2.findContours(binary_alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_count = 0
    total_area = position_map.shape[0] * position_map.shape[1]

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area / total_area < 0.0005): continue

        mask_count += 1
        mask = np.zeros_like(binary_alpha)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        cv2.imwrite(f'mask_{mask_count}.png', mask)

# w, h = 17381, 13513
w, h = 1738, 1351
x_max, y_max, z_max = 8096, 7888, 14370

# interp_x = np.interp(segment_data['vertices'][:, 2], center_data['vertices'][:, 2], center_data['vertices'][:, 0])
# interp_y = np.interp(segment_data['vertices'][:, 2], center_data['vertices'][:, 2], center_data['vertices'][:, 1])
# interp_centers = np.column_stack((interp_x, interp_y, segment_data['vertices'][:, 2]))

# angles = np.arctan2(segment_data['vertices'][:, 1] - interp_centers[:, 1], (segment_data['vertices'][:, 0] - interp_centers[:, 0]) * -1)

# u_value = (angles + np.pi / 2) / np.pi
# v_value = segment_data['vertices'][:, 2] / z_max
# # u_value -= 0.5
# # v_value -= 0.5
# # v_value *= 8

# flatten_data = segment_data
# flatten_data['vertices'][:, 0] = u_value
# flatten_data['vertices'][:, 1] = v_value
# flatten_data['vertices'][:, 2] = 0
# flatten_data['normals'][:, :] = [0, 0, 1]

# position map generate
# gen_pos_map(w, h, x_max, y_max, z_max)

# sub mask generate
clustering(x_max, y_max, z_max)




