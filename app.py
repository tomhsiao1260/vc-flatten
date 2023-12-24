import cv2
import numpy as np
from scipy.interpolate import griddata, splprep, splev

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
    segmentID    = '20230702185753'

    # load scroll center into (used for cutting)
    data    = parse_obj(center_name)
    center  = data['vertices']
    center /= np.array([x_max, y_max, z_max])

    # load position map (rgba 0~1)
    position_map = cv2.imread(pos_map_name, cv2.IMREAD_UNCHANGED)
    position_map = position_map[:, :, [2, 1, 0, 3]]
    position_map = position_map.astype(np.float32) / 65535

    # cutting along the scroll center (opacity change along the cutting edge)
    edge_x = np.interp(position_map[:, :, 2], center[:, 2], center[:, 0])
    mask_left = position_map[:, :, 0] - edge_x > 0
    mask_right = position_map[:, :, 0] - edge_x <= 0

    position_map_left = np.copy(position_map)
    position_map_right = np.copy(position_map)
    position_map_left[mask_left, -1] = 0
    position_map_right[mask_right, -1] = 0

    # use opacity value to distinguish between different areas
    binary_img_left = (position_map_left[:, :, 3] > 0.95).astype(np.uint8)
    binary_img_right = (position_map_right[:, :, 3] > 0.95).astype(np.uint8)
    contours_left, _ = cv2.findContours(binary_img_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_right, _ = cv2.findContours(binary_img_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw & save each contour mask (left, right)
    total_area = position_map.shape[0] * position_map.shape[1]

    mask_count = 0
    for i, contour in enumerate(contours_left):
        area = cv2.contourArea(contour)
        if (area / total_area < 0.0005): continue

        mask_count += 1
        mask = np.zeros_like(binary_img_left)

        mask_name = f'{segmentID}_l{mask_count}_mask.png'
        uv_name = f'{segmentID}_l{mask_count}_uv.png'

        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        cv2.imwrite(mask_name, mask)
        gen_sub_uv(mask_name, uv_name)

    mask_count = 0
    for i, contour in enumerate(contours_right):
        area = cv2.contourArea(contour)
        if (area / total_area < 0.0005): continue

        mask_count += 1
        mask = np.zeros_like(binary_img_right)

        mask_name = f'{segmentID}_r{mask_count}_mask.png'
        uv_name = f'{segmentID}_r{mask_count}_uv.png'

        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        cv2.imwrite(mask_name, mask)
        gen_sub_uv(mask_name, uv_name)

# Generate a new coordinate for each sub mask (with UV info)
def gen_sub_uv(mask_name, uv_name):
    # load sub mask
    mask = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # uv coordinate with mask
    h, w = mask.shape[:2]
    alpha = gray / 255
    u, v = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    uv = np.dstack((u, 1-v, np.ones_like(u), alpha))

    # find the contour & box region of that sub mask
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    # cv2.drawContours(mask, [box], 0, (0, 0, 255), 2)

    # crop that region with sub mask & uv info
    w, h = int(rect[1][0]), int(rect[1][1])
    src_pts = box.astype('float32')
    dst_pts = np.array([[0, h-1], [0, 0], [w-1, 0], [w-1, h-1]], dtype='float32')
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    cropped_image = cv2.warpPerspective(uv, matrix, (w, h))
    cropped_image = (cropped_image * 65535).astype(np.uint16)
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(uv_name, cropped_image)

def stretch_uv(source):
    pos = cv2.imread('20230702185753.png', cv2.IMREAD_UNCHANGED)
    pos = cv2.cvtColor(pos, cv2.COLOR_BGRA2RGBA)

    src  = cv2.imread(source, cv2.IMREAD_UNCHANGED)
    src  = cv2.cvtColor(src, cv2.COLOR_BGRA2RGBA)
    gray = (src[:, :, 3] * 255).astype(np.uint8)

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    # cv2.drawContours(src, [box], 0, (0, 0, 65535), 2)

    m = max_contour[:, 0, :]
    uvv = src[m[:, 1], m[:, 0]][:, :2] / 65535
    h, w = pos.shape[:2]
    x_max, y_max, z_max = 8096, 7888, 14370
    p  = pos[((1-uvv[:, 1]) * (h-1)).astype(int), (uvv[:, 0] * (w-1)).astype(int)][:, :3].astype(float)
    p /= 65535

    h, w = src.shape[:2]
    m = m.astype('float')
    m /= np.array([w, h])

    center_name  = 'scroll1_center.obj'
    data    = parse_obj(center_name)
    center  = data['vertices']
    center /= np.array([x_max, y_max, z_max])

    # cutting along the scroll center (opacity change along the cutting edge)
    edge_x = np.interp(p[:, 2], center[:, 2], center[:, 0])
    edge_y = np.interp(p[:, 2], center[:, 2], center[:, 1])
    mask1 = abs(p[:, 0] - edge_x) < 0.005
    mask2 = p[:, 1] - edge_y > 0
    mask3 = p[:, 1] - edge_y < 0
    m1 = np.bitwise_and(mask1, mask2)
    m2 = np.bitwise_and(mask1, mask3)
    p1 = m[m1]
    p2 = m[m2]
    avg1 = np.mean(p1, axis=0)
    avg2 = np.mean(p2, axis=0)
    p1 = p1[abs(p1[:, 0] - avg1[0]) < abs(p1[:, 0] - avg2[0])]
    p2 = p2[abs(p2[:, 0] - avg2[0]) < abs(p2[:, 0] - avg1[0])]
    if(p1[0, 1] > p1[-1, 1]): p1 = np.flip(p1, axis=0)
    if(p2[0, 1] > p2[-1, 1]): p2 = np.flip(p2, axis=0)

    h, w = src.shape[:2]
    e1_x = np.interp(np.linspace(0, 1, h), p1[:, 1], p1[:, 0])
    e2_x = np.interp(np.linspace(0, 1, h), p2[:, 1], p2[:, 0])
    if(avg1[0] > avg2[0]): e1_x = 1 - e1_x
    if(avg1[0] < avg2[0]): e2_x = 1 - e2_x

    img = np.zeros((h, w, 4), dtype=np.uint16)
    img[:, :, 0] = (e1_x[:, np.newaxis] * 65535).astype(np.uint16)
    img[:, :, 1] = (e2_x[:, np.newaxis] * 65535).astype(np.uint16)
    img[:, :, 3] = 65535

    # cv2.drawContours(src, [p1], 0, (0, 65535, 0), 2)
    # cv2.drawContours(src, [p2], 0, (0, 65535, 0), 2)

    img = img[:, :, [2, 1, 0, 3]]
    cv2.imshow('image', img)
    cv2.imwrite('20230702185753_r3_d.png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# w, h = 17381, 13513
w, h = 1738, 1351
x_max, y_max, z_max = 8096, 7888, 14370

# position map generate
# gen_pos_map(w, h, x_max, y_max, z_max)

# sub mask generate
# clustering(x_max, y_max, z_max)

stretch_uv('20230702185753_r3_uv.png')




