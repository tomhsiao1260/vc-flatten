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
        d_name = f'{segmentID}_l{mask_count}_d.png'

        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        cv2.imwrite(mask_name, mask)
        gen_sub_uv(mask_name, uv_name, position_map)
        gen_sub_d(uv_name, d_name, position_map)
        # if (mask_count == 1): gen_sub_d(uv_name, d_name, position_map)

    mask_count = 0
    for i, contour in enumerate(contours_right):
        area = cv2.contourArea(contour)
        if (area / total_area < 0.0005): continue

        mask_count += 1
        mask = np.zeros_like(binary_img_right)

        mask_name = f'{segmentID}_r{mask_count}_mask.png'
        uv_name = f'{segmentID}_r{mask_count}_uv.png'
        d_name = f'{segmentID}_r{mask_count}_d.png'

        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        cv2.imwrite(mask_name, mask)
        gen_sub_uv(mask_name, uv_name, position_map)
        gen_sub_d(uv_name, d_name, position_map)

# Generate a new coordinate for each sub mask (with UV info)
def gen_sub_uv(mask_name, uv_name, position_map):
    # load sub mask
    mask = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED)

    # uv coordinate with mask
    h, w = mask.shape[:2]
    alpha = mask / 255
    u, v = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    uv = np.dstack((u, 1-v, np.ones_like(u), alpha))

    # find the contour & box region of that sub mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    # cv2.drawContours(mask, [box], 0, (0, 0, 255), 2)

    # box index sorting (make larger scroll z position is always at the top of the image)
    center, size, angle = rect
    center = np.array(center)
    half_box = ((box + center) / 2).astype(int)
    z_pos = [position_map[y, x, 2] for x, y in half_box]
    sorted_ind = sorted(range(len(z_pos)), key=lambda k: z_pos[k])
    sorted_box = box[sorted_ind]

    db = np.cross(sorted_box[1] - sorted_box[0], sorted_box[0] - center)
    dt = np.cross(sorted_box[3] - sorted_box[2], sorted_box[2] - center)
    if (db < 0): sorted_ind[1], sorted_ind[0] = sorted_ind[0], sorted_ind[1]
    if (dt < 0): sorted_ind[3], sorted_ind[2] = sorted_ind[2], sorted_ind[3]
    sorted_box = box[sorted_ind]

    # crop that region with sub mask & uv info
    w, h = int(size[0]), int(size[1])
    src_pts = sorted_box.astype('float32')
    dst_pts = np.array([[0, h-1], [w-1, h-1], [w-1, 0], [0, 0]], dtype='float32')
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    cropped_image = cv2.warpPerspective(uv, matrix, (w, h))
    cropped_image = (cropped_image * 65535).astype(np.uint16)
    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(uv_name, cropped_image)

def gen_sub_d(uv_name, d_name, position_map):
    src  = cv2.imread(uv_name, cv2.IMREAD_UNCHANGED)
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
    h, w = position_map.shape[:2]
    p  = position_map[((1-uvv[:, 1]) * (h-1)).astype(int), (uvv[:, 0] * (w-1)).astype(int)][:, :3].astype(float)
    h, w = src.shape[:2]

    gap = 0.01
    bins = np.arange(0, 1+gap, gap)
    weights = np.power(m[:, 1] / h - 0.5, 2) * 4
    hist, edges = np.histogram(p[:, 2], bins=bins, weights=weights)
    peak_regions = np.argsort(hist)[-2:]

    if(abs(peak_regions[-1] - peak_regions[-2]) == 1):
        peak_regions[-2] = np.argsort(hist)[-3]

    corner_list = []

    for region in peak_regions:
        indices_in_region = np.where((p[:, 2] >= edges[region]) & (p[:, 2] <= edges[region + 1]))[0]

        if indices_in_region.size > 0:
            inside = False
            confident = 0
            threshold = 10
            avg_value = np.mean(p[indices_in_region, 2])

            for i in range(len(p[:, 2])):
                current_state = np.abs(p[i, 2] - avg_value) <= 0.03

                if(current_state != inside): confident += 1
                if(confident == threshold):
                    inside = not inside
                    confident = 0
                    index = i - threshold + 1
                    if (index < 10): continue
                    corner_list.append(index)

    x_ind = sorted(corner_list, key=lambda i: m[i, 0])

    tl = min(x_ind[:2], key=lambda i: m[i, 1])
    bl = max(x_ind[:2], key=lambda i: m[i, 1])
    tr = min(x_ind[2:], key=lambda i: m[i, 1])
    br = max(x_ind[2:], key=lambda i: m[i, 1])

    h, w = src.shape[:2]
    m = m.astype('float')
    m /= np.array([w, h])
    mt = np.concatenate((m[tl:0:-1], m[-1:tr:-1]), axis=0)

    dl = np.interp(np.linspace(0, 1, h), m[tl:bl, 1], m[tl:bl, 0])
    dr = np.interp(np.linspace(0, 1, h), m[tr:br:-1, 1], m[tr:br:-1, 0])
    db = np.interp(np.linspace(0, 1, w), m[bl:br, 0], m[bl:br, 1])
    dt = np.interp(np.linspace(0, 1, w), mt[:, 0], mt[:, 1])

    dr = 1 - dr
    db = 1 - db

    img = np.zeros((h, w, 4), dtype=np.uint16)
    img[:, :, 0] = (dl[:, np.newaxis] * 65535).astype(np.uint16)
    img[:, :, 1] = (dr[:, np.newaxis] * 65535).astype(np.uint16)
    img[:, :, 2] = (dt[np.newaxis, :] * 65535).astype(np.uint16)
    img[:, :, 3] = (db[np.newaxis, :] * 65535).astype(np.uint16)
    img = img[:, :, [2, 1, 0, 3]]
    cv2.imwrite(d_name, img)

    # cv2.drawContours(src, [max_contour[:, 0, :][tl:bl]], 0, (0, 0, 65535), 2)
    # cv2.drawContours(src, [max_contour[:, 0, :][br:tr]], 0, (0, 65535, 0), 2)
    # cv2.imshow('image', src)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# w, h = 17381, 13513
w, h = 1738, 1351
x_max, y_max, z_max = 8096, 7888, 14370

# position map generate
# gen_pos_map(w, h, x_max, y_max, z_max)

# sub mask & sub distance map generate
clustering(x_max, y_max, z_max)


# filename = '20230702185753_r3_d.png'
# img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
# img = (img / 65535 * 255).astype(np.uint8)
# img = img[:, :, [0, 2, 1, 3]]
# cv2.imwrite('ok.png', img)



