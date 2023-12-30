import os
import cv2
import json
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

# Generate position & normal map to store vertices info of a 3D model
def gen_pos_normal(segmentID, x_max, y_max, z_max):
    prefix          = f'../full-scrolls/Scroll1.volpkg/paths/{segmentID}/'
    obj_name        = f'{segmentID}.obj'
    mask_name       = f'{segmentID}_mask.png'
    pos_map_name    = f'{segmentID}_positions.png'
    normal_map_name = f'{segmentID}_normals.png'

    # load 3D geometry & mask image
    data = parse_obj(os.path.join(prefix, obj_name))
    mask_img = cv2.imread(os.path.join(prefix, mask_name), cv2.IMREAD_UNCHANGED)

    # vertices point normalize & add a channel for opacity (p, n: rgba 0~1)
    p = data['vertices']
    p = p / np.array([x_max, y_max, z_max])
    p = np.concatenate([p, np.ones((p.shape[0], 1))], axis=-1)
    n = data['normals']
    n = (n + 1) / 2
    n = np.concatenate([n, np.ones((n.shape[0], 1))], axis=-1)

    # fit uv points into the grid structure (w, h)
    h, w = mask_img.shape
    h, w = (h // 10, w // 10)
    u = (data['uvs'][:, 0] * w).astype(int)
    v = (data['uvs'][:, 1] * h).astype(int)
    v = h - v
    uv = np.column_stack((u, v))
    grid_u, grid_v = np.meshgrid(np.arange(0, w), np.arange(0, h))

    # generate the position & normal map (w, h, rgba)
    position_map = griddata(uv, p, (grid_u, grid_v), method='nearest', fill_value=0)
    normal_map   = griddata(uv, n, (grid_u, grid_v), method='nearest', fill_value=0)
    position_map = np.clip(position_map, 0, 1)
    normal_map   = np.clip(normal_map, 0, 1)

    # remove the part outside the mask (opacity 0)
    mask_img = mask_img.astype(np.float32) / 255
    mask_img = cv2.resize(mask_img, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
    mask = mask_img < 0.01
    position_map[mask, -1] = 0
    normal_map[mask, -1] = 0

    # save map with unit16 resolution (also rgba -> bgra for opencv)
    position_map = (position_map * 65535).astype(np.uint16)
    position_map = cv2.cvtColor(position_map, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(os.path.join(segmentID, pos_map_name), position_map)
    normal_map = (normal_map * 65535).astype(np.uint16)
    normal_map = cv2.cvtColor(normal_map, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(os.path.join(segmentID, normal_map_name), normal_map)

# Generate uv map for each sub segment
def gen_sub_uv(segmentID, x_max, y_max, z_max):
    center_name  = 'scroll1_center.obj'
    pos_map_name = f'{segmentID}_positions.png'

    # load scroll center into (used for cutting)
    data    = parse_obj(center_name)
    scroll_center  = data['vertices']
    scroll_center /= np.array([x_max, y_max, z_max])

    # load position map (rgba 0~1)
    position_map = cv2.imread(os.path.join(segmentID, pos_map_name), cv2.IMREAD_UNCHANGED)
    position_map = cv2.cvtColor(position_map, cv2.COLOR_BGRA2RGBA)
    position_map = position_map.astype(np.float32) / 65535

    # uv map
    h, w = position_map.shape[:2]
    alpha = position_map[:, :, 3]
    u, v = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    uv_map = np.dstack((u, 1-v, np.ones_like(u), alpha))

    # cutting along the scroll center (mask left & right)
    edge_x = np.interp(position_map[:, :, 2], scroll_center[:, 2], scroll_center[:, 0])
    mask_left = position_map[:, :, 0] - edge_x > 0
    mask_right = position_map[:, :, 0] - edge_x <= 0

    # find min rectangle box
    gray = (position_map[:, :, 3] * 255).astype(np.uint8)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

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

    # generate cropped uv map
    center, size, angle = rect
    wb, hb = int(size[0]), int(size[1])
    src_pts = sorted_box.astype('float32')
    dst_pts = np.array([[0, hb-1], [wb-1, hb-1], [wb-1, 0], [0, 0]], dtype='float32')
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    uv_map_left = np.copy(uv_map)
    uv_map_right = np.copy(uv_map)
    uv_map_left[mask_left, -1] = 0
    uv_map_right[mask_right, -1] = 0
    uv_map_left = cv2.warpPerspective(uv_map_left, matrix, (wb, hb))
    uv_map_right = cv2.warpPerspective(uv_map_right, matrix, (wb, hb))

    # find its contours
    binary_img_left = np.where(uv_map_left[:, :, 3] > 0.95, 255, 0).astype(np.uint8)
    binary_img_right = np.where(uv_map_right[:, :, 3] > 0.95, 255, 0).astype(np.uint8)
    contours_left, _ = cv2.findContours(binary_img_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_right, _ = cv2.findContours(binary_img_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # save each contour uv map (left, right)
    total_area = position_map.shape[0] * position_map.shape[1]

    uv_map_left = (uv_map_left * 65535).astype(np.uint16)
    uv_map_right = (uv_map_right * 65535).astype(np.uint16)
    uv_map_left = cv2.cvtColor(uv_map_left, cv2.COLOR_RGBA2BGRA)
    uv_map_right = cv2.cvtColor(uv_map_right, cv2.COLOR_RGBA2BGRA)

    contours_left = [(c, 'l') for c in contours_left]
    contours_right = [(c, 'r') for c in contours_right]
    contours = contours_left + contours_right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c[0])[0], reverse=True)

    mask_count = 0
    chunks = []
    for i, (contour, label) in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area / total_area < 0.0005): continue

        mask_count += 1
        uv_name = os.path.join(segmentID, f'{segmentID}_{label}{mask_count}_uv.png')
        d_name = os.path.join(segmentID, f'{segmentID}_{label}{mask_count}_d.png')

        uv_map = uv_map_left if (label == 'l') else uv_map_right
        mask = np.zeros_like(uv_map[:, :, 3]).astype(np.uint16)
        cv2.drawContours(mask, [contour], -1, 65535, thickness=cv2.FILLED)
        uv_map[:, :, 3] = mask
        [x, y, w, h] = cv2.boundingRect(contour)
        cv2.imwrite(uv_name, uv_map[:, x:(x+w), :])

        w, h, l, r = gen_sub_d(uv_name, d_name, position_map)
        # if (mask_count == 1): gen_sub_d(uv_name, d_name, position_map)

        # save meta info
        chunk = {}
        chunk['id'] = mask_count
        chunk['uv'] = f'{segmentID}_{label}{mask_count}_uv.png'
        chunk['d'] = f'{segmentID}_{label}{mask_count}_d.png'
        chunk['width'] = int(w)
        chunk['height'] = int(h)
        chunk['l'] = int(l)
        chunk['r'] = int(r)
        chunks.append(chunk)

    return chunks

# Generate distance map for each sub segment
def gen_sub_d(uv_name, d_name, position_map):
    # load uv map & find contour
    uv_map = cv2.imread(uv_name, cv2.IMREAD_UNCHANGED)
    uv_map = cv2.cvtColor(uv_map, cv2.COLOR_BGRA2RGBA)
    uv_map = uv_map.astype(np.float32) / 65535
    h, w   = uv_map.shape[:2]

    gray = (uv_map[:, :, 3] * 255).astype(np.uint8)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)[:, 0, :]

    # find 3d point positions on that contour
    hp, wp = position_map.shape[:2]
    uv = uv_map[contour[:, 1], contour[:, 0], :2]
    uv[:, 0] = (wp - 1) * uv[:, 0]
    uv[:, 1] = (hp - 1) * (1 - uv[:, 1])
    uv = uv.astype(int)
    p = position_map[uv[:, 1], uv[:, 0], :3]

    # use histogram to find top & bottom edge (with the same z value)
    gap = 0.01
    bins = np.arange(0, 1+gap, gap)
    weights = np.power(contour[:, 1] / h - 0.5, 2) * 4
    hist, edges = np.histogram(p[:, 2], bins=bins, weights=weights)
    peaks = np.argsort(hist)[-2:]
    if(abs(peaks[-1] - peaks[-2]) == 1): peaks[-2] = np.argsort(hist)[-3]

    # save 4 corner indices
    corner_list = []
    for peak in peaks:
        z_lower = edges[peak]
        z_upper = edges[peak + 1]
        peak_ind = np.where((p[:, 2] > z_lower) & (p[:, 2] < z_upper))[0]

        if peak_ind.size > 0:
            inside = False
            confident = 0
            threshold = 10
            avg_value = np.mean(p[peak_ind, 2])

            for i in range(len(p[:, 2])):
                current_state = np.abs(p[i, 2] - avg_value) <= 0.005

                if(current_state != inside): confident += 1
                if(current_state == inside): confident = max(confident - 1, 0)
                if(confident == threshold):
                    confident = 0
                    inside = not inside
                    index = i - threshold + 1
                    if (index < 5): continue
                    corner_list.append(index)

    if(len(corner_list) < 4): corner_list.append(0)
    y_ind = sorted(corner_list, key=lambda i: contour[i, 1])

    # tl = min(y_ind[:2], key=lambda i: contour[i, 0])
    # tr = max(y_ind[:2], key=lambda i: contour[i, 0])
    # bl = min(y_ind[2:], key=lambda i: contour[i, 0])
    # br = max(y_ind[2:], key=lambda i: contour[i, 0])

    tl = sorted(corner_list)[0]
    bl = sorted(corner_list)[1]
    br = sorted(corner_list)[2]
    tr = sorted(corner_list)[3]

    h, w = uv_map.shape[:2]
    contour = contour.astype('float')
    contour /= np.array([w, h])
    mt = np.concatenate((contour[tl:0:-1], contour[-1:tr:-1]), axis=0)

    dl = np.interp(np.linspace(0, 1, h), contour[tl:bl, 1], contour[tl:bl, 0])
    dr = np.interp(np.linspace(0, 1, h), contour[tr:br:-1, 1], contour[tr:br:-1, 0])
    db = np.interp(np.linspace(0, 1, w), contour[bl:br, 0], contour[bl:br, 1])
    dt = np.interp(np.linspace(0, 1, w), mt[:, 0], mt[:, 1])

    dr = 1 - dr
    db = 1 - db

    img = np.zeros((h, w, 4), dtype=np.uint16)
    img[:, :, 0] = (dl[:, np.newaxis] * 65535).astype(np.uint16)
    img[:, :, 1] = (dr[:, np.newaxis] * 65535).astype(np.uint16)
    img[:, :, 2] = (dt[np.newaxis, :] * 65535).astype(np.uint16)
    img[:, :, 3] = (db[np.newaxis, :] * 65535).astype(np.uint16)
    img = 65535 - img
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(d_name, img)

    l_line = np.column_stack((w * dl, h * np.linspace(0, 1, h))).astype(np.int32)
    r_line = np.column_stack((w * (1 - dr), h * np.linspace(0, 1, h))).astype(np.int32)
    t_line = np.column_stack((w * np.linspace(0, 1, w), h * dt)).astype(np.int32)
    b_line = np.column_stack((w * np.linspace(0, 1, w), h * (1- db))).astype(np.int32)

    l = np.mean(l_line, axis=0)[0].astype(int)
    r = np.mean(r_line, axis=0)[0].astype(int)
    r = w - r

    cv2.polylines(img, [l_line], isClosed=False, color=(0, 0, 65535), thickness=2)
    cv2.polylines(img, [r_line], isClosed=False, color=(0, 65535, 0), thickness=2)
    cv2.polylines(img, [t_line], isClosed=False, color=(65535, 0, 0), thickness=2)
    cv2.polylines(img, [b_line], isClosed=False, color=(0, 0, 0), thickness=2)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return w, h, l, r

def save_meta(segmentID, chunks):
    info = {}
    info['id'] = segmentID
    info['labels'] = ''
    info['positions'] = f'{segmentID}_positions.png'
    info['normals'] = f'{segmentID}_normals.png'
    info['chunks'] = chunks

    meta = {}
    meta['segment'] = [ info ]
    with open('meta.json', "w") as outfile: json.dump(meta, outfile, indent=4)

x_max, y_max, z_max = 8096, 7888, 14370

# segmentID = '20230702185753'
segmentID = '20231031143852'
if not os.path.exists(segmentID): os.makedirs(segmentID)

# position & normal map generate
# gen_pos_normal(segmentID, x_max, y_max, z_max)

# sub uv & distance map generate
chunks = gen_sub_uv(segmentID, x_max, y_max, z_max)

# save meta info as json
save_meta(segmentID, chunks)

