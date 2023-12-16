import numpy as np

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

def save_obj(filename, data):
    vertices = data.get('vertices', np.array([]))
    normals  = data.get('normals' , np.array([]))
    uvs      = data.get('uvs'     , np.array([]))
    faces    = data.get('faces'   , np.array([]))
    colors   = data.get('colors'  , np.array([]))

    with open(filename, 'w') as f:

        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Faces: {len(faces)}\n")

        f.write(f"mtllib test.mtl\n")
        f.write(f"usemtl default\n")

        for i in range(len(vertices)):
            vertex = vertices[i]
            normal = normals[i]
            color = colors[i] if len(colors) else ''

            f.write('v ')
            f.write(f"{' '.join(str(round(x, 6)) for x in vertex)}")
            # f.write(f"{' '.join(str(round(x, 2)) for x in vertex)}")
            f.write(' ')
            f.write(f"{' '.join(str(round(x, 6)) for x in color)}")
            f.write('\n')

            f.write('vn ')
            f.write(f"{' '.join(str(round(x, 6)) for x in normal)}")
            f.write('\n')

        for uv in uvs:
            f.write(f"vt {' '.join(str(round(x, 6)) for x in uv)}\n")

        for face in faces:
            indices = ' '.join(['/'.join(map(str, vertex)) for vertex in face])
            f.write(f"f {indices}\n")

segment_name = '20230702185753_s100_3.obj'
center_name = 'scroll1_center.obj'

z_max = 14370

segment_data = parse_obj(segment_name)
center_data = parse_obj(center_name)

interp_x = np.interp(segment_data['vertices'][:, 2], center_data['vertices'][:, 2], center_data['vertices'][:, 0])
interp_y = np.interp(segment_data['vertices'][:, 2], center_data['vertices'][:, 2], center_data['vertices'][:, 1])
interp_centers = np.column_stack((interp_x, interp_y, segment_data['vertices'][:, 2]))

angles = np.arctan2(segment_data['vertices'][:, 1] - interp_centers[:, 1], (segment_data['vertices'][:, 0] - interp_centers[:, 0]) * -1)

u_value = angles / np.pi
v_value = segment_data['vertices'][:, 2] / z_max
v_value -= 0.5
v_value *= 8

flatten_data = segment_data
flatten_data['vertices'][:, 0] = u_value
flatten_data['vertices'][:, 1] = v_value
flatten_data['vertices'][:, 2] = 0
flatten_data['normals'][:, :] = [0, 0, 1]

save_obj('flatten.obj', flatten_data)


