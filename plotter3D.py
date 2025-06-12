import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
#from matplotlib.pyplot import gca
#from mpl_toolkits.mplot3d import Axes3D
#from stl import mesh
#from stltovoxel import convert_meshes

from src import plotter
from src.preprocessing import normalize


def show_histogram(values):
    n, bins, patches = plt.hist(values.reshape(-1), 50, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for c, p in zip(normalize(bin_centers), patches):
        plt.setp(p, 'facecolor', cm.viridis(c))

    plt.show()

def scale_by(arr, fac):
    mean = np.mean(arr)
    return (arr - mean) * fac + mean

def make_ax(grid=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    return ax

def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3]*2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded

def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z

"""def plot_cube(cube, img_dim, angle=320):
    cube = normalize(cube)

    facecolors = cm.viridis(cube)
    facecolors[:, :, :, -1] = cube
    facecolors = explode(facecolors)

    filled = facecolors[:, :, :, -1] != 0
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

    fig = plt.figure(figsize=(30 / 2.54, 30 / 2.54))
    print(1)
    ax = Axes3D(fig)
    ax.view_init(30, angle)
    ax.set_xlim(right=img_dim * 2)
    ax.set_ylim(top=img_dim * 2)
    ax.set_zlim(top=img_dim * 2)
    print(2)

    ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)
    print(3)
    plt.show()
    print(4)"""


def plot_cube(cube, img_dim, angle=320):
    cube = normalize(cube)

    facecolors = cm.viridis(cube)
    facecolors[:, :, :, -1] = cube
    facecolors = explode(facecolors)

    filled = facecolors[:, :, :, -1] != 0
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection='3d')
    ax.view_init(30, angle)
    ax.set_xlim(right=img_dim * 2)
    ax.set_ylim(top=img_dim * 2)
    ax.set_zlim(top=img_dim * 2)

    ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)
    plt.show()


def voxelization(input_file_path, colors=[(255, 255, 255)], resolution=100, voxel_size=None, pad=1, parallel=False):
    meshes = []
    # for input_file_path in input_file_paths:
    # print(input_file_path)
    mesh_obj = mesh.Mesh.from_file(input_file_path)
    org_mesh = np.hstack((mesh_obj.v0[:, np.newaxis], mesh_obj.v1[:, np.newaxis], mesh_obj.v2[:, np.newaxis]))
    meshes.append(org_mesh)

    vol, scale, shift = convert_meshes(meshes, resolution, voxel_size, parallel)
    # output_file_pattern, output_file_extension = os.path.splitext(output_file_path)
    # if output_file_extension == '.npy':
    # export_npy(vol, output_file_path, scale, shift)
    voxels = vol.astype(bool)
    """print(voxels.shape)
    out = []
    for z in range(voxels.shape[0]):
        for y in range(voxels.shape[1]):
            for x in range(voxels.shape[2]):
                if voxels[z][y][x]:
                    point = (np.array([x, y, z]) / scale) + shift
                    out.append(point)"""
    return voxels

class IndexTracker:
    def __init__(self, ax, X, slice_ax):
        self.ax = ax
        self.slice_ax = slice_ax
        #ax.set_title('use scroll wheel to navigate images')

        self.X = X
        shape = X.shape
        self.slices = shape[slice_ax]
        self.ind = self.slices // 2
        self.im = ax.imshow(self.get_slice(), vmin=0, vmax=1)#[:, :, self.ind])
        self.update()

    def on_scroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def get_slice(self):
        return self.X.take(self.ind, self.slice_ax)

    def update(self):
        self.im.set_data(self.get_slice())
        self.ax.set_xlabel('slice %s' % self.ind)
        #self.im.axes.figure.canvas.draw()
        self.im.axes.figure.canvas.draw_idle()
        #self.im.axes.figure.canvas.blit(self.ax.bbox)

def plotVolumetricSlices(img, mask_list, axis_name, img_mean_projection=False, mask_mean_projection=False,
                         merge_masks=False, figure_title=f"Centered Volumetric Slices"):
    img_sh = img.shape
    figure_rows = 1 + len(mask_list)
    #msk_sh = []
    mask_slices = []

    # Slice VMean Projection of each view
    if mask_mean_projection:
        for mask in mask_list:
            les_vol_0 = np.mean(mask, axis=0)
            les_vol_1 = np.mean(mask, axis=1)
            les_vol_2 = np.mean(mask, axis=2)
            if (np.max(mask)>0):
                les_vol_0 /= np.max(les_vol_0)
                les_vol_1 /= np.max(les_vol_1)
                les_vol_2 /= np.max(les_vol_2)
            mask_slices.append([les_vol_0, les_vol_1, les_vol_2])
    else:
        for mask in mask_list:
            msk_sh = mask.shape
            mask_slices.append([mask[msk_sh[0] // 2, :, :], mask[:, msk_sh[1] // 2, :], mask[:, :, msk_sh[2] // 2]])


    if img_mean_projection:
        img_vol_0 = np.mean(img, axis=0)
        img_vol_1 = np.mean(img, axis=1)
        img_vol_2 = np.mean(img, axis=2)
        img_vol_0 /= np.max(img_vol_0)
        img_vol_1 /= np.max(img_vol_1)
        img_vol_2 /= np.max(img_vol_2)
        img_slices = [img_vol_0, img_vol_1, img_vol_2]
    else:
        img_slices = [img[img_sh[0] // 2, :, :], img[:, img_sh[1] // 2, :], img[:, :, img_sh[2] // 2]]


    if merge_masks:
        mask_slices = np.array(mask_slices)
        mask_slices = np.sum(mask_slices, axis=0)
        for i, slice in enumerate(mask_slices):
            mask_slices[i] = np.clip(slice, 0, 1)
        mask_slices_ravel = mask_slices
        figure_rows = 2
    else:
        mask_slices_ravel = []
        for mask in mask_slices:
            mask_slices_ravel = mask_slices_ravel + mask

    if mask_mean_projection:
        figure_title += "\n(Le maschere sono mostrate come Proiezioni di Intensità Media)"
    plotter.dinamicFigurePlot(figure_title,
                              [f"{axis_name[0]} (#{img_sh[0] // 2})",
                               f"{axis_name[1]} (#{img_sh[1] // 2})",
                               f"{axis_name[2]} (#{img_sh[2] // 2})"],
                              [*img_slices,
                               *mask_slices_ravel],
                              shape=(figure_rows, 3), cmaps=["gray", "gray", "gray", "magma","magma","magma","magma","magma","magma",])

def plotInteractive():
    return 0
