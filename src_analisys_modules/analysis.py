import time

from plotter3D import *

from src import plotter

from skimage.measure import label
from skimage import measure
from skimage.transform import warp_polar

from src_analisys_modules.preprocessing import project_circumference, mass_ends_clock_position


def generate_color_palette(palette_name, step):
    main_color = np.linspace(1, 0.75, num=step)
    gradient = np.linspace(0, 0.75, num=step)
    palette = []
    s = 0.5
    for i in range(0,step):
        if palette_name == "reds":
            palette.append((main_color[i],gradient[i]*s,gradient[i]))
        if palette_name == "greens":
            palette.append((gradient[i], main_color[i], gradient[i]*s))
        if palette_name == "blues":
            palette.append((gradient[i]*s,gradient[i],main_color[i]))

    return palette


def compute_volume(mask_shape, dicom_shape, dicom_spacing, voxels_count):
    # Get scaling between original dicom shape and prediction mask shape
    scaling = np.array(dicom_shape) / np.array(mask_shape)

    # Get the metrical sampling of a voxel (Height, Width, Slices)
    cube_metrical_sampling = scaling * dicom_spacing

    # Get the metrical volume
    volume = np.prod(cube_metrical_sampling) * voxels_count

    return round(volume,2)

def compute_max_diameter(binary_mask, slice_axis, sampling):
    # Itera su ogni slice 2D
    diameters = []
    for j in range(binary_mask.shape[slice_axis]):
        binary_slice = np.take(binary_mask, j, axis=slice_axis)
        if np.count_nonzero(binary_slice) > 0:
            region_props = measure.regionprops(binary_slice.astype(np.uint8))
            # Calcola il diametro maggiore dell'oggetto
            diameters.append(max([r.major_axis_length for r in region_props]))

    # Prendi il massimo dei diametri calcolati
    diameters = max(diameters) * sampling
    return round(diameters,2)


def neoplasia_analysis(mask, dicom_params, verbose=False, plot_circumference=False):
    lesion_infos = []

    start = time.time()
    if verbose: print(f"Start Neoplasia Segmentation Analysis")

    # Verifica l'esistenza di oggetti nella predizione
    neoplasia_exist = (np.count_nonzero(mask) > 0)
    # Eseguiamo il connected component labeling sulla maschera
    mask_label, num_lesion = label(mask[:, :, :, 0], return_num=True)
    mask_label_rgb = np.zeros((*(mask_label.shape),3),dtype=mask.dtype)

    palette = generate_color_palette("reds", num_lesion)

    if verbose:
        print(f"\tNeoplasia Exist: {neoplasia_exist}")
        print(f"\tNumber of lesion masses: {num_lesion}")

    for n in range(1, num_lesion + 1):
        lesion_n = (mask_label == n).astype(mask.dtype)
        #print(lesion_n.dtype)

        curr_palette = palette[n - 1]
        curr_palette -= np.min(curr_palette)
        curr_palette /= np.max(curr_palette)
        # lesion_n = lesion_n * (palette[n - 1][0])
        #plt.imshow(lesion_n[:, :, 10])
        #plt.show()
        # lesion_n_rgb = np.repeat(lesion_n_rgb, 3, axis=-1)
        mask_label_rgb[:, :, :, 0] += lesion_n * curr_palette[0]
        mask_label_rgb[:, :, :, 1] += lesion_n * curr_palette[1]
        mask_label_rgb[:, :, :, 2] += lesion_n * curr_palette[2]
        #print(np.max(mask_label_rgb))

        #plt.imshow(mask_label_rgb[:, :, 10, :])
        #plt.show()

        # Get number of positive voxels
        object_voxels_count = np.count_nonzero(lesion_n)

        # Compute Volume
        volume = compute_volume(mask.shape[0:3], dicom_params["shape"],
                                dicom_params["spacing"], object_voxels_count)

        # Compute Circumference Projection (with Polar Projection method)
        lesion_slice_projection,lesion_polar_projection,lesion_circm_projection = project_circumference(lesion_n)

        # Compute Circumference involved as coverage percentage
        circumference_involved = (np.count_nonzero(lesion_circm_projection) / lesion_circm_projection.shape[0]) * 100
        quadrants = np.split(lesion_circm_projection, 4, axis=0)
        quadrants_involved = [0, 0, 0, 0]
        for q, quadrant in enumerate(quadrants):
            if sum(quadrant) > 0:
                quadrants_involved[q] = round(((sum(quadrant) / len(quadrant)) * 100)[0], 2)
            q += 1

        # Compute Circumference involved as degrees/hours limits
        lesion_start, lesion_stop = mass_ends_clock_position(lesion_circm_projection,
                                                             circumference_involved=circumference_involved)

        lesion_infos.append({"label": n,
                             "volume": volume,
                             "circumference": circumference_involved,
                             "quadrants": quadrants_involved,
                             "clock_position_start": lesion_start,
                             "clock_position_end": lesion_stop})

        if verbose:
            print(f"\t\tLesion #{n}:")
            print(f"\t\t\tVolume: {volume} mm^3")
            print(f"\t\t\tCircumference Involved: {circumference_involved} %")
            print(f"\t\t\tQuadrants Involved: {quadrants_involved} % (start from bottom-right quadrant)")
            print(f"\t\t\tLesion Extension: From h{lesion_start} to h{lesion_stop}")
            print(f"Elapsed Time - Neoplasia Analisis : {time.time() - start} seconds")

        if plot_circumference:
            plotter.dinamicFigurePlot(f"Polar Coordinates Transformation "
                                      f"| Circumference Involved {circumference_involved} "
                                      f"| Quadrants Involved: {quadrants_involved}",
                                      ["Cartesian", "Polar", "Circumference"],
                                      [lesion_slice_projection, lesion_polar_projection, lesion_circm_projection],
                                      shape=(1, 3))

    return mask_label, lesion_infos, mask_label_rgb


def lymphnode_analysis(mask, dicom_params, verbose=False):
    lymphnode_infos = []

    start = time.time()
    if verbose: print(f"Start Lymphnode Segmentation Analysis")

    # Verifica l'esistenza di oggetti nella predizione
    lymphnode_exist = (np.count_nonzero(mask) > 0)

    # Eseguiamo il connected component labeling sulla maschera
    mask_nod_label, num_lymphnode = label(mask[:, :, :, 0], return_num=True)
    #mask_nod_label_rgb = label2rgb(mask_nod_label, bg_label=0)
    mask_nod_label_rgb = np.zeros((*(mask_nod_label.shape), 3), dtype=mask.dtype)

    if verbose:
        print(f"\tLymphNode Exist: {lymphnode_exist}")
        print(f"\tNumber of lymph node: {num_lymphnode}")

    # Ora puoi utilizzare i numeri di etichetta per isolare gli oggetti singoli
    # Ad esempio, per ottenere l'immagine di un singolo oggetto con etichetta "n"

    #palette = seaborn.color_palette("bright", num_lymphnode)
    palette_sus = generate_color_palette("blues", num_lymphnode)
    palette = generate_color_palette("greens", num_lymphnode)
    suspicious_lymphs = 0
    healty_lymphs = 0

    for n in range(1, num_lymphnode + 1):
        lymphnode_n = (mask_nod_label == n).astype(mask.dtype)

        object_voxels_count = np.count_nonzero(lymphnode_n)
        volume = compute_volume(mask.shape[0:3], dicom_params["shape"],
                                dicom_params["spacing"], object_voxels_count)

        # Get axis scaling between original dicom shape and prediction mask shape
        scaling = np.array(dicom_params["shape"]) / np.array(mask.shape[0:3])
        # Get the metric sampling of a voxel (Height, Width, Slices)
        cube_metric_sampling = scaling * dicom_params["spacing"]

        diameter_view_axial = compute_max_diameter(lymphnode_n,0 ,cube_metric_sampling[0])
        diameter_view_coron = compute_max_diameter(lymphnode_n,1 ,cube_metric_sampling[1])
        diameter_view_sagit = compute_max_diameter(lymphnode_n,2 ,cube_metric_sampling[2])
        diameters = [diameter_view_axial, diameter_view_coron, diameter_view_sagit]

        # Compute Circumference involved as degrees/hours limits
        _, _, lymphnode_circm_projection = project_circumference(lymphnode_n)
        print(f"projection count: {np.count_nonzero(lymphnode_circm_projection)}")
        mass_start, mass_stop = mass_ends_clock_position(lymphnode_circm_projection)
        mass_centroid = round((mass_start+mass_stop)/2,1)

        # Draw on RGB mask
        if np.max(diameters) > 5:
            curr_palette = palette_sus[suspicious_lymphs]
            suspicious_lymphs  += 1
        else:
            curr_palette = palette[healty_lymphs]
            healty_lymphs += 1
        curr_palette -= np.min(curr_palette)
        curr_palette /= np.max(curr_palette)
        # lesion_n_rgb = np.repeat(lesion_n_rgb, 3, axis=-1)
        mask_nod_label_rgb[..., 0] += lymphnode_n * curr_palette[0]
        mask_nod_label_rgb[..., 1] += lymphnode_n * curr_palette[1]
        mask_nod_label_rgb[..., 2] += lymphnode_n * curr_palette[2]

        if verbose:
            print(f"\t\tLymphNode #{n}:")
            print(f"\t\t\tVolume: {object_voxels_count} mm^3")
            print(f"\t\t\tMass Position: {mass_centroid} h")
            print(f"\t\t\tDiameter (View: Axial):    {diameter_view_axial} mm")
            print(f"\t\t\tDiameter (View: Coronal):  {diameter_view_coron} mm")
            print(f"\t\t\tDiameter (View: Sagittal): {diameter_view_sagit} mm")

        lymphnode_infos.append({"label": n,
                                "volume": volume,
                                "clock_position": mass_centroid,
                                "diameter_axial": diameter_view_axial,
                                "diameter_coronal": diameter_view_coron,
                                "diameter_sagittal": diameter_view_sagit})

    if verbose: print(f"Elapsed Time - Lymphnodes Analisis : {time.time() - start}")

    return mask_nod_label, lymphnode_infos, mask_nod_label_rgb
