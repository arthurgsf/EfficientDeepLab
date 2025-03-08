import SimpleITK as _sitk
import numpy as _np

def biggest_3D_object(volume):
    volume = volume.astype(_np.int8)
    image_sitk = _sitk.GetImageFromArray(volume)
    image_sitk.SetOrigin((0, 0, 0))

    connected_filter = _sitk.ConnectedComponentImageFilter()
    connected_filter.FullyConnectedOn()
    new_image = connected_filter.Execute(image_sitk)
    stats = _sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(new_image)
    maior_label=None
    qtd_maior_label=0
    for label in stats.GetLabels():
        if(stats.GetNumberOfPixels(label)>qtd_maior_label):
            maior_label=label
            qtd_maior_label=stats.GetNumberOfPixels(label)

    new_image_array=_sitk.GetArrayFromImage(new_image)
    new_image_array[new_image_array!=maior_label]=0
    new_image_array[new_image_array==maior_label]=255
    new_image = _sitk.GetImageFromArray(new_image_array)
    new_image.CopyInformation(image_sitk)
    return _sitk.GetArrayFromImage(new_image).astype(_np.float32)

def get_boundaries(volume):
    ix = _np.argwhere(volume == 1)
    return _np.min(ix), _np.max(ix)