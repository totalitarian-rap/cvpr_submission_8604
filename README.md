# cvpr_submission_8604

## Run the code

Create an environment from the file `environment.yml`:

> conda env create -f environment.yml

Activate the environment:

> conda activate cvpr_8604

Run the code:

> python run.py --data_dir=\$DATA_DIR --log_dir=$LOG_DIR

here `\$DATA_DIR` is the path to the folder with the dataset, 
in the `\$DATA_DIR` the script will look for `images` and `masks` folders, 
i.e. the folder structure should be as follow:


```
$DATA_DIR
|   +-- images
|   |   +-- pneumonia
|   |   |   2_pa.png
|   |   |   2_lat.png
|   |   |   ...
|   |   |   547_lat.png
|   |   +-- norma
|   |   |   2_pa.png
|   |   |   2_lat.png
|   |   |   ...
|   |   |   516_pa.png
|   +-- masks
|   |   +-- expert1
|   |   |   2_pa.png
|   |   |   2_lat.png
|   |   |   ...
|   |   |   547_lat.npz
|   |   +-- expert2
|   |   |   2_pa.png
|   |   |   2_lat.png
|   |   |   ...
|   |   |   547_lat.npz
|   |   +-- expert3
|   |   |   2_pa.png
|   |   |   2_lat.png
|   |   |   ...
|   |   |   547_lat.npz
```

In the `$LOG_DIR` script will log training statistics and save checkpoints.

In `.src/indices.json' we saved indices that we used in train/validation/test sets.

## Dicom ppreprocessing

Most studies comprise two images, namely, frontal and lateral projections. There are 30 studies of lungs without pathologies and a single study of lungs affected by pneumonia with only frontal projection. See table below:

Cases|Two views|Single View|Total|
-----|---------|-----------|-----|
Norma|386|30|416
Pathological|517|1|518

In order to export dicom study to a png image we used `RescaleSlope` and `RescaleIntercept` tags of the original dicom file to do a linear transform:
$im = k*im + b,$
where $k$ is the `RescaleSlope` and $b$ is the `RescaleIntercept`; `im` is the original values of pixels stored in the `PixelArray` attribute. After the linear transformation is done we divided each image by the maximal pixel value found on this image and multiplied every pixel value by 255; finally, we saved the obtained `uint8` image with the `opencv-python`’s method `imwrite`.
Masks were saved and compressed using numpy; 1 on a mask corresponds to pneumonia and 0 corresponds to the background.
We did not make a left-right flip for the `AP` projection, the number of studies with the`AP` is not greater than 16.


