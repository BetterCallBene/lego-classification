# # %%
# !python --version

# # %%
# !pip install --upgrade pip

# %%
# !pip install numpy --quiet
# !pip install pandas --quiet
# !pip install fiftyone --quiet
# !pip install beautifulsoup4 --quiet
# !pip install fastai --quiet

# %%
import glob
import shutil
import PIL
import numpy as np
import os
import fiftyone as fo
from bs4 import BeautifulSoup
from fastai.vision.all import *

# %%
dataset_name = "lego-classification"
current_dir = os.getcwd()
data_dir = os.path.join(current_dir, 'data')
dataset_dir = os.path.join(data_dir, "dataset_20210629145407_top_600")
destination_image_dir = os.path.join(data_dir, 'images')

# %% [markdown]
# 

# %%
annotations_dir = os.path.join(dataset_dir, 'annotations')
images_dir = os.path.join(dataset_dir, 'images')
annotations_patt = os.path.join(annotations_dir, '*')



# %%
def read_xml_file(annotation_path):
    
    with open(annotation_path, 'r') as annotation_file:
        annotation_string = annotation_file.read()
    soup = BeautifulSoup(annotation_string)

    
    height = None
    width = None
    elms = []

    try:

        height = np.float32(soup.annotation.size.height)
        width =  np.float32(soup.annotation.size.width)

            
        for x in soup.find_all('object'):
            name = x.findChild('name').contents[0]
            
            xmin = np.float32(x.bndbox.xmin.contents[0])
            xmax = np.float32(x.bndbox.xmax.contents[0])
            ymin = np.float32(x.bndbox.ymin.contents[0])
            ymax = np.float32(x.bndbox.ymax.contents[0])
            elm = dict()
            elm['bbox'] = [xmin/width, ymin/height, (xmax - xmin)/width, (ymax - ymin)/height]
            elm['label'] = name
            elms.append(elm)

        image_path = os.path.join(images_dir, soup.annotation.filename.contents[0])
        
    except Exception:
        print("File: " + annotation_path + " can not parse.")
        return False, None, None

    if not os.path.exists(image_path):
        print("Jpeg file '"+ image_path + "'do not exists.")
        return False, None, None
        
    try:
        img = PIL.Image.open(image_path)
        width_from_img, height_from_img = img.size 

        if width_from_img != width or height_from_img != height:
            print("annotation info (width={0}, height={1}) != image info (width={2}, height={3})".format(width, height, width_from_img, height_from_img))
            return False, None, None
    except Exception:
        print("File '" + image_path + "' could not be open as image")
    
    shutil.copy(image_path, destination_image_dir)

    return True, image_path, elms

    


# %%
files = glob.glob(annotations_patt)

samples = []
for index, annotation_file in zip(range(0, len(files)), files):
    #print("Parse File: {0} from {1}".format(index + 1, len(files)))
    result, image_path, elms = read_xml_file(annotation_file)
    if not result:
        print("File was skiped.")
        continue

    sample = fo.Sample(filepath=image_path)
    

    detections = []

    for elm in elms:
                
        detections.append(
            fo.Detection(label=elm['label'], bounding_box=elm['bbox'])
        )

    sample["ground_truth"] = fo.Detections(detections=detections)
    samples.append(sample)
    

# %%
if fo.dataset_exists(dataset_name):
    fo.delete_dataset(dataset_name)

print("Print to Dataset:")

dataset = fo.Dataset("lego-classification")
dataset.add_samples(samples)
dataset.persistent = True
dataset.export(labels_path="data/train.json", dataset_type=fo.types.COCODetectionDataset, label_field='ground_truth')


