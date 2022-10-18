from dis import dis
from email.mime import image
import os
import cv2
import falconn
import numpy as np
from skimage.metrics import structural_similarity as ssim
import splitfolders

# Gets the images in  a dataset and divides them up
def load_dataset(path):
    images_gray = []
    images = []
    # for testing manually impose all the labels
    labels = []
    image_path = []
    for img in os.listdir(path):
        if img.endswith(".png"):
            image_path.append(os.path.join(path, img))
            img_array_gray = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)

            img_array = cv2.resize(img_array_gray, (50,50))
            images_gray.append(img_array)

            img_array = img_array.reshape(-1)
            img_array = np.float32(img_array)

            images.append(img_array)
    
    images_gray = np.array(images_gray)
    images = np.array(images)
    return (images_gray, images, image_path)

def startFalconn(z, times):
    path = 'data/cifar10/train/'+z
    data_grey, data, data_image_path = load_dataset(path)
    data = data/255.0
    parameters = falconn.LSHConstructionParameters()
    parameters.dimension = data.shape[1]
    parameters.lsh_family = falconn.LSHFamily.Hyperplane
    parameters.distance_function = falconn.DistanceFunction.EuclideanSquared
    parameters.l = 1
    parameters.num_setup_threads = 1
    parameters.num_rotations = 1
    parameters.seed = 16
    parameters.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
    parameters.k = 30
    index = falconn.LSHIndex(parameters)
    index.setup(data)
    query_object = index.construct_query_object()
    query_object.set_num_probes(1)
    match = []
    similar_list = []
    index = 0


    for img in data:
        m = query_object.find_near_neighbors(img, 100)
        if len(m) > 1:
            for x in m:
                ssim_val = ssim(img.reshape(-1,50), data[x].reshape(-1,50), full=False)
                if(ssim_val >= 0.6):
                    if(ssim_val != 1):
                        if(x not in similar_list):
                            similar_list.append(x)
        match.append(m)
    dissimilar_list = []
    
    index = 0
    for x in match:
        for i in x:
            if i not in similar_list:
                if i not in dissimilar_list:
                    dissimilar_list.append(i)
        # if(x[0] not in dissimilar_list):
        #     if len(x) > 1:
        #         for i in x:
                    
        #     else:
        #         dissimilar_list.append(x[0])
        index += 1
    print(len(data))
    print(len(similar_list))

    path2 = 'data2/all/'+z+'/'
    print(len(dissimilar_list))
    index = 0
    while index <= times:
        locA = data_image_path[dissimilar_list[index]]
        name = locA.lstrip(path)
        img = cv2.imread(locA)
        cv2.imwrite(path2+name, img)
        index = index + 1

def simpleWork(val, times):
    index = 0
    locA = 'data/cifar10/train/'+val+'/'
    locB = 'data2/all/'+val+'/'
    for filename in os.listdir(locA):
        f = os.path.join(locA, filename)
    # # checking if it is a file
        if os.path.isfile(f):
            img = cv2.imread(f)
            cv2.imwrite(locB+filename, img)
        if index == times:
            break
        index += 1
    

    # Average Similar Image Printing Method
    # sum = 0
    # for x in match:
    #     sum += len(x)
    # print(sum/len(match)) 
           

def changeParameter():
    z1 = ''
    z2 = 'automobile'
    z3 = 'bird'
    z4 = 'cat'
    z5 = 'deer'
    z6 = 'dog'
    z7 = 'frog'
    z8 = 'horse'
    z9 = 'ship'
    z0 = 'truck'
    z = [z1,z2,z3,z4,z5,z6,z7,z8,z9,z0]
    # times = 3000
    times = 3060
    # z = [z2]
    ## times is the number of data points that needs to be considered
    for val in z:
        # simpleWork(val, times)
        startFalconn(val, times)
                    
# Get the Data
def split(dataLoc, fileLoc):
  splitfolders.ratio(dataLoc, fileLoc, seed=42, ratio=(1,0), group_prefix=None)
  return fileLoc

changeParameter()
split('data2/all/', 'data2/cifar10/')