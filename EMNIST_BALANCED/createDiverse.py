from dis import dis
from email.mime import image
import os
import cv2
import falconn
import numpy as np
from skimage.metrics import structural_similarity as ssim
import splitfolders


dataLength = []

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

def startFalconn(z):
    path = 'data/data1/emnist_balanced/train/'+z
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
        index += 1
    print(len(data))
    print(len(similar_list))
    dataLength.append(len(data))

    path2 = 'data/data3/all/'+z
    print(len(dissimilar_list))
    if not os.path.exists(path2):
        os.makedirs(path2)
    path2 = path2+'/'
    # index = 0
    for x in dissimilar_list:
        locA = data_image_path[x]
        name = locA.lstrip(path)
        img = cv2.imread(locA)
        cv2.imwrite(path2+name, img)
        # if index == 2500:
        #     break
        # else:
        #     index+=1
    # while index <= times:
    #     locA = data_image_path[dissimilar_list[index]]
    #     name = locA.lstrip(path)
    #     img = cv2.imread(locA)
    #     cv2.imwrite(path2+name, img)
    #     index = index + 1

def simpleWork(val):
    index = 0
    locA = 'data/data1/emnist_balanced/train/'+val+'/'
    # folder pat
    count = 0
    # Iterate directory
    for p in os.listdir(locA):
        # check if current path is a file
        if os.path.isfile(os.path.join(locA, p)):
            count += 1

    times = count/2
    locB = 'data/data2/all/'+val
    if not os.path.exists(locB):
        os.makedirs(locB)
    locB = locB+'/'
    for filename in os.listdir(locA):
        f = os.path.join(locA, filename)
    # # checking if it is a file
        if os.path.isfile(f):
            img = cv2.imread(f)
            cv2.imwrite(locB+filename, img)
        if index == times:
            break
        index += 1


def hardWork(val, minLength, curr):
    index = 0
    locA = 'data/data3/all/'+val+'/'
    # folder pat
    count = 0
    # Iterate directory
    for p in os.listdir(locA):
        # check if current path is a file
        if os.path.isfile(os.path.join(locA, p)):
            count += 1
    
    if(dataLength[curr] == minLength):
        times = minLength
    elif (dataLength[curr] > 3*minLength):
        times = minLength*3
    else:
        times = dataLength[curr]

    locB = 'data/data3/all2/'+val
    if not os.path.exists(locB):
        os.makedirs(locB)
    locB = locB+'/'
    for filename in os.listdir(locA):
        f = os.path.join(locA, filename)
    # # checking if it is a file
        if os.path.isfile(f):
            img = cv2.imread(f)
            cv2.imwrite(locB+filename, img)
        if index == times:
            break
        index += 1

def changeParameter():
    z = ('48','49','50','51','52','53','54','55','56','57',
          '65','66','67','68','69','70',
          '71','72','73','74','75','76','77','78','79','80',
          '81','82','83','84','85','86','87','88','89',
          '90','97','98',
          '100','101','102','103',
          '104','110','113','114','116')
   
    # times = 1
    # z = [z2]
    ## times is the number of data points that needs to be considered
    for val in z:
        # simpleWork(val)
        startFalconn(val)
    
    # min_dataLength = min(dataLength)
    # index = 0
    # for val in z:
    #     hardWork(val, min_dataLength, index)
    #     index+=1
    # print('Done')
                    
# Get the Data
def split(dataLoc, fileLoc):
  splitfolders.ratio(dataLoc, fileLoc, seed=42, ratio=(1,0), group_prefix=None)
  return fileLoc

changeParameter()
split('data/data3/all/', 'data/data3/emnist_balanced/')