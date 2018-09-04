from dependency import *
from data_utils import *
import os.path
v
# Data_mean
def cal_img_mean(train_image_names_classes):
    mean = np.array([0.0, 0.0, 0.0]); count = 0
    for file_path, _ in train_image_names_classes:
        count += 1
        image = np.array(imread(file_path, mode='RGB').astype(np.float))
        mean[0] += np.mean(image[:, :, 0])
        mean[1] += np.mean(image[:, :, 1])
        mean[2] += np.mean(image[:, :, 2])
    return mean / count


def get_data_mean():
    if os.path.isfile(FLAGS.DATA_MEAN_FILE):
        print("Reading from data_mean.txt...")
        with open(FLAGS.DATA_MEAN_FILE, "r") as f:
            data_mean = [float(x) for x in f.readline().split("\t")]
    else:
        print("Calculating data_mean...")
        data = dataset(FLAGS.TRAIN_DIR, FLAGS.TEST_DIR)
        data_mean = cal_img_mean(data.train_image_names_classes)
        print(data_mean)
        with open(FLAGS.DATA_MEAN_FILE, 'w') as f:
            f.write(str(data_mean[0])+"\t"+str(data_mean[1])+"\t"+str(data_mean[2]))
    return data_mean


def transformation_vgg(dict_1, skip="fc8"):
    dict_2 = {}
    for key, value in dict_1.items():
        if key != skip:
            dict_2[key] = {}
            for idx in range(len(value)):
                dict_2[key][idx] = value[idx]
    return dict_2