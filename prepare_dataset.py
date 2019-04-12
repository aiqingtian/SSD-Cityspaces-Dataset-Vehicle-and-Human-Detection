import random
import os
import pandas as pd

img_dir = 'cityscapes_samples'
json_dir = 'cityscapes_samples_labels'

def get_item_info(img_dir, json_dir):
    train_file_name_list = []
    train_img_dir_list = []
    train_json_dir_list = []
    val_file_name_list = []
    val_img_dir_list = []
    val_json_dir_list = []
    test_file_name_list = []
    test_img_dir_list = []
    test_json_dir_list = []
    for path, dir_list, file_list in os.walk(img_dir):
        for dir_name in dir_list:
            for img_path, sub_dir_list, img_file_list in os.walk(os.path.join(img_dir, dir_name)):
                for file in img_file_list:
                    rand = random.randint(1, 101)
                    if rand > 25:
                        train_file_name_list.append(file[:-16])
                        train_img_dir_list.append(os.path.join(img_dir, dir_name, file))
                        json_file_name = file[:-16] + '_gtCoarse_polygons.json'
                        train_json_dir_list.append(os.path.join(json_dir, dir_name, json_file_name))
                    elif rand <= 25 and rand > 10 :
                        val_file_name_list.append(file[:-16])
                        val_img_dir_list.append(os.path.join(img_dir, dir_name, file))
                        json_file_name = file[:-16] + '_gtCoarse_polygons.json'
                        val_json_dir_list.append(os.path.join(json_dir, dir_name, json_file_name))
                    else:
                        test_file_name_list.append(file[:-16])
                        test_img_dir_list.append(os.path.join(img_dir, dir_name, file))
                        json_file_name = file[:-16] + '_gtCoarse_polygons.json'
                        test_json_dir_list.append(os.path.join(json_dir, dir_name, json_file_name))

    print(len(train_json_dir_list), len(val_json_dir_list), len(test_json_dir_list))

    pd.DataFrame(train_img_dir_list).to_csv('train_img_dir_list.csv', header=None)
    pd.DataFrame(train_json_dir_list).to_csv('train_json_dir_list.csv', header=None)
    pd.DataFrame(val_img_dir_list).to_csv('val_img_dir_list.csv', header=None)
    pd.DataFrame(val_json_dir_list).to_csv('val_json_dir_list.csv', header=None)
    pd.DataFrame(test_img_dir_list).to_csv('test_img_dir_list.csv', header=None)
    pd.DataFrame(test_json_dir_list).to_csv('test_json_dir_list.csv', header=None)

    return train_file_name_list, train_img_dir_list, train_json_dir_list, val_file_name_list, val_img_dir_list, val_json_dir_list, test_file_name_list, test_img_dir_list, test_json_dir_list

get_item_info(img_dir, json_dir)