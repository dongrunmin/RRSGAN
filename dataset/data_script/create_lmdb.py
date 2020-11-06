"""Create lmdb files for [General images (291 images/DIV2K) | Vimeo90K | REDS] training datasets"""

import sys
import os
import os.path as osp
import glob
import pickle
from multiprocessing import Pool
import numpy as np
import lmdb
import cv2
import util as util
#sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
#import utils.util as util  # noqa: E402


def main():
    dataset = 'val_1st'
    if dataset == 'val_1st':
        opt = {}
        ## GT
        data_path = '../val/val_1st/'
        lmdb_path = '../val/val_1st_lmdb/'
        fold_list = glob.glob('../val/val_1st/*')
        util.mkdir_and_rename(lmdb_path)
        print(fold_list)
        for path in fold_list:
            base_name = osp.basename(path)
            print(base_name)
            opt['img_folder'] = path
            opt['lmdb_save_path'] = os.path.join(lmdb_path,base_name+'.lmdb')
            opt['name'] = dataset + '_' + base_name
            general_image_folder(opt) 
    elif dataset == 'train':
        opt = {}
        ## GT
        data_path = '../train/train_data/'
        lmdb_path = '../train/train_lmdb/'
        fold_list = glob.glob('../train/train_data/*')
        util.mkdir_and_rename(lmdb_path)
        print(fold_list)
        for path in fold_list:
            base_name = osp.basename(path)
            print(base_name)
            opt['img_folder'] = path
            opt['lmdb_save_path'] = os.path.join(lmdb_path,base_name+'.lmdb')
            opt['name'] = dataset + '_' + base_name
            general_image_folder(opt)


def read_image_worker(path, key):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return (key, img)


def general_image_folder(opt):
    """Create lmdb for general image folders
    Users should define the keys, such as: '0321_s035' for DIV2K sub-images
    If all the images have the same resolution, it will only store one copy of resolution info.
        Otherwise, it will store every resolution info.
    """
    #### configurations
    read_all_imgs = False  # whether real all images to memory with multiprocessing
    # Set False for use limited memory
    BATCH = 5000  # After BATCH images, lmdb commits, if read_all_imgs = False
    n_thread = 40
    ########################################################
    img_folder = opt['img_folder']
    lmdb_save_path = opt['lmdb_save_path']
    meta_info = {'name': opt['name']}
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    if osp.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    #### read all the image paths to a list
    print('Reading image path list ...')
    all_img_list = sorted(glob.glob(osp.join(img_folder, '*')))
    keys = []
    for img_path in all_img_list:
        keys.append(osp.splitext(osp.basename(img_path))[0])

    if read_all_imgs:
        #### read all images to memory (multiprocessing)
        dataset = {}  # store all image data. list cannot keep the order, use dict
        print('Read images with multiprocessing, #thread: {} ...'.format(n_thread))
        pbar = util.ProgressBar(len(all_img_list))

        def mycallback(arg):
            '''get the image data and update pbar'''
            key = arg[0]
            dataset[key] = arg[1]
            pbar.update('Reading {}'.format(key))

        pool = Pool(n_thread)
        for path, key in zip(all_img_list, keys):
            pool.apply_async(read_image_worker, args=(path, key), callback=mycallback)
        pool.close()
        pool.join()
        print('Finish reading {} images.\nWrite lmdb...'.format(len(all_img_list)))

    #### create lmdb environment
    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)

    #### write data to lmdb
    pbar = util.ProgressBar(len(all_img_list))
    txn = env.begin(write=True)
    resolutions = []
    for idx, (path, key) in enumerate(zip(all_img_list, keys)):
        pbar.update('Write {}'.format(key))
        key_byte = key.encode('ascii')
        data = dataset[key] if read_all_imgs else cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if data.ndim == 2:
            H, W = data.shape
            C = 1
        else:
            H, W, C = data.shape
        txn.put(key_byte, data)
        resolutions.append('{:d}_{:d}_{:d}'.format(C, H, W))
        if not read_all_imgs and idx % BATCH == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    #### create meta information
    # check whether all the images are the same size
    assert len(keys) == len(resolutions)
    if len(set(resolutions)) <= 1:
        meta_info['resolution'] = [resolutions[0]]
        meta_info['keys'] = keys
        print('All images have the same resolution. Simplify the meta info.')
    else:
        meta_info['resolution'] = resolutions
        meta_info['keys'] = keys
        print('Not all images have the same resolution. Save meta info for each image.')

    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')



if __name__ == "__main__":
    main()


