from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
from scipy.misc import imsave

from .bases import BaseImageDataset


class Market1501(BaseImageDataset):
    """
    Market1501

    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'Market-1501-v15.09.15'

    def __init__(self, root='data', verbose=True, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1 and os.environ.get('junk') is None:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1 and os.environ.get('junk') is None:
                continue  # junk images are just ignored
            assert -1 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

class REID_Keensense(object):
    """
    Partial_iLIDS

    Dataset statistics:
    # identities: 60
    # images: 600
    """
    def __init__(self, root='/home/xiaoxi.xjl/re_id_dataset/Pedestrain_Keensense',**kwargs):
        self.dataset_dir = root
        self.imgs_dir=osp.join(self.dataset_dir,'images')

        self._check_before_run()
        data_tuple,num_pids,num_imgs,person_dict=self._process_dir(self.imgs_dir)
        split_rate=0.98
        num_train=int(num_pids*split_rate) # 971

        train_tuple=[]
        query_tuple=[]
        gallery_tuple=[]

        train_id=[]
        query_id=[]
        gallery_id=[]


        for key,value in person_dict.items():
            if key<num_train:
                for metadata in value:
                    train_tuple.append(metadata)
                    train_id.append(key)
            if key>=num_train:
                for metadata in value:
                    test_data=metadata
                    cut=test_data[0].split('/')
                    person=cut[6]
                    qianzhui,houzhui=person.split('.')
                    split_qianzhui=qianzhui.split('_')
                    pid = split_qianzhui[0]
                    camid = split_qianzhui[1]
                    imgid = split_qianzhui[2]
                    imgid=int(imgid)
                    if imgid == 1:
                        query_tuple.append(test_data)
                        query_id.append(key)
                    if imgid != 1:
                        gallery_tuple.append(test_data)
                        gallery_id.append(key)

        num_train_pids=len(set(train_id))
        num_query_pids=len(set(query_id))
        num_gallery_pids=len(set(gallery_id))

        print("=> KeenSense_Holistic loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, len(train_tuple)))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, len(query_tuple)))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, len(gallery_tuple)))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_pids, num_imgs))
        print("  ------------------------------")

        self.train = train_tuple
        self.query = query_tuple
        self.gallery = gallery_tuple

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids
        self.num_train_cams=4

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.imgs_dir):
            raise RuntimeError("'{}' is not available".format(self.imgs_dir))

    def _process_dir(self, dir_path):

        img_paths = glob.glob(os.path.join(dir_path,'*.png'))
        person_dict={}
        pid_lst=[]
        camid_lst=[]
        imgid_lst=[]
        dataset = []
        for path in img_paths:
            file_name=path.split('/')
            name=file_name[6]
            person=name.split('_')
            if len(person)==3:
                pid = int(person[0])
                camid = int(person[1])
                imgid, _ = person[2].split('.')
            if len(person)==4:
                pid = int(person[0])
                camid = int(person[1])
                imgid=person[2]

            imgid=int(imgid)
            pid_lst.append(pid)
            camid_lst.append(camid)
            imgid_lst.append(imgid)

        pid_container = set(pid_lst)
        num_pids = len(pid_container)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for pid in pid_container:
            pid = pid2label[pid]
            person_dict[pid]=[]

        for index,img_path in enumerate(img_paths):
            pid=pid_lst[index]
            pid = pid2label[pid]
            camid=camid_lst[index]
            imgid=imgid_lst[index]
            person_dict[pid].append((img_path,pid,camid,imgid))
            dataset.append((img_path,pid,camid,imgid))

        num_imgs=len(dataset)


        return dataset, num_pids,num_imgs,person_dict
