# https://github.com/ljjsalt/TinyImageNetDataset

import zipfile
import imageio
import numpy as np
import os
from PIL import Image

from collections import defaultdict
from torch.utils.data import Dataset
import urllib

from tqdm.autonotebook import tqdm


def download_and_unzip(url, root, filename=None):
    """Download a file from a url, unzip and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
    """
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)
    unzipped_fpath = fpath + '#unzipped'
    os.makedirs(root, exist_ok=True)

    if os.path.exists(fpath) or os.path.exists(unzipped_fpath):
        print("File already downloaded")
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)

    if os.path.exists(unzipped_fpath):
        print("File already unzipped")
    else:
        print("Unzipping " + fpath)
        with zipfile.ZipFile(fpath, "r") as zip_ref:
            zip_ref.extractall("./data/")
        os.rename(fpath, unzipped_fpath)
        print("Unzipped")


def _add_channels(img, total_channels=3):
    while len(img.shape) < 3:  # third axis is the channels
        img = np.expand_dims(img, axis=-1)
    while(img.shape[-1]) < 3:
        img = np.concatenate([img, img[:, :, -1:]], axis=-1)
    return img


"""Creates a paths datastructure for the tiny imagenet.

Args:
  root_dir: Where the data is located
  download: Download if the data is not there

Members:
  label_id:
  ids:
  nit_to_words:
  data_dict:

"""


class TinyImageNetPaths:
    def __init__(self, root_dir, download=False):
        if download:
            download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip',
                               root_dir)

        root_dir = os.path.join(root_dir, 'tiny-imagenet-200')

        train_path = os.path.join(root_dir, 'train')
        val_path = os.path.join(root_dir, 'val')
        test_path = os.path.join(root_dir, 'test')

        wnids_path = os.path.join(root_dir, 'wnids.txt')
        words_path = os.path.join(root_dir, 'words.txt')

        self._make_paths(train_path, val_path, test_path,
                         wnids_path, words_path)

    def _make_paths(self, train_path, val_path, test_path,
                    wnids_path, words_path):
        self.ids = []
        with open(wnids_path, 'r') as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)
        self.nid_to_words = defaultdict(list)
        with open(words_path, 'r') as wf:
            for line in wf:
                nid, labels = line.split('\t')
                labels = list(map(lambda x: x.strip(), labels.split(',')))
                self.nid_to_words[nid].extend(labels)

        self.paths = {
            'train': [],  # [img_path, id, nid, box]
            'val': [],  # [img_path, id, nid, box]
            'test': []  # img_path
        }

        # Get the test paths
        self.paths['test'] = list(map(lambda x: os.path.join(test_path, x),
                                      os.listdir(test_path)))
        # Get the validation paths and labels
        with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = os.path.join(val_path, 'images', fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = self.ids.index(nid)
                self.paths['val'].append((fname, label_id, nid, bbox))

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            anno_path = os.path.join(train_path, nid, nid+'_boxes.txt')
            imgs_path = os.path.join(train_path, nid, 'images')
            label_id = self.ids.index(nid)
            with open(anno_path, 'r') as annof:
                for line in annof:
                    fname, x0, y0, x1, y1 = line.split()
                    fname = os.path.join(imgs_path, fname)
                    bbox = int(x0), int(y0), int(x1), int(y1)
                    self.paths['train'].append((fname, label_id, nid, bbox))


"""Datastructure for the tiny image dataset.

Args:
  root_dir: Root directory for the data
  mode: One of "train", "test", or "val"
  preload: Preload into memory
  load_transform: Transformation to use at the preload time
  transform: Transformation to use at the retrieval time
  download: Download the dataset

Members:
  tinp: Instance of the TinyImageNetPaths
  img_data: Image data
  label_data: Label data
"""


class TinyImageNetDataset(Dataset):
    def __init__(self, root, mode='train', task='classification', preload=True, load_transform=None,
                 transform=None, download=False, max_samples=None, output_shape=(32, 32)):
        assert(task == 'classification' or task == 'localization')
        tinp = TinyImageNetPaths(root, download)
        self.mode = mode
        self.label_idx = 1  # from [image, id, nid, box]
        self.loc_idx = 3  # from [image, id, nid, box]
        self.preload = preload
        self.transform = transform
        self.transform_results = dict()
        self.output_shape = output_shape

        self.task = task

        self.IMAGE_SHAPE = (64, 64, 3)

        self.img_data = []
        self.label_data = []
        self.loc_data = []

        self.max_samples = max_samples
        self.samples = tinp.paths[mode]
        self.samples_num = len(self.samples)

        if self.max_samples is not None:
            self.samples_num = min(self.max_samples, self.samples_num)
            self.samples = np.random.permutation(
                self.samples)[:self.samples_num]

        if self.preload:
            # Try to locate preload file on disk
            target_path = os.path.join(root, 'tiny-imagenet-200')
            label_exist = os.path.exists(os.path.join(target_path, 'label-'+mode+'.npy'))
            loc_exist = os.path.exists(os.path.join(target_path, 'loc-'+mode+'.npy'))
            img_exist = os.path.exists(os.path.join(target_path, 'img-'+mode+'.npy'))

            if img_exist and (label_exist and task == 'classification') or (loc_exist and task == 'localization'):
                print("Loading preloaded data from disk")
                self.img_data = np.load(os.path.join(target_path, 'img-'+mode+'.npy'), allow_pickle=True)
                if task == 'classification':
                    self.label_data = np.load(os.path.join(target_path, 'label-'+mode+'.npy'))
                elif task == 'localization':
                    self.loc_data = np.load(os.path.join(target_path, 'loc-'+mode+'.npy'))
            else:
                load_desc = "Preloading {} data...".format(mode)
                if img_exist:
                    self.img_data = np.load(os.path.join(target_path, 'img-'+mode+'.npy'), allow_pickle=True)
                else:
                    self.img_data = np.zeros((self.samples_num,) + self.IMAGE_SHAPE,
                                            dtype=np.float32)

                if self.task == 'classification':
                    self.label_data = np.zeros((self.samples_num,), dtype=np.int)
                elif self.task == 'localization':
                    self.loc_data = np.zeros((self.samples_num, 4), dtype=np.int)

                for idx in tqdm(range(self.samples_num), desc=load_desc):
                    s = self.samples[idx]
                    if not img_exist:
                        img = imageio.imread(s[0])
                        img = _add_channels(img)
                        self.img_data[idx] = img
                    if mode != 'test':
                        if self.task == 'classification':
                            self.label_data[idx] = s[self.label_idx]
                        elif self.task == 'localization':
                            self.loc_data[idx] = s[self.loc_idx]

                np.save(os.path.join(target_path, 'img-'+mode+'.npy'), self.img_data)
                if self.task == 'classification':
                    np.save(os.path.join(target_path, 'label-'+mode+'.npy'), self.label_data)
                elif self.task == 'localization':
                    np.save(os.path.join(target_path, 'loc-'+mode+'.npy'), self.loc_data)
                print("Preloaded data saved to disk")

            if load_transform:
                for lt in load_transform:
                    result = lt(self.img_data, self.label_data)
                    self.img_data, self.label_data = result[:2]
                    if len(result) > 2:
                        self.transform_results.update(result[2])

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        if self.preload:
            img = self.img_data[idx]
            if self.mode == 'test':
                target = None
            else:
                if self.task == 'classification':
                    target = self.label_data[idx]
                else:
                    target = self.loc_data[idx]
        else:
            s = self.samples[idx]
            img = imageio.imread(s[0])
            target = None if self.mode == 'test' else s[self.label_idx]

        img = Image.fromarray(img.astype(np.uint8)).resize(self.output_shape)

        if self.transform:
            img = self.transform(img)

        sample = (img, target)

        return sample


if __name__ == "__main__":
    download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip')