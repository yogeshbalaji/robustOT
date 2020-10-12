from .vision import VisionDataset

from PIL import Image

from random import shuffle
import os
import os.path
import sys
import numpy as np


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, ids_to_load, extensions=None, is_valid_file=None):

    # ids_to_load are a tuple of (true_attr_id, mapped_id / class_id, Fraction, enable_flag)

    # Read attribute files
    attr_file_path = os.path.join(dir, 'list_attr_celeba.txt')
    fp = open(attr_file_path, 'r')
    file_lines = fp.readlines()
    file_lines = [line.strip('\n') for line in file_lines]
    fp.close()

    file_attributes = {}
    total_attribute_num = {}

    for line_id, line in enumerate(file_lines):
        if line_id >= 2:
            line = line.replace('  ', ' ')
            fields = line.split(' ')
            img_name = fields[0]

            attribute_vector_string = fields[1:]
            attributes_mapped = []

            for id_load in ids_to_load:
                attr_id = id_load[0]

                if int(attribute_vector_string[attr_id]) == id_load[3]:
                    mapped_id = id_load[1]
                    attributes_mapped.append(mapped_id)
                    if mapped_id in total_attribute_num:
                        total_attribute_num[mapped_id] += 1
                    else:
                        total_attribute_num[mapped_id] = 1

            file_attributes[img_name] = attributes_mapped

    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)

    xall = []

    for id_load in ids_to_load:
        class_id = id_load[1]
        fraction = id_load[2]
        xall.append(float(total_attribute_num[class_id]) / fraction)
    xmin = np.array(xall).min()

    # num_to_fill indicates number of data points that needs to be filled for attribute attr_id
    num_to_fill = {}
    num_filled = {}
    for id_load in ids_to_load:
        class_id = id_load[1]
        fraction = id_load[2]
        num_to_fill[class_id] = min(total_attribute_num[class_id], int(xmin * fraction))
        num_filled[class_id] = 0

    # TODO: Use shuffle
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):

            path = os.path.join(root, fname)
            if is_valid_file(path):

                img_name_mapped = root.split('/')[-1] + fname
                attribute_vector = file_attributes[img_name_mapped]

                append_flag = False
                label = -1

                for id_load in ids_to_load:
                    class_id = id_load[1]

                    if class_id in attribute_vector:
                        if num_filled[class_id] >= num_to_fill[class_id]:
                            append_flag = False
                        else:
                            append_flag = True
                            num_filled[class_id] += 1
                            label = class_id

                if append_flag:
                    item = (path, label)
                    images.append(item)

    return images


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CelebAAttributes(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader=default_loader, extensions=None, transform=None,
                 target_transform=None, is_valid_file=None, input_attribute_list=None):
        super(CelebAAttributes, self).__init__(root, transform=transform,
                                               target_transform=target_transform)

        extensions = IMG_EXTENSIONS if is_valid_file is None else None

        assert input_attribute_list is not None
        # attribute_list is a list of tuples (attribute_name, attribute_flag, fraction)

        # Assert class fractions sum to 1
        frac_sum = 0
        for attribute in input_attribute_list:
            frac_sum += attribute[2]

        if frac_sum < 1.0:
            raise ValueError('Sum of attribute fractions should be 1')

        # Reading attributes
        attr_file_path = os.path.join(root, 'list_attr_celeba.txt')
        fp = open(attr_file_path, 'r')
        file_lines = fp.readlines()
        fp.close()

        # First line is num_samples
        # Second line is header
        attributes_all = file_lines[1].split(' ')
        attribute_mappings_all = {}
        for id, attribute in enumerate(attributes_all):
            attribute_mappings_all[attribute] = id

        # Each value field is a tuple of (attr_id_to_class_mapping, class fraction)
        attribute_ids_to_load = []
        for inp_attr in input_attribute_list:
            attr_name = inp_attr[0]
            assert attr_name in attribute_mappings_all

            attr_id = attribute_mappings_all[attr_name]

            tup = (attr_id, len(attribute_ids_to_load), inp_attr[2], inp_attr[1])
            attribute_ids_to_load.append(tup)

        self.attribute_ids_to_load = attribute_ids_to_load

        samples = make_dataset(self.root, self.attribute_ids_to_load, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

    def __len__(self):
        return len(self.samples)




#
# class ImageFolder(DatasetFolder):
#     """A generic data loader where the images are arranged in this way: ::
#
#         root/dog/xxx.png
#         root/dog/xxy.png
#         root/dog/xxz.png
#
#         root/cat/123.png
#         root/cat/nsdf3.png
#         root/cat/asd932_.png
#
#     Args:
#         root (string): Root directory path.
#         transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         loader (callable, optional): A function to load an image given its path.
#         is_valid_file (callable, optional): A function that takes path of an Image file
#             and check if the file is a valid_file (used to check of corrupt files)
#
#      Attributes:
#         classes (list): List of the class names.
#         class_to_idx (dict): Dict with items (class_name, class_index).
#         imgs (list): List of (image path, class_index) tuples
#     """
#
#     def __init__(self, root, transform=None, target_transform=None,
#                  loader=default_loader, is_valid_file=None):
#         super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
#                                           transform=transform,
#                                           target_transform=target_transform,
#                                           is_valid_file=is_valid_file)
#         self.imgs = self.samples
