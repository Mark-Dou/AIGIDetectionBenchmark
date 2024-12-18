import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
import copy
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn
import torchvision.transforms.functional as TF

from io import BytesIO
from PIL import Image
from random import random, choice
from scipy import fftpack
from skimage import img_as_ubyte
from scipy.ndimage.filters import gaussian_filter
from torchvision.transforms import InterpolationMode

from utils.denoising_rgb import DenoiseNet
from utils.util import load_checkpoint
from preprocessing_model.LGrad_models import build_model


def data_augment(img, opt):
    img = np.array(img)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur_gray(img, sigma):
    if len(img.shape) == 3:
        img_blur = np.zeros_like(img)
        for i in range(img.shape[2]):
            img_blur[:, :, i] = gaussian_filter(img[:, :, i], sigma=sigma)
    else:
        img_blur = gaussian_filter(img, sigma=sigma)
    return img_blur


def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)


def cv2_jpg_gray(img, compress_val):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 0)
    return decimg


def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}


def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


class RandomMask(object):
    def __init__(self, ratio=0.5, patch_size=16, p=0.5):
        """
        Args:
            ratio (float or tuple of float): If float, the ratio of the image to be masked.
                                             If tuple of float, random sample ratio between the two values.
            patch_size (int): the size of the mask (d*d).
        """
        if isinstance(ratio, float):
            self.fixed_ratio = True
            self.ratio = (ratio, ratio)
        elif isinstance(ratio, tuple) and len(ratio) == 2 and all(isinstance(r, float) for r in ratio):
            self.fixed_ratio = False
            self.ratio = ratio
        else:
            raise ValueError("Ratio must be a float or a tuple of two floats.")

        self.patch_size = patch_size
        self.p = p

    def __call__(self, tensor):

        if random.random() > self.p: return tensor

        _, h, w = tensor.shape
        mask = torch.ones((h, w), dtype=torch.float32)

        if self.fixed_ratio:
            ratio = self.ratio[0]
        else:
            ratio = random.uniform(self.ratio[0], self.ratio[1])

        # Calculate the number of masks needed
        num_masks = int((h * w * ratio) / (self.patch_size ** 2))

        # Generate non-overlapping random positions
        selected_positions = set()
        while len(selected_positions) < num_masks:
            top = random.randint(0, (h // self.patch_size) - 1) * self.patch_size
            left = random.randint(0, (w // self.patch_size) - 1) * self.patch_size
            selected_positions.add((top, left))

        for (top, left) in selected_positions:
            mask[top:top+self.patch_size, left:left+self.patch_size] = 0

        return tensor * mask.expand_as(tensor)

class RandomJPEG():
    def __init__(self, quality=95, interval=1, p=0.1):
        if isinstance(quality, tuple):
            self.quality = [i for i in range(quality[0], quality[1]) if i % interval == 0]
        else:
            self.quality = quality
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            if isinstance(self.quality, list):
                quality = random.choice(self.quality)
            else:
                quality = self.quality
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img = Image.open(buffer)
        return img

def Get_Transforms(args):
    # for SAFE model
    size = args.input_size

    TRANSFORM_DICT = {
        'resize_BILINEAR': {
            'train': [
                transforms.RandomResizedCrop([size, size], interpolation=InterpolationMode.BILINEAR),
            ],
            'eval': [
                transforms.Resize([size, size], interpolation=InterpolationMode.BILINEAR),
            ],
        },

        'resize_NEAREST': {
            'train': [
                transforms.RandomResizedCrop([size, size], interpolation=InterpolationMode.NEAREST),
            ],
            'eval': [
                transforms.Resize([size, size], interpolation=InterpolationMode.NEAREST),
            ],
        },

        'crop': {
            'train': [
                transforms.RandomCrop([size, size], pad_if_needed=True),
            ],
            'eval': [
                transforms.CenterCrop([size, size]),
            ],
        },

        'source': {
            'train': [
                transforms.RandomCrop([size, size], pad_if_needed=True),
            ],
            'eval': [
            ],
        },
    }

    # region [Augmentations]
    transform_train, transform_eval = TRANSFORM_DICT[args.transform_mode]['train'], TRANSFORM_DICT[args.transform_mode]['eval']

    transform_train.extend([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.ToTensor(),
        RandomMask(ratio=(0.00, 0.75), patch_size=16, p=0.5),
    ])

    transform_eval.append(transforms.ToTensor())
    # endregion

    # region [Perturbatiocns in Testing]
    if args.jpeg_factor is not None:
        transform_eval.insert(0, RandomJPEG(quality=args.jpeg_factor, p=1.0))
    if args.blur_sigma is not None:
        transform_eval.insert(0, transforms.GaussianBlur(kernel_size=5, sigma=args.blur_sigma))
    if args.mask_ratio is not None and args.mask_patch_size is not None:
        transform_eval.append(RandomMask(ratio=args.mask_ratio, patch_size=args.mask_patch_size, p=1.0))
    # endregion

    return transforms.Compose(transform_train), transforms.Compose(transform_eval)

def get_processing_model(opt, device):
    if opt.detection_model == "LGrad":
        gen_model = build_model(gan_type='stylegan',
                                module='discriminator',
                                resolution=256,
                                label_size=0,
                                # minibatch_std_group_size = 1,
                                image_channels=3)
        gen_model.load_state_dict(torch.load(opt.LGrad_modelpath), strict=True)
        gen_model.to(device)
        gen_model.eval()
        opt.gen_model = gen_model

    elif opt.detection_model == 'LNP':
        model_restoration = DenoiseNet()
        load_checkpoint(model_restoration, opt.LNP_modelpath)
        # print("===>Testing using weights: ", opt.LNP_modelpath)
        model_restoration.to(device)
        model_restoration.eval()
        opt.model_restoration = model_restoration

    elif opt.detection_model == 'FreDect':
        opt.dct_mean = torch.load('./weights/auxiliary/dct_mean').permute(1, 2, 0).numpy()
        opt.dct_var = torch.load('./weights/auxiliary/dct_var').permute(1, 2, 0).numpy()

    elif opt.detection_model in ['GramNet', 'CNNSpot', 'UnivFD', 'NPR', 'SAFE', 'SimE', 'FatFormer']:
        opt = opt
    else:
        raise ValueError(f"Unsupported model_type: {opt.detection_model}")
    
    return opt


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}


def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])


def processing(img, opt, name):
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
    trans = transforms.Compose([
        rz_func,
        transforms.Lambda(lambda img: data_augment(img, opt) if opt.isTrain else img),
        crop_func,
        flip_func,
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN[name], std=STD[name]),
    ])
    return trans(img)


MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip": [0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip": [0.26862954, 0.26130258, 0.27577711]
}


def normlize_np(img):
    img -= img.min()
    if img.max() != 0: img /= img.max()
    return img * 255.


processimg = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])


def processing_LGrad(img, gen_model, opt, device):
    img_list = []
    img_list.append(torch.unsqueeze(processimg(img), 0))
    img = torch.cat(img_list, 0)
    img_cuda = img.to(torch.float32)
    img_cuda = img_cuda.to(device)
    img_cuda.requires_grad = True
    pre = gen_model(img_cuda)
    gen_model.zero_grad()
    grads = torch.autograd.grad(pre.sum(), img_cuda, create_graph=True, retain_graph=True, allow_unused=False)[0]
    for idx, grad in enumerate(grads):
        img_grad = normlize_np(grad.permute(1, 2, 0).cpu().detach().numpy())
    retval, buffer = cv2.imencode(".png", img_grad)
    if retval:
        img = Image.open(BytesIO(buffer)).convert('RGB')
    else:
        print("failing in save to memory")
    img = processing(img, opt, 'imagenet')
    return img


def dct2_wrapper(image, mean, var, log=True, epsilon=1e-12):
    """apply 2d-DCT to image of shape (H, W, C) uint8
    """
    # dct
    image = np.array(image)
    image = fftpack.dct(image, type=2, norm="ortho", axis=0)
    image = fftpack.dct(image, type=2, norm="ortho", axis=1)
    # log scale
    if log:
        image = np.abs(image)
        image += epsilon  # no zero in log
        image = np.log(image)
    # normalize
    image = (image - mean) / np.sqrt(var)
    return image


def processing_DCT(img, opt):
    input_img = copy.deepcopy(img)
    input_img = transforms.ToTensor()(input_img)
    input_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(input_img)

    img = transforms.Resize(opt.loadSize)(img)
    img = transforms.CenterCrop(opt.cropSize)(img)
    cropped_img = torch.from_numpy(dct2_wrapper(img, opt.dct_mean, opt.dct_var)).permute(2, 0, 1).to(dtype=torch.float)
    return cropped_img


def processing_PSM(img, opt):
    height, width = img.height, img.width

    input_img = copy.deepcopy(img)
    input_img = transforms.ToTensor()(input_img)
    input_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(input_img)

    img = transforms.Resize(opt.CropSize)(img)
    img = transforms.CenterCrop(opt.CropSize)(img)
    cropped_img = transforms.ToTensor()(img)
    cropped_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(cropped_img)

    scale = torch.tensor([height, width])

    return input_img, cropped_img, scale


def processing_LNP(img, model_restoration, opt):
    img_list = []
    img = np.array(img).astype(np.float32)
    img = img / 255.
    img = torch.from_numpy(np.float32(img))
    img = img.permute(2, 0, 1)
    img_list.append(torch.unsqueeze(img, 0))
    img = torch.cat(img_list, 0)

    rgb_restored = model_restoration(img.to(opt.device))

    rgb_restored = torch.clamp(rgb_restored, 0, 1)
    rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    for batch in range(len(rgb_restored)):
        denoised_img = img_as_ubyte(rgb_restored[batch])
        retval, buffer = cv2.imencode(".png", denoised_img * 255)
        if retval:
            denoised_img = Image.open(BytesIO(buffer)).convert('RGB')
        else:
            print("failing in save to memory")
    denoised_img = processing(denoised_img, opt, 'imagenet')
    return denoised_img



