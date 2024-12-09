import torch
from math import floor
import os
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
#from models import get_custom_transformer, get_model

import argparse
from utils.utils import collate_features
from utils.file_utils import save_hdf5
from utils.hit_cache import hit_cache, RamDiskCache
import h5py
import openslide
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.library.impl_abstract.*")

__implemented_models = {
    'resnet50': 'image-net',
    'resnet101': 'image-net',
    'plip': 'https://huggingface.co/vinid/plip',
    'ctranspath': 'models/ckpts/ctranspath.pth',
    'dinov2_vitl': 'models/ckpts/dinov2_cpath_v1.pth'
}


def list_models():
    print('The following are implemented models:')
    for k, v in __implemented_models.items():
        print('{}: {}'.format(k, v))
    return __implemented_models


def get_model_test(model_name, device, gpu_num):
    """_summary_

    Args:
        model_name (str): the name of the requried model
        device (torch.device): device, e.g. 'cuda'
        gpu_num (int): the number of GPUs used in extracting features

    Raises:
        NotImplementedError: if the model name does not exist

    Returns:
        nn.Module: model
    """
    print('get_model')
    if model_name == 'resnet50':
        from models.resnet_custom import resnet50_baseline
        model = resnet50_baseline(pretrained=True).to(device)
    elif model_name == 'resnet101':
        from models.resnet_custom import resnet101_baseline
        model = resnet101_baseline(pretrained=True).to(device)
    elif model_name == 'vit_base_patch16_224_21k':
        from models.transforms_model import vit_base_patch16_224_21k
        model = vit_base_patch16_224_21k(device, gpu_num)
    elif model_name == 'vit_large_patch16_224_21k':
        from models.transforms_model import vit_large_patch16_224_21k
        model = vit_large_patch16_224_21k(device, gpu_num)
    elif model_name == 'vit_huge_patch14_224_21k':
        from models.transforms_model import vit_huge_patch14_224_21k
        model = vit_huge_patch14_224_21k(device, gpu_num)
        
    # our models
    elif model_name in ['mae_vit_large_patch16-1-40000', 'mae_vit_large_patch16-1-140000',
                        'mae_vit_l_1000slides_19epoch', 'mae_vit_l_10000slides_3epoch', 
                        'mae_vit_large_patch16-1epoch-180M',
                        
                        ]:
        from models.mae_endoder import mae_pretrained_model
        model = mae_pretrained_model(device, gpu_num, 'mae_vit_large_patch16',ckpt=__implemented_models[model_name] ,input_size=224)
    
    elif model_name in ['mae_vit_huge_patch14_1000slides_9epoch',
                        'mae_vit_huge_patch14_1000slides_0epoch',
                        'mae_vit_huge_patch14_1000slides_22epoch',
                        ]:
        
        from models.mae_endoder import mae_pretrained_model
        model = mae_pretrained_model(device, gpu_num, 'mae_vit_huge_patch14',ckpt=__implemented_models[model_name] ,input_size=224)

    elif model_name in ['dinov2_vitl']:
        print('model_name: ', model_name)
        from models.dinov2 import build_model
        model, _ = build_model(device, gpu_num, model_name, __implemented_models[model_name])

    elif model_name == 'ctranspath':
        from models.ctrans import ctranspath
        print('\n!!!! please note that ctranspath requires the modified timm 0.5.4, you can find package at here: models/ckpts/timm-0.5.4.tar , please install if needed ...\n')
        model = ctranspath(ckpt_path=__implemented_models['ctranspath']).to(device)
    elif model_name == 'plip':
        from models.plip import plip
        model = plip(device, gpu_num)
        
    else:
        raise NotImplementedError(f'{model_name} is not implemented')
    
    if model_name in ['resnet50', 'resnet101']:
        if gpu_num > 1:
            model = torch.nn.parallel.DataParallel(model)
        model = model.eval()
    return model

def get_custom_transformer(model_name):
    """_summary_

    Args:
        model_name (str): the name of model

    Raises:
        NotImplementedError: not implementated

    Returns:
        torchvision.transformers: the transformers used to preprocess the image
    """
    if model_name in ['resnet50', 'resnet101']:
        from models.resnet_custom import custom_transforms
        custom_trans = custom_transforms()
    elif model_name in ['vit_base_patch16_224_21k', 'vit_large_patch16_224_21k', 'vit_huge_patch14_224_21k']:
        # Do nothing, let vit process do the image processing
        from torchvision import transforms as tt
        custom_trans = tt.Lambda(lambda x: torch.from_numpy(np.array(x)))
    elif model_name in ['mae_vit_large_patch16-1-40000', 'mae_vit_large_patch16-1-140000',
                        'mae_vit_l_1000slides_19epoch', 'mae_vit_l_10000slides_3epoch',
                        'mae_vit_large_patch16-1epoch-180M',
                        ]:
        from models.mae_endoder import mae_transform
        custom_trans = mae_transform(224)

    elif model_name in ['mae_vit_huge_patch14_1000slides_9epoch',
                        'mae_vit_huge_patch14_1000slides_0epoch',
                        'mae_vit_huge_patch14_1000slides_22epoch',
                        ]:
        from models.mae_endoder import mae_transform
        custom_trans = mae_transform(224)
          
    elif model_name == 'ctranspath':
        from models.ctrans import ctranspath_transformers
        custom_trans = ctranspath_transformers()
    elif model_name == 'hipt':
        from torchvision import transforms as tt
        custom_trans = tt.functional.to_tensor()
    elif model_name == 'plip':
        # Do nothing, let CLIP process do the image processing
        from torchvision import transforms as tt
        custom_trans = tt.Lambda(lambda x: torch.from_numpy(np.array(x)))
    
    elif model_name in ['dinov2_vitl']:
        from models.dinov2 import build_transform
        custom_trans = build_transform()
        
    else:
        raise NotImplementedError('Transformers for {} is not implemented ...'.format(model_name))

    return custom_trans


def light_compute_w_loader(file_path, output_path, wsi, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1, custom_transformer=None):
	"""
	Do not save features to h5 file to save storage
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, custom_transforms=custom_transformer,
		custom_downsample=custom_downsample, target_patch_size=target_patch_size)
	x, y = dataset[0]
	kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}
	print('Data Loader args:', kwargs)
	loader = DataLoader(dataset=dataset, batch_size=batch_size,  **kwargs, collate_fn=collate_features, prefetch_factor=16)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	features_list = []
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			features = model(batch)
			features = features.cpu()
			asset_dict = {'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
			features_list.append(features)

	features = torch.cat(features_list, dim=0)
	return features


def compute_w_loader(file_path, output_path, wsi, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1, custom_transformer=None):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, custom_transforms=custom_transformer,
		custom_downsample=custom_downsample, target_patch_size=target_patch_size)
	x, y = dataset[0]
	kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size,  **kwargs, collate_fn=collate_features, prefetch_factor=None)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			
			features = model(batch)
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


def find_path_for_tcga_wsi(root, slide_id, ext, datatype):
	if datatype.lower() == 'tcga':
		dirs = os.listdir(root)
		for d in dirs:
			slide_file_path = os.path.join(root, d, slide_id+ext)
			if os.path.exists(slide_file_path):
				return slide_f
		slide_file_path = os.path.join(root, slide_id+ext)
		return slide_file_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.tif')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--model', type=str)
parser.add_argument('--datatype', type=str)
parser.add_argument('--save_storage', type=str, default='no')

# speed up
parser.add_argument('--ramdisk_cache', default='', type=str)

# Histlogy-pretrained MAE setting
parser.add_argument('--mae_checkpoint', type=str, default=None, help='path to pretrained mae checkpoint')

args = parser.parse_args()

if __name__ == '__main__':
     
	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError
     
	# Check if dataset creation works
	print('Creating Dataset_All_Bags...', flush=True)
	bags_dataset = Dataset_All_Bags(csv_path)
	print('Dataset created successfully', flush=True)

	# Check if directories are being created properly
	print('Creating directories...', flush=True)
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files', args.model), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files', args.model), exist_ok=True)
	print('Directories created', flush=True)

	# List destination files
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files', args.model))
	print('Destination files listed', flush=True)

	# Proceed to model checkpoint loading
	print('loading model checkpoint:', args.model, flush=True)
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	print('Device:{}, GPU Count:{}'.format(device.type, torch.cuda.device_count()), flush=True)

	# Check before calling the model function
	print('Calling get_model_test...', flush=True)
	model = get_model_test(args.model, device, torch.cuda.device_count())
	print('Model loaded successfully', flush=True)

	custom_transformer = get_custom_transformer(args.model)

	total = len(bags_dataset)
	print('######',total)
	
	# obtain slide_id
	get_slide_id = lambda idx: bags_dataset[idx].split(args.slide_ext)[0]
	
	# check the exists wsi
	exist_idxs = []

	for bag_candidate_idx in range(total):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		if not os.path.exists(h5_file_path):
			print(h5_file_path, 'does not exist ...')
			continue
		else:
			exist_idxs.append(bag_candidate_idx)


	for index, bag_candidate_idx in enumerate(exist_idxs):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		if not os.path.exists(h5_file_path):
			print(h5_file_path, 'does not exist ...')
			continue

		# TCGA
		slide_file_path = find_path_for_tcga_wsi(args.data_slide_dir, slide_id, args.slide_ext, args.datatype)
		
		print('\nprogress: {}/{}'.format(bag_candidate_idx, len(exist_idxs)))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		# ----------- hit cache, speed up next iteration-------------
		_next_id = exist_idxs[(index+1)%len(exist_idxs)]
		_next_slide_id = get_slide_id(_next_id)
		print('caching next id is:', _next_slide_id)
		_next_file_path = find_path_for_tcga_wsi(args.data_slide_dir, _next_slide_id, args.slide_ext, args.datatype)
		hit_cache(_next_file_path)
		# ----------------------------------------------------------

		output_path = os.path.join(args.feat_dir, 'h5_files', args.model, bag_name)
		time_start = time.time()
		print(slide_file_path)
		wsi = openslide.open_slide(slide_file_path)
		
		if args.save_storage == 'yes':
			features = light_compute_w_loader(h5_file_path, output_path, wsi, 
						model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
						custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size,
						custom_transformer=custom_transformer)
		else:
			output_file_path = compute_w_loader(h5_file_path, output_path, wsi, 
						model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
						custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size,
						custom_transformer=custom_transformer
						)
			time_elapsed = time.time() - time_start
			print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
			file = h5py.File(output_file_path, "r")

			features = file['features'][:]
			print('features size: ', features.shape)
			print('coordinates size: ', file['coords'].shape)
			features = torch.from_numpy(features)

		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', args.model, bag_base+'.pt'))

	print('Extracting end!')

