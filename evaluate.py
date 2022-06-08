from __future__ import absolute_import, division, print_function
import numpy as np
from Mytransforms import transform_preds
import cv2

def calc_dists(preds, target, normalize):
	#print(preds.shape, target.shape) #(1,15,2)
	preds  =  preds.astype(np.float32)
	target = target.astype(np.float32)
	dists  = np.zeros((preds.shape[1], preds.shape[0])) #15,1

	for n in range(preds.shape[0]): #1
		for c in range(preds.shape[1]): #15
			if target[n, c, 0] > 1 and target[n, c, 1] > 1: #as pixel values
				normed_preds   =  preds[n, c, :] / normalize[n]
				normed_targets = target[n, c, :] / normalize[n]
				dists[c, n]    = np.linalg.norm(normed_preds - normed_targets)
			else:
				dists[c, n]    = -1

	return dists


def dist_acc(dists, threshold = 0.5):
	"""
	Returns number of pixels/joints which are valid & under threshold
	"""
	dist_cal     = np.not_equal(dists, -1)
	num_dist_cal = dist_cal.sum()

	if num_dist_cal > 0:
		return np.less(dists[dist_cal], threshold).sum() * 1.0 / num_dist_cal
	else:
		return -1


def get_max_preds(batch_heatmaps):
	batch_size = batch_heatmaps.shape[0]
	num_joints = batch_heatmaps.shape[1]
	width      = batch_heatmaps.shape[3]

	heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
	idx               = np.argmax(heatmaps_reshaped, 2)
	maxvals           = np.amax(heatmaps_reshaped, 2)

	maxvals = maxvals.reshape((batch_size, num_joints, 1))
	idx     = idx.reshape((batch_size, num_joints, 1))

	preds   = np.tile(idx, (1,1,2)).astype(np.float32)

	preds[:,:,0] = (preds[:,:,0]) % width
	preds[:,:,1] = np.floor((preds[:,:,1]) / width)

	pred_mask    = np.tile(np.greater(maxvals, 0.0), (1,1,2))
	pred_mask    = pred_mask.astype(np.float32)

	preds *= pred_mask
	# print(preds.shape)

	return preds, maxvals

def taylor(hm, coord):
	heatmap_height = hm.shape[0]
	heatmap_width = hm.shape[1]
	px = int(coord[0])
	py = int(coord[1])
	if 1 < px < heatmap_width-2 and 1 < py < heatmap_height-2:
		dx  = 0.5 * (hm[py][px+1] - hm[py][px-1])
		dy  = 0.5 * (hm[py+1][px] - hm[py-1][px])
		dxx = 0.25 * (hm[py][px+2] - 2 * hm[py][px] + hm[py][px-2])
		dxy = 0.25 * (hm[py+1][px+1] - hm[py-1][px+1] - hm[py+1][px-1] \
			+ hm[py-1][px-1])
		dyy = 0.25 * (hm[py+2*1][px] - 2 * hm[py][px] + hm[py-2*1][px])
		derivative = np.matrix([[dx],[dy]])
		hessian = np.matrix([[dxx,dxy],[dxy,dyy]])
		if dxx * dyy - dxy ** 2 != 0:
			hessianinv = hessian.I
			offset = -hessianinv * derivative
			offset = np.squeeze(np.array(offset.T), axis=0)
			coord += offset
	return coord

def gaussian_blur_darkpose(hm, kernel):
	border = (kernel - 1) // 2
	batch_size = hm.shape[0]
	num_joints = hm.shape[1]
	height = hm.shape[2]
	width = hm.shape[3]
	for i in range(batch_size):
		for j in range(num_joints):
			origin_max = np.max(hm[i,j])
			dr = np.zeros((height + 2 * border, width + 2 * border))
			dr[border: -border, border: -border] = hm[i,j].copy()
			dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
			hm[i,j] = dr[border: -border, border: -border].copy()
			hm[i,j] *= origin_max / np.max(hm[i,j])
	return hm

def get_final_preds_darkpose(hm, center, scale):
	# print(center.shape)
	#center = [center[0].item(), center[1].item()]
	coords, maxvals = get_max_preds(hm) #coords 8,15,2
	#print(hm.shape) #1,15,46,46
	# post-processing
	hm = gaussian_blur_darkpose(hm, 11)
	hm = np.maximum(hm, 1e-10)
	hm = np.log(hm)
	for n in range(coords.shape[0]): #batch
			for p in range(coords.shape[1]): #joint
				coords[n,p] = taylor(hm[n][p], coords[n][p])

	preds = coords.copy()
	# Transform back
	# for i in range(coords.shape[0]):
	# 	# preds[i] = transform_preds(
	# 	# 	coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
	# 	# )
	# 	preds[i] = transform_preds(
	# 		coords[i], center, scale, [heatmap_width, heatmap_height]
	# 	)
	return preds, maxvals

def accuracy(output, target, thr_PCK, thr_PCKh, dataset, center, scale,DARK= False,hm_type='gaussian', threshold=0.5):
	#print("Output shape", output.shape) #(1,15,46,46)
	idx  = list(range(output.shape[1]))
	norm = 1.0

	if hm_type == 'gaussian':
		if not DARK:
			pred, _   = get_max_preds(output)
			target, _ = get_max_preds(target)
		else:
			pred, _   = get_final_preds_darkpose(output,center,scale)
			target, _ = get_final_preds_darkpose(target,center,scale)
		h         = output.shape[2]
		w         = output.shape[3]
		norm      = np.ones((pred.shape[0], 2)) * np.array([h,w]) / 10

	dists = calc_dists(pred, target, norm)

	acc     = np.zeros((len(idx)))
	avg_acc = 0
	cnt     = 0
	visible = np.zeros((len(idx)))

	for i in range(len(idx)): #accuracy per joint
		acc[i] = dist_acc(dists[idx[i]])
		if acc[i] >= 0:
			avg_acc = avg_acc + acc[i]
			cnt    += 1
			visible[i] = 1
		else:
			acc[i] = 0

	avg_acc = avg_acc / cnt if cnt != 0 else 0

	if cnt != 0:
		acc[0] = avg_acc

	# PCKh
	PCKh = np.zeros((len(idx)))
	avg_PCKh = 0

	if dataset == "LSP":
		headLength = np.linalg.norm(target[0,14,:] - target[0,13,:])
	# elif dataset == "COCO":
	# 	headLength = np.linalg.norm(target[0,4,:] - target[0,5,:])
	# elif dataset == "Penn_Action":
	# 	neck = [(target[0,1,0]+target[0,2,0])/2, (target[0,1,1]+target[0,2,1])/2]
	# 	headLength = np.linalg.norm(target[0,0,:] - neck)
	# elif dataset == "NTID":
	# 	headLength = 2*(np.linalg.norm(target[0,4,:] - target[0,3,:]))
	# elif dataset == "PoseTrack":
	# 	headLength = 2*(np.linalg.norm(target[0,1,:] - target[0,2,:]))
	# elif dataset == "BBC":
	# 	neck = [(target[0,6,0]+target[0,7,0])/2, (target[0,6,1]+target[0,7,1])/2]
	# 	headLength = np.linalg.norm(target[0,1,:] - neck)
	# elif dataset == "MPII":
	# 	headLength = np.linalg.norm(target[0,9,:] - target[0,10,:])


	for i in range(len(idx)):
		PCKh[i] = dist_acc(dists[idx[i]], thr_PCKh*headLength)
		if PCKh[i] >= 0:
			avg_PCKh = avg_PCKh + PCKh[i]
		else:
			PCKh[i] = 0

	avg_PCKh = avg_PCKh / cnt if cnt != 0 else 0

	if cnt != 0:
		PCKh[0] = avg_PCKh


	# PCK
	PCK = np.zeros((len(idx)))
	avg_PCK = 0

	if dataset == "COCO":
		pelvis = [(target[0,12,0]+target[0,13,0])/2, (target[0,12,1]+target[0,13,1])/2]
		torso  = np.linalg.norm(target[0,13,:] - pelvis)


	elif dataset == "LSP":
		pelvis = [(target[0,3,0]+target[0,4,0])/2, (target[0,3,1]+target[0,4,1])/2]
		torso  = np.linalg.norm(target[0,13,:] - pelvis)

	elif dataset == "MPII":
		torso  = np.linalg.norm(target[0,7,0] - target[0,8,0])

	for i in range(len(idx)):
		PCK[i] = dist_acc(dists[idx[i]], thr_PCK*torso)

		if PCK[i] >= 0:
			avg_PCK = avg_PCK + PCK[i]
		else:
			PCK[i] = 0

	avg_PCK = avg_PCK / cnt if cnt != 0 else 0

	if cnt != 0:
		PCK[0] = avg_PCK


	return acc, PCK, PCKh, cnt, pred, visible