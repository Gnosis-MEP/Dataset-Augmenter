#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
import cv2 as cv

from dataset_augmenter.composition_utils import composite_from_bbox

np.set_printoptions(suppress=True, linewidth=120)

def intersect_rects(r1, r2):
	(r1x, r1y, r1w, r1h) = r1
	(r2x, r2y, r2w, r2h) = r2

	rx = max(r1x, r2x)
	rw = min(r1x+r1w, r2x+r2w) - rx

	ry = max(r1y, r2y)
	rh = min(r1y+r1h, r2y+r2h) - ry

	if rw > 0 and rh > 0:
		return np.array([rx, ry, rw, rh])
	else:
		return None

def rect_to_slice(rect):
	(rx, ry, rw, rh) = rect
	return np.s_[ry:ry+rh, rx:rx+rw]

def composite(src, dst=None, position=(0,0), background_color=None):
	# this is pronounced "com-PUH-sit" to all you non-native speakers

	(src_height, src_width, src_ch) = src.shape
	src_has_alpha = (src_ch == 4)

	# shortcut
	# composite() may be useful to use in imshow, so people "see" transparent data
	# and if data has no alpha channel, pass through
	if dst is None and not src_has_alpha:
		return src

	if dst is None:
		# background (grid or flat color)
		dst = np.empty((src_height, src_width, 3), dtype=np.uint8)
		if background_color is None:
			(i,j) = np.mgrid[0:src_height, 0:src_width]
			i = (i // 8) % 2
			j = (j // 8) % 2
			dst[i == j] = 192
			dst[i != j] = 255
		else:
			dst[:,:] = background_color

	(dst_height, dst_width) = dst.shape[:2]
	dst_has_alpha = (dst.shape[2] == 4)

	src_rect = np.array([0, 0, src_width, src_height])
	dst_rect = np.array([0, 0, dst_width, dst_height])
	offset = position + (0,0) # 4-tuple

	src_roi = intersect_rects(src_rect, dst_rect - offset)
	dst_roi = intersect_rects(dst_rect, src_rect + offset)

	if src_roi is not None: # there is overlap
		assert dst_roi is not None
		dst_slice = dst[rect_to_slice(dst_roi)]
		src_slice = src[rect_to_slice(src_roi)]

		if src_has_alpha:
			src_alpha = src_slice[:,:,3][..., None] # None adds a dimension for numpy broadcast rules
			if src_alpha.dtype == np.uint8:
				src_alpha = src_alpha / np.float32(255)

			blended = src_slice[:,:,0:3] * src_alpha + dst_slice[:,:,0:3] * (1-src_alpha)

		else:
			blended = src_slice[:,:,0:3]

		dst_slice[:,:,0:3] = blended.astype(dst.dtype)

		if dst_has_alpha:
			# new_alpha = (src_slice[:,:,3] + dst_slice[:,:,3])

			new_alpha = src_slice[:,:,:] + dst_slice[:,:,:] * (1-src_alpha)
			dst_slice[:,:,:] = new_alpha.astype(dst.dtype)

	return dst

def main():
	# logo = cv.imread(cv.samples.findFile("opencv-logo.png"), cv.IMREAD_UNCHANGED)
	# lena = cv.imread(cv.samples.findFile("lena.jpg"), cv.IMREAD_UNCHANGED)

	logo = cv.imread(cv.samples.findFile("t_oi.png"), cv.IMREAD_UNCHANGED)
	lena = cv.imread(cv.samples.findFile("testing.png"), cv.IMREAD_UNCHANGED)

	# smaller logo...
	# inpainting here to fill ordinarily transparent areas with color
	# those transparent pixels will become part of edge pixels (where alpha channel has gradients)
	# and if they stay black, those edge pixels will turn dark
	# a different fix would require alpha-aware resizing
	# logo[:,:,0:3] = cv.inpaint(logo[:,:,0:3], inpaintMask=255-logo[:,:,3], inpaintRadius=0, flags=cv.INPAINT_NS)
	# logo = cv.pyrDown(logo)

	# composite0 = composite(logo)
	composite1 = composite(logo, background_color=(128, 255, 255))
	composite2 = composite(logo, dst=lena.copy(), position=(250, 250))
	# NOTE: `dst` will be altered
	# this is a useful behavior in case we want to composite multiple things onto the same background
	# it's like the drawing primitives in imgproc
	# if you don't want it to be touched, pass a copy instead (as is done here)

	logo = (cv.pyrDown(cv.pyrDown(logo)))
	logo[:,:,3] //= 2 # reduce transparency to half strength for this demo
	for k in range(100):
		x = np.random.randint(-100, +600)
		y = np.random.randint(-100, +600)
		composite(logo, dst=lena, position=(x,y))

	# cv.save ("composite0", composite0)
	# cv.imwrite("composite0.png", composite0)
	cv.imwrite("composite1.png", composite1)
	cv.imwrite("composite2.png", composite2)
	cv.imwrite("lena2.png", lena)

	# while True:
	# 	key = cv.waitKey(-1)
	# 	if key in (13, 27): # ESC, ENTER
	# 		break

	# cv.destroyAllWindows()


def crop_simple(image_ndarray):
	threshold = 30  # Adjust this threshold as needed

	# Get the coordinates where alpha values are not close to zero
	y, x = np.where(np.abs(image_ndarray[:, :, 3] - 0) > threshold)

	minx = np.min(x)
	miny = np.min(y)
	maxx = np.max(x)
	maxy = np.max(y)
	cropImg = image_ndarray[miny:maxy, minx:maxx]
	return cropImg

if __name__ == '__main__':
	# main()

	fg_image = np.zeros((1080, 1920, 4), np.uint8)
	oi_image = cv.imread(cv.samples.findFile("../inputs/OIs/origin/car/00160-127.png"), cv.IMREAD_UNCHANGED)
	# oi_image = cv.imread(cv.samples.findFile("opencv-logo.png"), cv.IMREAD_UNCHANGED)


	oi_image = crop_simple(oi_image)
	cv.imwrite('crop_oi_fig.png', oi_image)
	composite1 = composite_from_bbox(
		src=oi_image, dst=fg_image, bbox=[
				0.7971365610343892, 0.7383325641074374, 1.0, 0.8305200641074374
			]
	)
	cv.imwrite("composite1.png", composite1)