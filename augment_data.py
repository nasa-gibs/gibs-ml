import sys
import os

from PIL import Image

train_directory = 'data/train'

if __name__ == '__main__':
	
	assert os.path.isdir(train_directory), "Couldn't find the dataset at {}".format(train_directory)

	# Get the filenames in the training directory
	filenames = os.listdir(train_directory)
	filenames = [os.path.join(train_directory, f) for f in filenames if f.endswith('.jpg')]

	for filename in filenames:
		image = Image.open(filename)
		img_name = os.path.split(filename)[-1].split(".")[0]

		# Rotate the image 90, 180, and 270 degrees
		image_rot_90 = image.rotate(90)
		image_rot_90.save("%s/%s-%s.jpg" % (train_directory, img_name, "R1"))

		image_rot_180 = image.rotate(180)
		image_rot_180.save("%s/%s-%s.jpg" % (train_directory, img_name, "R2"))

		image_rot_270 = image.rotate(270)
		image_rot_270.save("%s/%s-%s.jpg" % (train_directory, img_name, "R3"))

		# Flip the image horizontally and vertically
		image_hor_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
		image_hor_flip.save("%s/%s-%s.jpg" % (train_directory, img_name, "F1"))

		image_vert_flip = image.transpose(Image.FLIP_TOP_BOTTOM)
		image_vert_flip.save("%s/%s-%s.jpg" % (train_directory, img_name, "F2"))
