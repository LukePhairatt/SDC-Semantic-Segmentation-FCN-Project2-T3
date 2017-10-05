import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import cv2
from moviepy.editor import VideoFileClip
from IPython.display import HTML

#
# Image Augmentation 
#

# x,y shift
def ImageShift(image, label):
    shift_rc = [int(image.shape[0]/5), int(image.shape[1]/5)]
    # shift max range up/down (row), and left/right (column)
    tx = shift_rc[1] * (np.random.uniform() - 0.5)   # shift column (+,- range)
    ty = shift_rc[0] * (np.random.uniform() - 0.5)   # shift row (+,- range)
    M = np.float32([[1, 0, tx], [0, 1,ty]])
    dst_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    # apply the same translation to label
    label_f = label.astype(float)
    dst_label_f = cv2.warpAffine(label_f, M, (label_f.shape[1], label_f.shape[0]))
    # convert back to bool type
    dst_label = dst_label_f.astype(bool)
    return dst_image, dst_label

# rotation
def ImageRotation(image, label, max_rot = 10):
    rows = int(image.shape[0])
    cols = int(image.shape[1])
    rotz = max_rot*(np.random.uniform() - 0.5)
    M = cv2.getRotationMatrix2D(( int(cols/2) , int(rows/2) ), rotz, 1)
    dst_image = cv2.warpAffine(image,M,(cols,rows))

    label_f = label.astype(float)
    dst_label_f = cv2.warpAffine(label_f,M,(cols,rows))
    dst_label = dst_label_f.astype(bool)
    return dst_image, dst_label


# Adjust brightness
# Source: https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def AdjustBrightness(image, label, gamma_minmax = [0.5,1.5]):
    # generate a random brightness gamma min-max range
    gamma = np.random.uniform()%((gamma_minmax[1]  - gamma_minmax[0] ) + 1) + gamma_minmax[0] 
    invGamma = 1.0 / gamma
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    # apply gamma correction using the lookup table
    dst_image = cv2.LUT(image, table)
    return dst_image, label


def AddShadow(image, label):
    dst_image= np.copy(image)
    # Add shadow
    brightness = 0.25
    height, width = dst_image.shape[:2]
    # Shadoe ROI
    # pick width, height at least 25 pixels
    shd_w = 20 + int(width*np.random.uniform())
    shd_h = 20 + int(height*np.random.uniform())
    
    # pick top corner
    corner_xt = int(width*np.random.uniform())
    corner_yt = int(height*np.random.uniform())
    # compute bottom corner
    corner_xb = corner_xt + shd_w
    corner_yb = corner_yt + shd_h
    # check bound
    if corner_xb > width:
        corner_xb = width
    if corner_yb > height:
        corner_yb = height
        
    # Mask this image region with the shadow
    for i in range(3):
        dst_image[corner_yt:corner_yb,corner_xt:corner_xb,i] = image[corner_yt:corner_yb,corner_xt:corner_xb,i]*brightness
        
    return dst_image, label

# Flip image horizontally
def ImageFlip(image,label):
    label_f = label.astype(float)
    dst_image = cv2.flip(image,1)
    dst_label_f = cv2.flip(label_f,1)
    dst_label = dst_label_f.astype(bool)
    return dst_image, dst_label

#
# Train/Test/Prediction helper functions
#
class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size, img_augment=False):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        
        ##
        ## Image augmentation file (due to GPU memory size limited I can use only 1 batch size)
        ## so the augmentation preparation goes here
        ##
        
        if (img_augment):
            # size of data to be augment 
            aug_length = int(len(image_paths)/3)
            aug_files = image_paths[:aug_length]
            # make full image list
            aug_image_paths = image_paths + aug_files
        
            # make augmentation indicator for aug_length
            aug_indicator = [0 for _ in range(len(aug_image_paths))]                # no augment 
            aug_indicator[:aug_length] = [1 for _ in range(aug_length)]             # add 1/3 augmentation
            # mix them up
            random.shuffle(aug_indicator)
            # set the new image files
            image_paths = aug_image_paths
        else:
            aug_indicator = [0 for _ in range(len(image_paths))]                    # no augment 
            
        
        
        # generate batch data
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            #for index, image_file in enumerate(image_paths[batch_i:batch_i+batch_size]):
            for img_index in range(batch_i,batch_i+batch_size):
                image_file = image_paths[img_index]
                gt_image_file = label_paths[os.path.basename(image_file)]
                # read image (0-255) 
                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                # read ground truth binary
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
                
                # use augment image
                if(aug_indicator[img_index]):             
                    
                    aug_img = np.copy(image)
                    aug_label = np.copy(gt_image)
                    # aug_img,aug_label = ImageShift(aug_img,aug_label)
                    # aug_img,aug_label = ImageRotation(aug_img,aug_label)
                    aug_img,aug_label = AdjustBrightness(aug_img,aug_label)
                    aug_img,aug_label = ImageFlip(aug_img,aug_label)
                    aug_img,aug_label = AddShadow(aug_img,aug_label)   
                        
                    # add to the batch data
                    images.append(aug_img)
                    gt_images.append(aug_label)
                # use original    
                else:
                    # add to the batch data
                    images.append(image)
                    gt_images.append(gt_image)
                
                
            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
        
# ------------------------------------------------------------------------------------------------- #
def gen_test_video(sess, logits, keep_prob, image_pl, image_shape, image_file):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
   
    image = scipy.misc.imresize(image_file, image_shape)
    im_softmax = sess.run([tf.nn.softmax(logits)],{keep_prob: 1.0, image_pl: [image]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)
    return np.array(street_im)

         
def save_inference_video(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):    
    # Prediction processing
    def process_image(image):
        # crop height to the ratio
        img_nn = image[125:414,:]
        # Run NN on test images
        image_outputs = gen_test_video(sess, logits, keep_prob, input_image, image_shape, img_nn)
        return image_outputs
    
    # process video image
    out_video = 'out_video_raw.mp4'
    clip1 = VideoFileClip("solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image) 
    white_clip.write_videofile(out_video, audio=False)    
    print('Prediction Finished. Saving test video')









