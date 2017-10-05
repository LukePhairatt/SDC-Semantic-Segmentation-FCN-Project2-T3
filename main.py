import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

#
# Check TensorFlow Version
#
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

#
# Check for a GPU
#
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    

#
# Global variables
#
num_classes = 2                     # classes depth: e.g. road/not road (wxhxdepth image)
image_shape = (160, 576)            # image training/testing resize: reduce it to fit the memory (original:375x1242)
data_dir = './data'
runs_dir = './runs'
LEARNING_RATE = 1e-4
KEEP_PROB = 0.7
EPOCHS = 10
BATCH_SIZE = 1                      # note: there may be an out of memory issue with Adamoptimizer (other optimizers are fine with a bigger batch)
                                    #       the image size is quite big for this project. It requires big GPU memory to store and gradient calculations


#    
# TF place holder variables use by this FCN model#
#
#input_image   = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], 3])            # for flexibility, will set shape from the input image 
correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
learning_rate = tf.placeholder(tf.float32)
keep_prob     = tf.placeholder(tf.float32)

#
# NN functions #
#
def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    # Use tf.saved_model.loader.load to load the model and weights
    # Use tf.Print(img_input, [tf.shape(img_input)]) to print out dimension
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    # load the model and pre-trained weights
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    # get the graph
    graph = tf.get_default_graph()
    # get layer weights and return
    img_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    l3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    l4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    l7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return img_input, keep_prob, l3_out, l4_out, l7_out


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # note: the wxh is referenced for dimension scalling/ see below for the actual size
    # 1x1 convolution from the encoder (last layer) (7x7)
    # result 7x7
    vgg_layer7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding= 'same',
                                      kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                      kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    
    # upsampling to meet the layer 4 pooling (14x14)
    # result 14x14
    decode_1 = tf.layers.conv2d_transpose(vgg_layer7_1x1,num_classes,4,2,padding='same',\
                                        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                        kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    
    # add skip connection with layer 4 pooling 1x1 cov
    # result 14x14
    vgg_layer4_1x1 = tf.layers.conv2d(vgg_layer4_out,num_classes,1,padding='same',\
                                      kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                      kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    
    decode_2 = tf.add(decode_1, vgg_layer4_1x1)
    
    # upsampling to meet the layer 3 pooling (28x28)
    # result 28x28
    decode_3 = tf.layers.conv2d_transpose(decode_2,num_classes,4,2,padding='same',\
                                          kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                          kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    
    # add skip connections with layer 3 pooling 1x1 conv
    # result 28x28
    vgg_layer3_1x1 = tf.layers.conv2d(vgg_layer3_out,num_classes,1,padding='same',\
                                      kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                      kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    
    decode_4 = tf.add(decode_3, vgg_layer3_1x1)
    
    # finally upsampling to meet the original size (e.g. vgg input 224x224 so we need to scale up by 8)
    # Note: doesn't matter if our image is not 224x224 we scale up by the same factors anyway through out
    output = tf.layers.conv2d_transpose(decode_4,num_classes,16,8,padding='same',\
                                        kernel_initializer= tf.random_normal_initializer(stddev=0.01),
                                        kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    
    
    # Print network dimension e.g. VGG layers
    # output = tf.Print(output, [tf.shape(vgg_layer3_out)]) #[1 20 72...] with depth 256
    # output = tf.Print(output, [tf.shape(vgg_layer4_out)]) #[1 10 36...] with depth 512
    # output = tf.Print(output, [tf.shape(vgg_layer7_out)]) #[1 5 18...] with depth 4096
    
    return output   

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    # reshape to 2D for cross entropy loss function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    # loss function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    # Adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)   # AdamOptimizer provides the better result
    
    return logits, optimizer, cost



def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    
    # Run n epochs
    mean_training_loss = []
    for epoch in range(epochs):
        # training loss
        training_loss = []
        itr = 0
        # get batch data every epoch run
        # image: 3 channel 160, 576, 3
        # label: 2 classes binary 160. 576, 2 (e.g. channel [0]- not road highlight, channel [1]- road highlight)
        # so we could use the model to learn what the segmentation could be by given the prediction of the same size as the label 
        for image, label in get_batches_fn(batch_size, img_augment=True):
            itr += 1
            # feed data to the given layers
            feed = {input_image: image,
                    correct_label: label,
                    keep_prob: KEEP_PROB,
                    learning_rate: LEARNING_RATE }
            
            # train the network
            _, current_loss = sess.run([train_op, cross_entropy_loss], feed_dict = feed)
            
            # print progressing loss
            print("iteration {} : training loss: {} ---- ".format(itr, current_loss))
            # keep track of training loss
            training_loss.append(current_loss)
            
    
        # visualise loss/IoU training and validation accuracy here
        mean_loss = sum(training_loss) / len(training_loss)
        mean_training_loss.append(mean_loss)
        print("----- EPOCH {}: mean training loss: {} ---- ".format(epoch+1, mean_loss))
    
    # training loss summary
    print("Summary- Training loss every EPOCH: ", mean_training_loss)        
#
# Unit testing
#
def unit_test():
    tests.test_load_vgg(load_vgg, tf)
    tests.test_layers(layers)
    tests.test_optimize(optimize)
    tests.test_for_kitti_dataset(data_dir)
    tests.test_train_nn(train_nn)
    


#
# NN build and train
#
def run():
    # Download pretrained vgg model if not exist
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    
    # file to load/save
    save_file = 'train_model.ckpt'
    
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN (FCN-8) using load_vgg, layers, and optimize function
        input_image, keep_prob, l3_out, l4_out, l7_out = load_vgg(sess, vgg_path)
        l_out = layers(l3_out,l4_out,l7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(l_out, correct_label, learning_rate, num_classes)
        
        # init all TF variables: this model and graph
        sess.run(tf.global_variables_initializer())
        
        # load/save model
        all_vars = tf.all_variables()
        saver = tf.train.Saver(all_vars)
        # load previous trained data if exist
        if os.path.exists(save_file + '.meta'):
            saver.restore(sess, save_file)
        
        # Train NN using the train_nn function: passing tf place holder
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_image,\
                 correct_label, keep_prob, learning_rate)
        
        # save model data 
        saver.save(sess, save_file)
        
        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        


def run_video():    
    # trained model to load
    save_file = 'train_model.ckpt'
    
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        
        # Build NN (FCN-8) using load_vgg, layers, and optimize function
        input_image, keep_prob, l3_out, l4_out, l7_out = load_vgg(sess, vgg_path)
        l_out = layers(l3_out,l4_out,l7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(l_out, correct_label, learning_rate, num_classes)
        
        # init all TF variables: this model and graph
        sess.run(tf.global_variables_initializer())        
        
        # load/save model
        all_vars = tf.all_variables()
        saver = tf.train.Saver(all_vars)
        # load previous trained data if exist
        if os.path.exists(save_file + '.meta'):
            saver.restore(sess, save_file)
        else:
            print("No prediction- No tranied model found!")
        
        # Save inference data using helper.save_inference_samples
        helper.save_inference_video(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        

if __name__ == '__main__':
    unit_test()
    run()
    #run_video()
