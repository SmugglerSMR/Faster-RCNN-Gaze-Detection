# Performed Steps:
Create Pascal dataset for eyes first:
        Eye dataset was missing left eyes.
ffmpeg -i $file -filter:v "crop=600:600:250:250" $file -y

python object_detection/dataset_tools/create_pascal_tf_record.py --data_dir=VOCdevkit/ --year=VOC2012 --output_path=pascal_single.record --label_map_path=label_full.pbtxt --set=trainval

File for settings: faster_rcnn_inception_v2_coco_single_eye
```
python object_detection/legacy/train.py --train_dir=train_resnet_10 --pipeline_config_path=faster_rcnn_inception_v2_coco_single_eye.config
```
lyra_train_single_10.sh - file to run job





+++++++++++++++++++++++++++++++++++++++++++++++++
File for settings: faster_rcnn_inception_resnet_v2_atrous_oid_single_eye
```
python object_detection/legacy/train.py --train_dir=train_resnet_55 --pipeline_config_path=faster_rcnn_inception_resnet_v2_atrous_oid_single_eye.config
```
lyra_train_single_55.sh - file to run job

rm -R output/

python object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=faster_rcnn_inception_resnet_v2_atrous_oid_single_eye.config --trained_checkpoint_prefix=train_resnet_single_55/model.ckpt-15412 --output_directory=output_resnet_55


python object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=faster_rcnn_inception_resnet_v2_atrous_oid.config --trained_checkpoint_prefix=train_resnet_full_56/model.ckpt-14465 --output_directory=output_resnet_56


========================================================================
Create Pascal dataset for full view second:
        File contains 50 images. dataset-fullView-Fliped
Create graph
```
export PYTHONPATH=$(pwd):$(pwd)/slim

python object_detection/dataset_tools/create_pascal_tf_record.py --data_dir=VOCdevkit/ --year=VOC2012 --output_path=pascal_full.record --label_map_path=label_full.pbtxt --set=trainval
```
File for settings: faster_rcnn_inception_v2_coco_eye
```
python object_detection/legacy/train.py --train_dir=train_incep_10 --pipeline_config_path=faster_rcnn_inception_v2_coco.config
```
lyra_train_full_10.sh - file to run job

rm -R output/

python object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=faster_rcnn_inception_v2_coco_eye.config --trained_checkpoint_prefix=train_incep_10/model.ckpt-24031 --output_directory=output_incep_10

+++++++++++++++++++++++++++++++++++++++++++++++++
File for settings: faster_rcnn_inception_resnet_v2_atrous_oid
```
python object_detection/legacy/train.py --train_dir=train_resnet_10 --pipeline_config_path=faster_rcnn_inception_resnet_v2_atrous_oid.config
```
lyra_train_full_10.sh - file to run job


=====================================================================
# Second day. Trying improve model fitting
python object_detection/dataset_tools/create_pascal_tf_record.py --data_dir=VOCdevkit/ --year=VOC2012 --output_path=pascal_single_gray.record --label_map_path=label_full.pbtxt --set=trainval

File for settings: faster_rcnn_inception_v2_coco_single_eye_gray

python object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=faster_rcnn_inception_v2_coco_single_eye_gray.config --trained_checkpoint_prefix=train_gray_24/model.ckpt-66259 --output_directory=output_gray_24


# Third day. Send for training new dataset with two models.
# Send additional set with only iris eye from single view.
dataset called RGB-processed. Has 50 images / 48 annotations

python object_detection/dataset_tools/create_pascal_tf_record.py --data_dir=VOCdevkit --year=VOC2012 --output_path=pascal_rgb.record --label_map_path=label_full.pbtxt --set=trainval

python object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=faster_rcnn_resnet101_coco.config --trained_checkpoint_prefix=train_rgb_resnet_17/model.ckpt-13699 --output_directory=output_rgb_resnet_17



Same with:
faster_rcnn_resnet101_coco model

# Testing day:

55 hours requres 7 minutes to run with some success

17 yongest one takes 3 minutes

# Faster-RCNN using Inception model (equivalent to Xception)
Code build on top of following examples:
https://github.com/jaspereb/FasterRCNNTutorial/
Model/research link
https://drive.google.com/file/d/1JR2LtfsrcDgdXY6-6waZeT6VvO_Bbyea/view?usp=sharing

# WARNING!!
Code uses full path to certain locations. Be sure it will be removed later.

## Prereqs
You must have:

##
Create text file with frame and label on it.


* installed tensorflow (I have 1.4 installed locally using PIP although they suggest using a virtualenv). If you are doing this for the first time you will need cuda, Nvidia drivers which work, cudnn and a bunch of other packages, make sure to set up the paths in .bashrc properly, including the LD_LIBRARY_PATH and add the 'research' directory and 'research/slim' from the next step to your pythonpath.
* cloned and built the tensorflow/models/research folder into the tensorflow directory, you may not need to run the build files which are included with this. If you get script not found errors from the python commands then try running the various build scripts. (https://github.com/tensorflow/models/tree/master/research) 
* Jupyter notebook (pip install --user jupyter) 
* labelimg https://github.com/tzutalin/labelImg
* ImageMagick cli utilities

## Creating the Dataset and Training
The goal is to take rgb images and create a dataset in the same format as Pascal VOC, this can then be used to create the 'pascal.record' TFRecord files which is used for training.

What we need to create is the following. Start by creating all of the empty folders.

Datasets must have following structures:
~~~
+VOCdevkit
    +VOC2012
        +Annotations
                -A bunch of .xml labels
        +JPEGImages
                -A bunch of .jpg images
        +ImageSets
                +Main
                        -aeroplane_trainval.txt (This is just a list of the jpeg files without file extensions, the train.py script reads this file for all the images it is supposed to include.
                        -trainval.txt (An exact copy of the aeroplane_trainval.txt)

        +trainingConfig.config (training config file similar to https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)
~~~

Resize images is to X*600 (See 'How big should my images be?')

~~~
cd .../JPEGImages
for file in $PWD/*.jpg
do
	# convert $file -resize 717x600 $file
	# Using ffmpeg is more effective
	ffmpeg -i $file -vf scale=717:600 $file -y
done
~~~

Optionally, rename them to consecutive numbers to make referencing them easier later on. (note: do not run this command if your images are already labelled 'n.jpg' because it will overwrite some of them

~~~
cd .../JPEGImages
count=1
for file in $PWD/*.jpg
	do
	mv $file $count.jpg
	count=$((count+1))
done
~~~

Important: LabelImg grabs the folder name when writing the xml files and this needs to be VOC2012. We will fix the error that this leads to in the next step.

Run LabelImg. Download a release from https://tzutalin.github.io/labelImg/ then just extract it and run sudo ./labelImg (it segfaults without sudo)

* set autosave on
* set the load and save directories (save should be .../Annotations, load is .../JPEGImages)
* set the default classname to something easy to remember
* press d to move to the next image
* press w to add a box
* Label all examples of the relevant classes in the dataset

From the Annotations dir run
~~~
for file in $PWD/*.xml
	do sed -i 's/>JPEGImages</>VOC2012</g' $file
done
Cd to the JPEGImages dir and run the command
~~~

For use in Lyra directory
~~~
for file in $PWD/*.xml
	do sed -i 's/>JPEGImages</>VOC2012</g' $file
done
~~~

Create trainvalue text set
~~~
ls | grep .jpg | sed "s/.jpg//g" > aeroplane_trainval.txt
cp aeroplane_trainval.txt trainval.txt
mv *.txt ../ImageSets/Main/
~~~
The Pascal VOC type dataset should now be all created. If you messed up any of the folder structure, you will need to change the XML file contents. If you rename any of the JPEG files you will need to change both the aeroplane_trainval.txt and XML file contents.

Best way to save memory is usage os symbolink links. You have to create following two links.
~~~
ln -s ../../object_detection object_detection
ln -s ../../slim slim
~~~

Open bash in models/research and run the following command 'python object_detection/create_pascal_record.py -h' follow the help instructions to create a pascal.record and file from the dataset.

It should look something like this, stf here stands for serrated tussock full-size. You will need to create an output folder (anywhere you like), also use the --set=trainval option.



For my machine
~~~
export PYTHONPATH=$(pwd):$(pwd)/slim

python object_detection/dataset_tools/create_pascal_tf_record.py --data_dir=VOCdevkit --year=VOC2012 --output_path=pascal.record --label_map_path=label.pbtxt --set=trainval
~~~


Download and extract a tensorflow model to use as the training checkpoint, see 'Which model do I use?'.

Set up the model config file, this will be similar to 'faster_rcnn_resnet101_coco.config' which is in /models/research/object_detection/samples/configs'. Copy the relevant one for the model you are using and edit it. You will need to change approximately 5 directories, the rest should be set up correctly. 
Once the two record files have been created check they are > 0 bytes. Then run the script (from .../models/research/) 'python object_detection/train.py -h' and follow the help instructions to train the model. Create an output folder (train_dir) for your model checkpoints to go in.

It should look something like this. Also see 'Which model do I use?'
for my machine

Edit paths in checkpoint to local placem without full path.
Delete all checkpints, graphs and pipeline, if they present.

~~~
python object_detection/legacy/train.py --train_dir=train --pipeline_config_path=faster_rcnn_inception_v2_coco.config
~~~

You can open tensorboard at this point using the following. Generally if the loss in the bash output from the train.py script is dropping, then training is working fine. How long to train for is something you will need to experiment with. Training on 7 serrated tussock images was accurate after about an hour with loss around 0.02, many more images and a longer training time could improve the accuracy. (Click on the link that tensorboard creates to open it in a browser).
~~~
tensorboard --logdir=train
~~~

Let the model train!

Hit CTRL-C when you're happy with the loss value, checkpoints are periodically saved to the train_dir folder
You now have a trained model, the next step is to test it. The easiest way to do this is to use the jupyter notebook provided in the /models/research/object_detection folder.
From the /models/research folder run the following. You must have created the output folder.
My PC

Instead model.ckpt-2540 to whatever model step is created.
~~~
# Remove directory
rm -R output/

# Create graph
python object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix=train/model.ckpt-2540 --output_directory=output
~~~

Use environment of tensorflow
~~~
source activate tensorflow
~~~

Create a 'test' directory and copy over some images which have not been used for training. Experiment with resizing these to see what sort of scales you can detect at, the first step is to resize them to the same size as your training data and look at the results.
From the directory with the jupyer notebook, run

jupyter notebook marulanDetection.ipynb
This will open a browser window with the notebook, click 'Cell>Run All' to run your model (several directories in red will need to be set, also the number of images you want to test). The results will appear at the bottom of the page. 

jupyter notebook marulanDetection.ipynb
You need to set the following, and also remove or comment the code to download the model, because you are using a retrained one.
~~~
PATH_TO_CKPT = '/home/jasper/stf/output/frozen_inference_graph.pb'PATH_TO_LABELS = ('/home/jasper/stf/label.pbtxt')NUM_CLASSES = 1PATH_TO_TEST_IMAGES_DIR = '/home/jasper/stf/test'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'st{}.jpg'.format(i)) for i in range(1, 64) ] #Set the range here to be 1:(number of images in test directory +1)

# You want to increase this to make the output easier to see
IMAGE_SIZE = (12, 8)
~~~

Congratualtions, you now have a trained faster-rcnn model in tensorflow. See 'It doesn't work?' for issues.


Create video directory using symbolinc link
~~~
ln -s ../../video video
~~~

## Which model do I use?
Which model you grab is up to you. There is some guidance on the https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md page. I have used faster_rcnn_resnet_101_coco with no issues, you may need to alter the config files differently if using an alternate model. Out of the box, faster_rcnn_resnet_101 runs at around 0.5Hz on my laptop (GTX860M), with no optimisation.

To set up a model for training on simply click the link on the model zoo page to download it. Move it to somewhere sensible and then extract it so that you have a folder called 'faster_rcnn_resnet101_coco'. You will need to set the path to this model in the .config file.

It doesn't work?
If your object detection is not working at all there are a few things you may try:

Check your pascal.record is not empty. TF will happily train on empty records without any errors.
Are your objects >30x30 pixels?
Test it on one of the training images, if it works here then your dataset may just be too hard for the amount of training data, although the usual culprit is an error in setting up your dataset files.
A good way to learn tensorflow is https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0 which is also much faster to do and trains a classifier (rather than detector).
If your object detection is working badly you may try:

Expanding the dataset
Training for longer
Data augmentation (there are options for this in the config file, or you can do it manually) see https://stackoverflow.com/questions/44906317/what-are-possible-values-for-data-augmentation-options-in-the-tensorflow-object
Consider illumination, were your test images taken at a different time or with a different camera to the training images?
Tweak the bounding box aspect ratios and sizes in the .config file. If you are detecting people (tall and skinny) you could change the default aspect ratios from (0.1, 1.0, 2.0) to (1.0 1.5 2.0) for example. For very small objects try reducing the scales.
