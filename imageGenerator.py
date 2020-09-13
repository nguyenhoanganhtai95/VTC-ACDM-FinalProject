from tensorflow.keras.preprocessing.image import ImageDataGenerator
trainlib='./Temp/train_data'

def generator():
	data_gen_args = dict(rescale = 1./255,
		            # featurewise_center=True,
		            #  featurewise_std_normalization=True,
		            rotation_range=0.2,
		            width_shift_range=0.05,
		            height_shift_range=0.05,
		            shear_range=0.05,
		            zoom_range=0.05,
		            horizontal_flip=True,
		            fill_mode='nearest',
		            validation_split=0.2)
	image_datagen = ImageDataGenerator(**data_gen_args)
	mask_datagen = ImageDataGenerator(**data_gen_args)
	image_generator = image_datagen.flow_from_directory(
	    trainlib,
	    target_size=(112, 112),
	    class_mode=None,
	    batch_size=4,
	    seed=1)
	return image_generator

