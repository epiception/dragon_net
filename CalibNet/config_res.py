import numpy as np

depth_img_params = dict(
	IMG_HT = 375,
	IMG_WDT = 1242
	)

color_img_params = dict(
	IMG_HT = 373,
	IMG_WDT = 1240
	)


camera_params = dict(
	fx = 7.215377e+02,
    fy = 7.215377e+02,
    cx = 6.095593e+02,
    cy = 1.728540e+02,

    cam_transform_02 =  np.array([1.0, 0.0, 0.0, (-4.485728e+01)/7.215377e+02,
	                              0.0, 1.0, 0.0, (-2.163791e-01)/7.215377e+02,
	                              0.0, 0.0, 1.0, -2.745884e-03,
	                              0.0, 0.0, 0.0, 1.0]).reshape(4,4),

	cam_transform_03 = np.array([1.0, 0.0, 0.0, (-3.395242e+02)/7.215377e+02,
	                              0.0, 1.0, 0.0, (2.199936e+00)/7.215377e+02,
	                              0.0, 0.0, 1.0, 2.729905e-03,
	                              0.0, 0.0, 0.0, 1.0]).reshape(4,4)
	)

net_params = dict(
	batch_size = 16,
	total_frames = 30000,
	total_frames_train = 24000,
	total_frames_validation = 6000,
	partition_limit = 1200,
	epochs = 40,
	learning_rate = 1e-4,
	beta1 = 0.9,
	resnet_path = "../../Extrinsic_Calibration_2/parameters.json",
	alpha_const = 1.0,
	beta_const = 1.0,
	current_epoch = 0

	)
