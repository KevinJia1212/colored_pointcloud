colored_pointcloud package

Install: 
cd to the src folder of a ros workspace, git clone this package and catkin_make this workspace.

Preparation: 
calibrate your camera, and your lidar-camera system,
then write the intrinsic matrix, distortion coefficients and the extrinsic matrix to config/calib_result.yaml,
finally change the camera_topic and lidar_topic to fit your own system. 

Usage: 
<launch your camera and lidar nodes>
roslaunch colored_pointcloud colored_pointcloud161.launch 
