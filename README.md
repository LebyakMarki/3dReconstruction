# Repository with 3D reconstruction implementation.
This repository presents a ready version of 3d reconstruction pipeline with calibration of your camera, getting disparity map and reprojecting points. Code implementation also features two more implementations with c++ threads and mpi.
What is need is:
one camera
openCV chessboard pattern
30 pictures of pattern, from different angles
2 pictures of an object to reconstruct, camera moved only horizontally
Take two pictures of scene, by moving camera for several centimeters only horizontally, do not move vertically. This is done to imitate stereo photos.

### Steps:
1) Print out openCV's chessboard pattern 
<a href="https://www.google.com/url?sa=i&url=https%3A%2F%2Fstackoverflow.com%2Fquestions%2F25233198%2Fopencv-2-4-9-for-python-cannot-find-chessboard-camera-calibration-tutorial&psig=AOvVaw2jzPieQ-iSL9dN0-v9lvF5&ust=1592231916323000&source=images&cd=vfe&ved=0CA0QjhxqFwoTCOjTgaDEgeoCFQAAAAAdAAAAABAD">**Link is here**</a>.
2) Put it on flat surface and make not less than 30 photos of it. Try to make photos accurately, whole pattern should 
be visible on your photos. Put them in `calibration_images` folder.
3) Run program with conf.txt, where `with_calibration` is set to be **true**, and try to make reprojection error as much as 
possible. It should be less than 1. After you got normal reprojection error, the results of your calibration will be 
stored in `CalibrationMatrices.yml` file. You should set `with_calibration` to be **false** to avoid rerunning calibration.
4) Next step you need to take two pictures on same camera, moving it a little bit horizontally, 
to simulate stereo camera pictures. Try not to move camera at all, only horizontal movement! 
Put them in `working_images` folder and name them correctly: **left.jpg** and **right.jpg**.
5) Run program with `find_points` set to be **true** and select algorithm of searching disparity
`(sgbm: true/false)`. I recommend you to use `sgbm`: **true** and `downscale`: **true** to get good results, 
but if you don't want to set all sgbm algorithm parameter, set `sgbm` to **false** and program will 
run bm algorithm.  Watch what you got in `working_images` folder. Have a look at 
`filtered_disparity.jpg` and `undistorted_left.jpg/undistorted_right.jpg` pictures. 
If you got a good disparity (smooth) and without black regions(uncertain regions) you are 
lucky enough. If not, you should tune parameters in `conf.txt`.
6) If you got smooth disparity map, in `conf.txt` set `save_points` to be **true** and after 
running program you will get `points.txt` in **xyzrgb** format. Recommended to put this file in 
Meshlab to watch what you got!

### conf.txt

**`with_calibration`** - "true" if you want to run calibration process, "false" if read configurations 
from CalibrationMatrices.yml file.<br></br>
**`find_points`** - "true" if you want to run disparity map search, "false" if you run calibration 
process.<br></br>
**`save_points`** - "true" if you have smooth disparity and want to get point cloud, "false" 
if you want to avoid getting point cloud.<br></br>
**`visualize`** - not yet implemented, put "false"<br></br>
**`sgbm`** - "true" if SGBM, "false" if BM.<br></br>
**`downscale`** - "true" if downscale image, "false" if not.<br></br>
**`surf`** - "true" if SURF detector, "false" if AKAZE detector.<br></br>
**`maxThreadNumber`** - number of threads.<br></br>
**`lambda`** - parameter defining the amount of regularization during filtering.<br></br>
**`sigma`** - parameter defining how sensitive the filtering process is to source image edges.<br></br>
**`vis_mult`** - disparity map will be multiplied by this value for visualization.<br></br>
**`minHessian`** - how large the output from the Hessian filter must be in order for a point 
to be used as an interest point.<br></br>
**`preFilterCap`** - Truncation value for the prefiltered image pixels.<br></br>
**`disparityRange`** - maximum disparity minus minimum disparity.<br></br>
**`minDisparity`** - minimum possible disparity value.<br></br>
**`uniquenessRatio`** - Margin in percentage by which the best (minimum) computed cost function 
value should "win" the second best value to consider the found match correct. Normally, a 
value within the 5-15 range is good enough.<br></br>
**`windowSize`** - matched block size. It must be an odd number >=1. SGBM = 3, BM = 15.<br></br>
**`smoothP1`** - first parameter controlling the disparity smoothness.<br></br>
**`smoothP2`** - second parameter controlling the disparity smoothness. The larger the values are, 
the smoother the disparity is. P1 is the penalty on the disparity change by plus or minus 1 
between neighbor pixels. P2 is the penalty on the disparity change by more than 1 between 
neighbor pixels.<br></br>
**`disparityMaxDiff`** - Maximum allowed difference (in integer pixel units) in the left-right 
disparity check. Set it to a non-positive value to disable the check.<br></br>
**`speckleRange`** - truncation value for the prefiltered image pixels. <br></br>
**`speckleWindowSize`** - maximum size of smooth disparity regions to consider their noise speckles 
and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 
50-200 range.<br></br>

### Dependencies:
It is not recommended to visualize point clouds in our visualizer, because it is not efficient and ready for current moment.
```
OpenCV
OpenGL (for visualization)
GLUT (for visualization)
GLM (for visualization)
Threads
Boost (for MPI)
```

### Build && Run :
```
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release  -G"Unix Makefiles" ..
$ make
```
Then, to run program
```
$ ./reconstruct
```
To run parallel version 
```
$ ./reconstruct_t
```

To run mpi vers, you should **uncomment** `# for mpi` lines (5, 6, 7, 34, 35 lines) and **rebuild** project.
Run mpi like this
```
$ mpirun -np [number of processes] ./mpi
```

## Examples
#### Left photo :

![](README_EXAMPLES/left.jpg?raw=true)

#### Right photo :

![](README_EXAMPLES/right.jpg?raw=true)

#### 3D :

![](README_EXAMPLES/example.png?raw=true)

