---
layout: post
title: Stereo Vision Summary
category: ComputerVision
header-style: text
catalog: true
tags: 
    - 2017
    - ComputerVision
    - OpenCV
    - C++
---

# 1.Stereo Vision Summary

* The source code is in my [github](https://github.com/Donche/StereoVision). There are several implementations of stereo vision that have been done (not Feature Matching yet. Although feature point matching is more accurate, only a few dozen of them are often available after filtering. If the edge of the environment are not obvious, there may even be less. The matrices R and T are very unreliable in this way. So I just stopped here) The accuracy of Block Matching is quite good. In general, as long as the calibration is fine, after a slight adjustment of nDisparities and minDisparity, the code can be used then.    

* In fact, there are enough blogs about stereo vision. I have also referred to a lot of blogs[[1]](#1)[[2]](#2), so I'm not going to review the basic knowledges.         

* This blog will focuse on the implementation of OpenCV for stereo vision. The basics involved can be found in many books or tech blogs.      

*------------8.24 update：-------------*      
Can't help but complain about OpenCV bugs (perhaps not bugs but super inconvenient features):      
* When opening the camera, it must be opened by the sequence number from high to low, otherwise it cannot work.
* Some of OpenCV's built-in functions will cause some very strange errors, such as the vector can not be released. And this error will suddenly appear without warning. The code that was previously available to run and recompile may also cause problems as well. At this time, you can use the Mat type of opencv to replace vector.
>When OpenCV with c++ is used, it may happen this runtime error problem when feeding some Container (eg:vector) to the opencv built-in function. So it is good to use Mat datatype to save the output received by the opencv built-in function and then transfer them to whatever datatype you need.      

* OpenCV's camera input stream is cached, and the default cache is one frame. If the processing speed between each frame is fast, it's okay to ignore it. However if the processing time between each frame is like 5 seconds, a ten second delay of the result will occur. The simplest solution is to read one more frame at the time of getting image.      


# 2. Two methods of stereo vision

The current stereo vision matching algorithms are Block Matching, Feature Matching and Phase differencing algorithms. We only consider the first two.       
The former is to pre-calibrate the cameras. After image rectification, the matching is used to obtain the disparity maps. Then the three-dimensional reconstruction is performed using the extrinsic parameters. The result is very good and the speed is also fast. The disadvantage is that it must be calibrated beforehand. If the relative position of the camera is slightly changed while operating, you have to re-calibrate it (for example, I used two cameras to freely set the table, and it's a mess).       

The latter is to take the feature points of the left and right images respectively, calculate the descriptors, and then match and filter them to obtain the one-to-one corresponding feature points and then get the camera's external parameters. Then the external parameters are used to re-construct the environment. The three-dimensional reconstruction is slow and the external parameters unstable. As a result, only sparse fields can be obtained.     

Therefore, I mainly use Block Matching for three-dimensional reconstruction.    

# 3. Block Matching
Main ideas: 1. Calibration  2. Disparity map    3. 3D reconstruction    
## 3.1 Calibration
The image used for calibration can theoretically use any image. Generally speaking, the standard images used for calibration are chessboard, circle grid and asymmetric circles grid. It should be noted that for the chessboard, the corresponding corners detected in both directions is the value of the black and white squares minus one, and the square size is the distance between the two corners. For the circles Grid, the parameters length and width are the number of circles in two directions, square size is the distance between the nearest two circles' centers; the square size of asymmetric circles grid is half of the distance between two circles in the horizontal direction.     


Zhang's method is used for calibration. So in general ten calibration images with different postures are enough. I usually use 20-30, and the re-projection error can be maintained at about 0.08. Another thing to note is that if you use the circles grid, you shouldn't make the calibration plate close to the camera, or the error will be very large.     

### 3.1.1 Intrinsic parameters
First, take a look at the parameters required for calibration:
```c++
CV_EXPORTS_W double calibrateCamera( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints, Size imageSize,
                                     InputOutputArray cameraMatrix, InputOutputArray distCoeffs,
                                     OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                                     int flags = 0, TermCriteria criteria = TermCriteria(
                                        TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON) );
```

* objectPoints: has been explained in opencv manual:    
>In the new interface it is a vector of vectors of calibration pattern points in the calibration pattern coordinate space (e.g. std::vector<std::vector<cv::Vec3f>>).    

So it is the coordinate of the corner point that needs to be detected. The previously set height and width and square size are for this purpose. Generating this parameter is very simple, opencv source code can be referenced:    

```c++
void StereoCalibration::calcBoardCornerPositions( vector<Point3f>& corners)
{
	corners.clear();

	for (int i = 0; i < boardSize.height; ++i)
		for (int j = 0; j < boardSize.width; ++j)
			corners.push_back(Point3f(float(j*squareSize), float(i*squareSize), 0));
}
```
   
* imagePoints:it is a vector of vectors of the projections of calibration pattern points。It is the coordinates of the points in the image. This can be found with ```findChessboardCorners``` or ```findCirclesGrid```.
* imageSize：Image size
* cameraMatrix： Output camera intrinsic parameters
* distCoeffs： Output camera distortion matrix
* rvecs(tvecs)：Output vector of rotation(translation) vectors estimated for each pattern view. It can be used to calculate the reprojection error.
* flag：some calibration methods settings, such as```CV_CALIB_FIX_K4```。

### 3.1.2 Extrinsic parameters
After getting the intrinsic parameters, the extrinsic parameters is much easier to calculate. Look at the function first:      
```c++
CV_EXPORTS_W double stereoCalibrate( InputArrayOfArrays objectPoints,
                                     InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
                                     InputOutputArray cameraMatrix1, InputOutputArray distCoeffs1,
                                     InputOutputArray cameraMatrix2, InputOutputArray distCoeffs2,
                                     Size imageSize, OutputArray R,OutputArray T, OutputArray E, OutputArray F,
                                     int flags = CALIB_FIX_INTRINSIC,
                                     TermCriteria criteria = TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6) );
```
* objectPoints：the same as before
* imagePoints：as before
* cameraMatrix、distCoeffs1：as before
* imageSize：as before
* R：Rotation matrix between two cameras
* T：Translating matrix between two cameras
* E：Essential matrix
* F：Foundation matrix
* flag：as before

### 3.1.3 Rectification
  
For matching purposes, four maps are calculated to remap the image to obtain rectified images:     
```c++
CV_EXPORTS_W void stereoRectify( InputArray cameraMatrix1, InputArray distCoeffs1,
                                 InputArray cameraMatrix2, InputArray distCoeffs2,
                                 Size imageSize, InputArray R, InputArray T,
                                 OutputArray R1, OutputArray R2,
                                 OutputArray P1, OutputArray P2,
                                 OutputArray Q, int flags = CALIB_ZERO_DISPARITY,
                                 double alpha = -1, Size newImageSize = Size(),
                                 CV_OUT Rect* validPixROI1 = 0, CV_OUT Rect* validPixROI2 = 0 );

CV_EXPORTS_W void initUndistortRectifyMap( InputArray cameraMatrix, InputArray distCoeffs,
                        InputArray R, InputArray newCameraMatrix,
                        Size size, int m1type, OutputArray map1, OutputArray map2 );
```

## 3.2 Disparity map    
The Block Matching algorithms provided by opencv mainly include BM, SGBM and GC.      

For opencv3, StereoBM is an inherited class by StereoMatcher. Both sides have some parameter settings. Because it is inherited, it is set directly in the StereoBM function.     
```c++
//StereoMatcher
CV_WRAP virtual void compute( InputArray left, InputArray right,
                              OutputArray disparity ) = 0;
CV_WRAP virtual void setMinDisparity(int minDisparity) = 0;
CV_WRAP virtual void setNumDisparities(int numDisparities) = 0;
CV_WRAP virtual void setBlockSize(int blockSize) = 0;
CV_WRAP virtual void setSpeckleWindowSize(int speckleWindowSize) = 0;
CV_WRAP virtual void setSpeckleRange(int speckleRange) = 0;
CV_WRAP virtual void setDisp12MaxDiff(int disp12MaxDiff) = 0;
//StereoBM
CV_WRAP virtual void setPreFilterType(int preFilterType) = 0;
CV_WRAP virtual void setPreFilterSize(int preFilterSize) = 0;
CV_WRAP virtual void setPreFilterCap(int preFilterCap) = 0;
CV_WRAP virtual void setTextureThreshold(int textureThreshold) = 0;
CV_WRAP virtual void setUniquenessRatio(int uniquenessRatio) = 0;
CV_WRAP virtual void setSmallerBlockSize(int blockSize) = 0;
CV_WRAP virtual void setROI1(Rect roi1) = 0;
CV_WRAP virtual void setROI2(Rect roi2) = 0;
```
There are too many parameters, but only the SADWindowSize, numberOfDisparities, and uniquenessRatio are more important. The others can take a little less time. My final choice of parameters is:

```c++
int preFilterSize = 13;
int preFilterCap = 24;
int minDisparity = -16;

int ndisparities = 8 * 16;
int SADWindowSize = 29;

int textureThreshold = 507;
int uniquenessRatio = 8;
int speckleWindowSize = 67;
int speckleRange = 14;
```


## 3.3 3D reconstruction
It is implemented with reprojectImageTo3D:

```c++
CV_EXPORTS_W void reprojectImageTo3D( InputArray disparity,
                                      OutputArray 3dImage, InputArray Q,
                                      bool handleMissingValues = false,
                                      int ddepth = -1 );
```
The resulting 3D points can be exported to a file to be drawn to MatLab, or displayed in real time using OpenGL.    


# 4. Feature Matching
Main ideas 1. Find feature points 2. Match and filter    
## 4.1 feature points
Commonly used methods are SURF, ORB and SIFT. In general, ORB>SURF>SIFT for calculation speed, and the computational complexity is reversed. Ordinarily, the use of ORB is a good choice. After filtering, it will still leave a lot of matched features.    

The implementation of feature points in OpenCV3 has changed quite a lot. We need to use Feature2D to calculate key points and descriptors, and then use BFMatcher to match. It should be noted that the ORB uses the Hamming distance as a measure to describe the sub-distance, and other feature points can use different norms (such as NORM_L2).   

```c++
Ptr<BFMatcher> matcher;
Ptr<Feature2D> feature_l, feature_r;

matcher = BFMatcher::create(NORM_L2);
feature_l = xfeatures2d::SURF::create();

feature_l->detectAndCompute(imgLeft, noArray(), key_points_l, descriptor_l);
feature_r->detectAndCompute(imgRight, noArray(), key_points_r, descriptor_r);
```

## 4.2 Matching and filtering
There are mainly three types of matching, KNNMatch, radiusMatch and BFMatch.    
* KNNMatch：Finds the k best matches for each descriptor from a query set.
* radiusMatch：For each query descriptor, finds the training descriptors not farther than the specified distance.
* match：Finds the best match for each descriptor from a query set.

It's pretty easy to understand. I used KnnMatch and match the best of the first two, and then filtered.    

Mainly two methods are used for filtering, one is ratioTest and the other is symmetryTest.   
* ratioTest：
>"Therefore, for each feature point, we have two candidate matches in the other view. These are the two best ones based on the distance between their descriptors. If this measured distance is very low for the best match, and much larger for the second best match, we can safely accept the first match as a good one since it is unambiguously the best choice. Reciprocally, if the two best matches are relatively close in distance, then there exists a possibility that we make an error if we select one or the other. In this case, we should reject both matches."   

So I took 0.7 as the ratio and 50 as the maximum distance to get the best match for each feature point. See source code for details.

* symmetryTest：Perform two-way verification, if a bunch of feature points in the left and right images match each other, it is considered as a valid match.

The above two methods can be combined to get very good results, with few false matches and slow speeds.
 


*Reference Materials*
1. <span id="1"></span> http://blog.csdn.net/chenyusiyuan/article/details/5961769
2. <span id="2"></span> http://blog.csdn.net/wangyaninglm/article/details/52142217