#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <boost/mpi.hpp>
#include <boost/multi_array.hpp>
#include <utility>
#include <cassert>
#include <vector>
#include "helping_functions/config_parser.h"


void process_images(cv::Size &chessboard_size, std::vector<std::vector<cv::Point2f>>& allFoundCorners,
                    std::vector<cv::String>& filenames, int start, int end) {
//    Function to iterate through images
//    taken for camera calibration

    std::vector<cv::Point2f> pointBuffer;
    for (int i = start; i < end; i++) {
        const auto & filename = filenames[i];
        if (filename != "../calibration_images/.DS_Store"){
            cv::Mat im = cv::imread(filename);
            bool found = findChessboardCorners(im, chessboard_size, pointBuffer,
                                               cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
            if (!found) {
                std::cout << filename << std::endl;
            }
            if (found) {
                cv::Mat gray_im;
                cvtColor(im, gray_im, cv::COLOR_BGR2GRAY);
                cv::TermCriteria criteria = cv::TermCriteria( cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
                                                              30,0.1 );
                cv::Size winSize = cv::Size( 11, 11);
                cv::Size zeroZone = cv::Size( -1, -1 );
                //cornerSubPix is the algorithm focused on relocating the points. it receives the image, the corners
                // a window size, zeroZone and the actual criteria. The window size is the search area.
                cornerSubPix(gray_im, pointBuffer, winSize, zeroZone, criteria );
                allFoundCorners.push_back(pointBuffer);
            }
        }
    }
}


void undistort(cv::Mat& cameraMatrix, cv::Mat& distortionCoefficients) {
    cv::Mat im1 = cv::imread("../working_images/left.jpg");
    cv::Mat im2 = cv::imread("../working_images/right.jpg");
    if (im1.size().width != im2.size().width || im1.size().height != im2.size().height){
        std::cerr << "Images size does not match!"<< std::endl;
    }
    int width = im1.size().width;
    int height = im1.size().height;
    cv::Mat new_camera_matrix = getOptimalNewCameraMatrix(cameraMatrix, distortionCoefficients,
                                                          {width, height}, 1, {width, height}, nullptr);
    cv::Mat im1_udist, im2_undist;
    undistort(im1, im1_udist, cameraMatrix, distortionCoefficients, new_camera_matrix);
    undistort(im2, im2_undist, cameraMatrix, distortionCoefficients, new_camera_matrix);
    imwrite("../working_images/undistorted_left.jpg", im1);
    imwrite("../working_images/undistorted_right.jpg", im2);
}


void disparity(Configuration& configuration) {
    cv::Mat filtered_disp, conf_map;
    cv::Mat im1 = cv::imread("../working_images/undistorted_left.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat im2 = cv::imread("../working_images/undistorted_right.jpg", cv::IMREAD_GRAYSCALE);
    conf_map = cv::Mat(im1.rows,im2.cols,CV_8U);
    conf_map = cv::Scalar(255);

    double lambda = configuration.lambda;
    double sigma = configuration.sigma;
    double vis_mult = configuration.vis_mult;
    int preFilterCap = configuration.preFilterCap, disparityRange = configuration.disparityRange,
            minDisparity = configuration.minDisparity, uniquenessRatio = configuration.uniquenessRatio,
            windowSize = configuration.windowSize, smoothP1 = configuration.smoothP1 * windowSize * windowSize,
            smoothP2 = configuration.smoothP2 * windowSize * windowSize, disparityMaxDiff = configuration.disparityMaxDiff,
            speckleRange = configuration.speckleRange, speckleWindowSize = configuration.speckleWindowSize;
//    if bm window size = 15
    bool mode = cv::StereoSGBM::MODE_SGBM_3WAY;
    cv::Mat left_disparity, right_disparity ,norm_disparity;
    if (configuration.downscale) {
        disparityRange /= 2;
        resize(im1 ,im1 ,cv::Size(),0.5,0.5, cv::INTER_LINEAR_EXACT);
        resize(im2, im2, cv::Size(),0.5,0.5, cv::INTER_LINEAR_EXACT);
    }
    cv::Rect ROI;
    if (configuration.sgbm){
        cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create(minDisparity,
                                                                      disparityRange * 16, windowSize, smoothP1, smoothP2, disparityMaxDiff, preFilterCap,
                                                                      uniquenessRatio, speckleWindowSize, speckleRange, mode);
        cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);
        left_matcher->compute(im1, im2, left_disparity);
        right_matcher ->compute(im2, im1, right_disparity);

        cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter = cv::ximgproc::createDisparityWLSFilterGeneric(left_matcher);
        wls_filter->setLambda(lambda);
        wls_filter->setSigmaColor(sigma);
        wls_filter->filter(left_disparity, im1, filtered_disp, right_disparity);
        conf_map = wls_filter->getConfidenceMap();
        ROI = wls_filter->getROI();
    }
    else if (!configuration.sgbm){
        cv::Ptr<cv::StereoBM> left_matcher = cv::StereoBM::create(disparityRange * 16, windowSize);
        cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter = cv::ximgproc::createDisparityWLSFilterGeneric(left_matcher);
        cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(left_matcher);
        left_matcher->compute(im1, im2, left_disparity);
        right_matcher ->compute(im2, im1, right_disparity);
        wls_filter->setLambda(lambda);
        wls_filter->setSigmaColor(sigma);
        wls_filter->filter(left_disparity, im1, filtered_disp, right_disparity);
        conf_map = wls_filter->getConfidenceMap();
        ROI = wls_filter->getROI();
    }
    if (configuration.downscale) {
        resize(left_disparity, left_disparity, cv::Size(),2.0,2.0, cv::INTER_LINEAR_EXACT);
        left_disparity = left_disparity*2.0;
        ROI = cv::Rect(ROI.x*2,ROI.y*2,ROI.width*2,ROI.height*2);
    }

    cv::Mat raw_disp_vis, filtered_disp_vis;
    cv::ximgproc::getDisparityVis(left_disparity,raw_disp_vis,vis_mult);
    cv::imwrite("../working_images/raw_disparity.jpg", raw_disp_vis);

    normalize(filtered_disp_vis, filtered_disp_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::ximgproc::getDisparityVis(filtered_disp,filtered_disp_vis,vis_mult);
    cv::imwrite("../working_images/filtered_disparity.jpg", filtered_disp_vis);

}


void findRTQ(cv::Mat &Q, cv::Mat &camera_matrix, cv::Mat &distortion, Configuration& configuration) {
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat im1 = cv::imread("../working_images/undistorted_left.jpg");
    cv::Mat im2 = cv::imread("../working_images/undistorted_right.jpg");
    cv::Mat desc1, desc2;
    if (configuration.surf) {
        cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(configuration.minHessian);
        detector->detectAndCompute( im1, cv::noArray(), keypoints1, desc1 );
        detector->detectAndCompute( im2, cv::noArray(), keypoints2, desc2 );
    }
    else if (!configuration.surf){
        cv::Ptr<cv::AKAZE> akaze_detctor = cv::AKAZE::create();
        akaze_detctor -> detectAndCompute(im1, cv::noArray(), keypoints1, desc1);
        akaze_detctor -> detectAndCompute(im2, cv::noArray(), keypoints2, desc2);
    }

    auto* matcher = new cv::BFMatcher(cv::NORM_L2, false);
    std::vector< std::vector<cv::DMatch> > matches_2nn_12, matches_2nn_21;
    matcher->knnMatch( desc1, desc2, matches_2nn_12, 2 );
    matcher->knnMatch( desc2, desc1, matches_2nn_21, 2 );
    std::vector<cv::Point2f> selected_points1, selected_points2;
    const double ratio = 0.8;
    for(auto & i : matches_2nn_12) { // i is queryIdx
        if (i[0].distance / i[1].distance < ratio
            and
            matches_2nn_21[i[0].trainIdx][0].distance
            / matches_2nn_21[i[0].trainIdx][1].distance < ratio) {
            if (matches_2nn_21[i[0].trainIdx][0].trainIdx
                == i[0].queryIdx) {
                selected_points1.push_back(keypoints1[i[0].queryIdx].pt);
                selected_points2.push_back(
                        keypoints2[matches_2nn_21[i[0].trainIdx][0].queryIdx].pt
                );
            }
        }
    }
    cv::Mat Kd, mask;
    camera_matrix.convertTo(Kd, CV_64F);
    cv::Mat E = cv::findEssentialMat(selected_points1, selected_points2, Kd.at<double>(0,0),
                                     cv::Point2d(im1.cols/2., im1.rows/2.),
                                     cv::RANSAC, 0.999, 1.0, mask);

    std::vector<cv::Point2f> inlier_match_points1, inlier_match_points2;
    for(int i = 0; i < mask.rows; i++) {
        if(mask.at<unsigned char>(i)){
            inlier_match_points1.push_back(selected_points1[i]);
            inlier_match_points2.push_back(selected_points2[i]);
        }
    }
    mask.release();

    cv::Mat R, t, R1, R2, P1, P2;
    cv::recoverPose(E, inlier_match_points1, inlier_match_points2, R, t, Kd.at<double>(0,0),
                    cv::Point2d(im1.cols/2., im1.rows/2.), mask);
    cv::stereoRectify(camera_matrix, distortion, camera_matrix, distortion, im1.size(), R, t, R1, R2, P1, P2, Q,
                      cv::CALIB_ZERO_DISPARITY, 1, im1.size());
    cv::Mat map1x, map1y, map2x, map2y;
    initUndistortRectifyMap( camera_matrix, distortion, R1, P1, im1.size(), CV_32FC1, map1x, map1y );
    initUndistortRectifyMap( camera_matrix, distortion, R2, P2, im1.size(), CV_32FC1, map2x, map2y );
    cv::Mat im1_remap;
    remap( im1, im1_remap, map2x, map2y, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar());
    cv::imwrite("../working_images/left_remap.jpg", im1_remap);
}


void save(const cv::Mat& image3D, const std::string& fileName, Configuration& configuration, cv::Mat &im1)
{
    std::ofstream outFile(fileName);
    if (!outFile.is_open())
    {
        std::cerr << "ERROR: Could not open " << fileName << std::endl;
        return;
    }
    for (int i = 0; i < image3D.rows; i++)
    {
        const auto* image3D_ptr = image3D.ptr<cv::Vec3f>(i);
        for (int j = 0; j < image3D.cols; j++)
        {
            if (std::isfinite(image3D_ptr[j][0]) && std::isfinite(image3D_ptr[j][1]
                                                                  && std::isfinite(image3D_ptr[j][2]))){
                outFile << image3D_ptr[j][0] << " " << image3D_ptr[j][1] << " " << image3D_ptr[j][2] << " " <<
                        static_cast<unsigned>(im1.at<uchar>(i,j)) << " " << static_cast<unsigned>(im1.at<uchar>(i,j)) << " "
                        << static_cast<unsigned>(im1.at<uchar>(i,j)) << std::endl;
            }
        }
    }
    outFile.close();
}


void point_cloud(cv::Mat &Q, Configuration& configuration) {
    double min, max;
    cv::Mat image3DOCV, colors,  scaledDisparityMap;
    cv::Mat im1 = cv::imread("../working_images/undistorted_left.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat disp = cv::imread("../working_images/filtered_disparity.jpg", cv::IMREAD_GRAYSCALE);
    minMaxIdx(disp, &min, &max);
    convertScaleAbs( disp, scaledDisparityMap, 255 / ( max - min ) );
    disp.convertTo( disp, CV_32FC1 );
//    reprojectImageTo3D(disp, image3DOCV, Q, true, CV_32F);
    reprojectImageTo3D(disp, image3DOCV, Q, true, -1);

    cv::Mat dst, thresholded_disp, pointcloud_tresh, color_tresh;
    cv::adaptiveThreshold(scaledDisparityMap, thresholded_disp, 255,cv::ADAPTIVE_THRESH_GAUSSIAN_C,cv::THRESH_BINARY,3,1);
    resize( thresholded_disp, dst, cv::Size( image3DOCV.cols, image3DOCV.rows ), 1, 1, cv::INTER_LINEAR_EXACT );
    cv::imwrite("../working_images/disparity_thresh.jpg", dst);
    image3DOCV.copyTo( pointcloud_tresh, thresholded_disp );
    if (configuration.save_points){
        if (pointcloud_tresh.size() != im1.size()){
            cv::resize(im1, im1, cv::Size(pointcloud_tresh.cols, pointcloud_tresh.rows));
            save(pointcloud_tresh, "../points.txt", configuration, im1);
        }
    }
}

void write_yml_file(cv::Mat& cameraMatrix, cv::Mat& distortionCoefficient, std::string file_path) {
    cv::FileStorage fs(file_path, cv::FileStorage::WRITE);
    fs << "cameraMatrix" << cameraMatrix;
    fs << "distortionCoefficient" << distortionCoefficient;
    fs.release();
}

void read_yml_file(cv::Mat& cameraMatrix, cv::Mat& distortionCoefficient, std::string file_path) {
    cv::FileStorage fs(file_path, cv::FileStorage::READ);
    fs["cameraMatrix"] >> cameraMatrix;
    fs["distortionCoefficient"] >> distortionCoefficient;
    fs.release();
}

void point2f_to_vector(std::vector<std::vector<cv::Point2f>>& old,
                       boost::multi_array<double, 3>& arr) {
    for (int i = 0; i < old.size(); ++i) {
        for (int j = 0; j < old[0].size(); ++j) {
            arr[i][j][0] = old[i][j].x;
            arr[i][j][1] = old[i][j].y;
        }
    }
}
void vector_to_point2f(boost::multi_array<double, 3>& arr,
                       std::vector <std::vector<cv::Point2f>>& new_arr) {
    for (int i = 0; i < arr.shape()[0]; ++i) {
        std::vector<cv::Point2f> pointBuffer;
        for (int j = 0; j < arr.shape()[1]; ++j) {
            cv::Point2f pt(arr[i][j][0], arr[i][j][1]);
            pointBuffer.push_back(pt);
        }
        new_arr.push_back(pointBuffer);
    }
}

int main(int argc, char * argv[]) {
    boost::mpi::environment env{argc, argv};
    boost::mpi::communicator world;

    if (world.size() < 2) { throw std::runtime_error("Incorrect amount of processes! Try 2 or more."); }
    const int child_processes_number = world.size() - 1;

    // conf read
    Configuration configuration{};

    // Reading configuration
    std::string config_file = "../conf.txt";
    std::ifstream config_stream(config_file);
    if(!config_stream.is_open()){
        throw std::runtime_error("Failed to open configurations file " + config_file);
    }
    configuration = read_configuration(config_stream);

    cv::Size chessboardDimensions = cv::Size(6, 9);
    std::vector<cv::String> filenames;
    std::string path_to_directory = "../calibration_images";
    cv::glob(path_to_directory, filenames);
    if(world.rank() == 0) {

        cv::Mat cameraMatrix, distortionCoefficient;

        //calibration process
        if (configuration.with_calibration){
            float calibrationSquareDimension = 0.029f;

            std::vector <std::vector<cv::Point2f>> checkerboardImageSpacePoints;

            // process_images
            // send
            int part = filenames.size()/child_processes_number;
            int start = 0, end = part;
            for(int i = 1; i <= child_processes_number; i++) {
                end = (i == child_processes_number) ? filenames.size() : (i * part);
                int bordes[2] = {start, end};
                world.send(i, 0, bordes);
                start = end;
            }

            // recv
            for (int j = 1; j <= child_processes_number; j++) {
                size_t sizes[2];
                world.recv(j, 0, sizes);

                boost::multi_array<double, 3> img_points(boost::extents[sizes[0]][sizes[1]][2]);
                world.recv(j, 0, &img_points[0][0][0], sizes[0]*sizes[1]*2);

                std::vector <std::vector<cv::Point2f>> temp;
                vector_to_point2f(img_points, temp);
                checkerboardImageSpacePoints.insert(checkerboardImageSpacePoints.end(), temp.begin(), temp.end());
            }

            // end communication

            std::vector <std::vector<cv::Point3f>> worldSpaceCornerPoints(1);
            for (int i = 0; i < chessboardDimensions.height; i++) {
                for (int j = 0; j < chessboardDimensions.width; j++) {
                    worldSpaceCornerPoints[0].emplace_back((float)j * calibrationSquareDimension, (float)i * calibrationSquareDimension, 0.0f);
                }
            }
            worldSpaceCornerPoints.resize(checkerboardImageSpacePoints.size(), worldSpaceCornerPoints[0]);

            //rotation and translation vectors (rVectors, tVectors)
            std::vector <cv::Mat> rVectors, tVectors;
            double res = calibrateCamera(worldSpaceCornerPoints, checkerboardImageSpacePoints, chessboardDimensions, cameraMatrix,
                                         distortionCoefficient, rVectors, tVectors,
                                         (cv::CALIB_FIX_K4 | cv::CALIB_FIX_K5) + cv::CALIB_FIX_INTRINSIC);
            std::cout << "Reprojection Error (from calibrateCamera): " << res << std::endl;


            write_yml_file(cameraMatrix, distortionCoefficient, "../CalibrationMatrices.yml");
        } else {
            read_yml_file(cameraMatrix, distortionCoefficient, "../CalibrationMatrices.yml");
        }
        if (configuration.find_points){
            cv::Mat Q;
            undistort(cameraMatrix, distortionCoefficient);
            disparity(configuration);
            findRTQ(Q, cameraMatrix, distortionCoefficient, configuration);
            point_cloud(Q, configuration);
        }
        //end
    } else {
        if (configuration.with_calibration) {
            int borders[2];
            world.recv(0, 0, borders);
            std::vector <std::vector<cv::Point2f>> checkerboardImageSpacePoints;

            process_images(chessboardDimensions, checkerboardImageSpacePoints, filenames, borders[0], borders[1]);

            size_t sizes[2] = {checkerboardImageSpacePoints.size(), checkerboardImageSpacePoints[0].size()};
            world.send(0, 0, sizes);

            boost::multi_array<double, 3> img_points(boost::extents[checkerboardImageSpacePoints.size()]
                                                     [checkerboardImageSpacePoints[0].size()][2]);
            point2f_to_vector(checkerboardImageSpacePoints, img_points);
            world.send(0, 0, &img_points[0][0][0], sizes[0] * sizes[1] * 2);
        }
    }
    return 0;
}

