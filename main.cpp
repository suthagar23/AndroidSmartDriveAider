

/*
 * OpenCV
 * Real Time Object Recognition using SURF
 *
 *  Created on: Nov 15, 2013
 *      Author: K.Suthagar
 */

//Include statements
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

//Name spaces used
using namespace cv;
using namespace std;

int main() {
    //turn performance analysis functions on if testing = true
    bool testing = false;
    double t; //timing variable

    //load training image
    Mat object = imread("1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    if (!object.data) {
        cout << "Can't open image";
        return -1;
    }



    Mat object2 = imread("2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    if (!object2.data) {
        cout << "Can't open image";
        return -1;
    }

    Mat object3 = imread("3.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    if (!object3.data) {
        cout << "Can't open image";
        return -1;
    }


    Mat object4 = imread("4.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    if (!object4.data) {
        cout << "Can't open image";
        return -1;
    }


    //SURF Detector, and descriptor parameters
    int minHess = 3000;
    vector<KeyPoint> kpObject, kpImage;
    Mat desObject, desImage;


    //Performance measures calculations for report
    if (testing) {
        cout << object.rows << " " << object.cols << endl;

        //calculate integral image
        Mat iObject;
        integral(object, iObject);
        imshow("Good Matches", iObject);
        imwrite("trainedImage.jpg", iObject);
        cvWaitKey(0);

        //calculate number of interest points, computation time as f(minHess)
        int minHessVector[] = {100, 500, 1000, 1500, 2000, 2500, 3000,
            3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500,
            8000, 8500, 9000, 9500, 10000};
        int minH;
        std::ofstream file;
        file.open("TimingC.csv", std::ofstream::out);
        for (int i = 0; i < 20; i++) {
            minH = minHessVector[i];
            t = (double) getTickCount();
            SurfFeatureDetector detector(minH);
            detector.detect(object, kpObject);
            t = ((double) getTickCount() - t) / getTickFrequency();
            file << minHess << "," << kpObject.size() << "," << t << ",";
            cout << t << " " << kpObject.size() << " " << desObject.size() << endl;

            t = (double) getTickCount();
            SurfDescriptorExtractor extractor;
            extractor.compute(object, kpObject, desObject);
            t = ((double) getTickCount() - t) / getTickFrequency();
            file << t << endl;
        }
        file.close();

        //Display keypoints on training image
        Mat interestPointObject = object;
        for (unsigned int i = 0; i < kpObject.size(); i++) {
            if (kpObject[i].octave) {
                circle(interestPointObject, kpObject[i].pt, kpObject[i].size, 0);
                string octaveS;
                switch (kpObject[i].octave) {
                    case 0:
                        octaveS = "0";
                        break;
                    case 1:
                        octaveS = '1';
                        break;
                    case 2:
                        octaveS = '2';
                        break;
                    default:
                        break;

                }
                putText(interestPointObject, octaveS, kpObject[i].pt,
                        FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(0, 0, 250), 1, CV_AA);
            }

        }
        imshow("Good Matches", interestPointObject);
        imwrite("bookIP2.jpg", interestPointObject);
        cvWaitKey(0);
    }


    //SURF Detector, and descriptor parameters, match object initialization
    minHess = 2500;
    SurfFeatureDetector detector(minHess);
    detector.detect(object, kpObject);
    SurfDescriptorExtractor extractor;
    extractor.compute(object, kpObject, desObject);
    FlannBasedMatcher matcher;

    //Initialize video and display window
    VideoCapture cap(0); //camera 1 is webcam
    if (!cap.isOpened()) return -1;

    //Object corner points for plotting box
    vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0, 0);
    obj_corners[1] = cvPoint(object.cols, 0);
    obj_corners[2] = cvPoint(object.cols, object.rows);
    obj_corners[3] = cvPoint(0, object.rows);

    //video loop
    char escapeKey = 'k';
    double frameCount = 0;
    float thresholdMatchingNN = 0.55;
    unsigned int thresholdGoodMatches = 5;
    unsigned int thresholdGoodMatchesV[] = {4, 5, 6, 7, 8};

    for (int j = 0; j < thresholdGoodMatches; j++) {
        thresholdGoodMatches = thresholdGoodMatchesV[j];
        //thresholdGoodMatches=8;
        cout << thresholdGoodMatches << endl;

        if (true) {
            t = (double) getTickCount();
        }




        vector<KeyPoint> kpObject2, kpImage2;
        Mat desObject2, desImage2;


        //SURF Detector, and descriptor parameters, match object initialization
        minHess = 2000;
        SurfFeatureDetector detector2(minHess);
        detector2.detect(object2, kpObject2);
        SurfDescriptorExtractor extractor2;
        extractor2.compute(object2, kpObject2, desObject2);
        FlannBasedMatcher matcher2;


        //Object corner points for plotting box
        vector<Point2f> obj_corners2(4);
        obj_corners2[0] = cvPoint(0, 0);
        obj_corners2[1] = cvPoint(object2.cols, 0);
        obj_corners2[2] = cvPoint(object2.cols, object2.rows);
        obj_corners2[3] = cvPoint(0, object2.rows);









        vector<KeyPoint> kpObject3, kpImage3;
        Mat desObject3, desImage3;


        //SURF Detector, and descriptor parameters, match object initialization
        minHess = 2000;
        SurfFeatureDetector detector3(minHess);
        detector3.detect(object3, kpObject3);
        SurfDescriptorExtractor extractor3;
        extractor3.compute(object3, kpObject3, desObject3);
        FlannBasedMatcher matcher3;


        //Object corner points for plotting box
        vector<Point2f> obj_corners3(4);
        obj_corners3[0] = cvPoint(0, 0);
        obj_corners3[1] = cvPoint(object3.cols, 0);
        obj_corners3[2] = cvPoint(object3.cols, object3.rows);
        obj_corners3[3] = cvPoint(0, object3.rows);







        vector<KeyPoint> kpObject4, kpImage4;
        Mat desObject4, desImage4;


        //SURF Detector, and descriptor parameters, match object initialization
        minHess = 2000;
        SurfFeatureDetector detector4(minHess);
        detector4.detect(object4, kpObject4);
        SurfDescriptorExtractor extractor4;
        extractor4.compute(object4, kpObject4, desObject4);
        FlannBasedMatcher matcher4;


        //Object corner points for plotting box
        vector<Point2f> obj_corners4(4);
        obj_corners4[0] = cvPoint(0, 0);
        obj_corners4[1] = cvPoint(object4.cols, 0);
        obj_corners4[2] = cvPoint(object4.cols, object4.rows);
        obj_corners4[3] = cvPoint(0, object4.rows);




        int x = 0;

        while (1) {
            x = 0;
            frameCount++;
            Mat frame;
            Mat image;
            cap>>frame;
            cvtColor(frame, image, CV_RGB2GRAY);




            Mat des_image, img_matches, H;
            vector<KeyPoint> kp_image;
            vector<vector<DMatch > > matches;
            vector<DMatch > good_matches;
            vector<Point2f> obj;
            vector<Point2f> scene;
            vector<Point2f> scene_corners(4);

            detector.detect(image, kp_image);
            extractor.compute(image, kp_image, des_image);
            matcher.knnMatch(desObject, des_image, matches, 2);



            Mat des_image2, img_matches2, H2;
            vector<KeyPoint> kp_image2;
            vector<vector<DMatch > > matches2;
            vector<DMatch > good_matches2;
            vector<Point2f> obj2;
            vector<Point2f> scene2;
            vector<Point2f> scene_corners2(4);

            detector2.detect(image, kp_image2);
            extractor2.compute(image, kp_image2, des_image2);
            matcher2.knnMatch(desObject2, des_image2, matches2, 2);


            Mat des_image3, img_matches3, H3;
            vector<KeyPoint> kp_image3;
            vector<vector<DMatch > > matches3;
            vector<DMatch > good_matches3;
            vector<Point2f> obj3;
            vector<Point2f> scene3;
            vector<Point2f> scene_corners3(4);

            detector3.detect(image, kp_image3);
            extractor3.compute(image, kp_image3, des_image3);
            matcher3.knnMatch(desObject3, des_image3, matches3, 2);


            Mat des_image4, img_matches4, H4;
            vector<KeyPoint> kp_image4;
            vector<vector<DMatch > > matches4;
            vector<DMatch > good_matches4;
            vector<Point2f> obj4;
            vector<Point2f> scene4;
            vector<Point2f> scene_corners4(4);

            detector4.detect(image, kp_image4);
            extractor4.compute(image, kp_image4, des_image4);
            matcher4.knnMatch(desObject4, des_image4, matches4, 2);


            for (int i = 0; i < min(des_image4.rows - 1, (int) matches4.size()); i++) //THIS LOOP IS SENSITIVE TO SEGFAULTS
            {
                if ((matches4[i][0].distance < thresholdMatchingNN * (matches4[i][1].distance)) && ((int) matches4[i].size() <= 2 && (int) matches4[i].size() > 0)) {
                    good_matches4.push_back(matches4[i][0]);
                }
            }


            for (int i = 0; i < min(des_image3.rows - 1, (int) matches3.size()); i++) //THIS LOOP IS SENSITIVE TO SEGFAULTS
            {
                if ((matches3[i][0].distance < thresholdMatchingNN * (matches3[i][1].distance)) && ((int) matches3[i].size() <= 2 && (int) matches3[i].size() > 0)) {
                    good_matches3.push_back(matches3[i][0]);
                }
            }


            for (int i = 0; i < min(des_image2.rows - 1, (int) matches2.size()); i++) //THIS LOOP IS SENSITIVE TO SEGFAULTS
            {
                if ((matches2[i][0].distance < thresholdMatchingNN * (matches2[i][1].distance)) && ((int) matches2[i].size() <= 2 && (int) matches2[i].size() > 0)) {
                    good_matches2.push_back(matches2[i][0]);
                }
            }



            for (int i = 0; i < min(des_image.rows - 1, (int) matches.size()); i++) //THIS LOOP IS SENSITIVE TO SEGFAULTS
            {
                if ((matches[i][0].distance < thresholdMatchingNN * (matches[i][1].distance)) && ((int) matches[i].size() <= 2 && (int) matches[i].size() > 0)) {
                    good_matches.push_back(matches[i][0]);
                }
            }


            
            
            
            if (good_matches.size() >= thresholdGoodMatches) {
                drawMatches(object, kpObject, image, kp_image, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


                //Display that the object is found
                putText(img_matches, "Max Speed 60 kmps", cvPoint(250, 50), FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(0, 0, 250), 1, CV_AA);
                for (unsigned int i = 0; i < good_matches.size(); i++) {
                    //Get the keypoints from the good matches
                    obj.push_back(kpObject[ good_matches[i].queryIdx ].pt);
                    scene.push_back(kp_image[ good_matches[i].trainIdx ].pt);
                }

                H = findHomography(obj, scene, CV_RANSAC);

                perspectiveTransform(obj_corners, scene_corners, H);

                //Draw lines between the corners (the mapped object in the scene image )
                line(img_matches, scene_corners[0] + Point2f(object.cols, 0), scene_corners[1] + Point2f(object.cols, 0), Scalar(0, 255, 0), 4);
                line(img_matches, scene_corners[1] + Point2f(object.cols, 0), scene_corners[2] + Point2f(object.cols, 0), Scalar(0, 255, 0), 4);
                line(img_matches, scene_corners[2] + Point2f(object.cols, 0), scene_corners[3] + Point2f(object.cols, 0), Scalar(0, 255, 0), 4);
                line(img_matches, scene_corners[3] + Point2f(object.cols, 0), scene_corners[0] + Point2f(object.cols, 0), Scalar(0, 255, 0), 4);
                x = 1;
            }

            else if (good_matches2.size() >= thresholdGoodMatches) {
                drawMatches(object2, kpObject2, image, kp_image2, good_matches2, img_matches2, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


                //Display that the object is found
                putText(img_matches2, "Do not take U turn", cvPoint(250, 50), FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(0, 0, 250), 1, CV_AA);
                for (unsigned int i = 0; i < good_matches2.size(); i++) {
                    //Get the keypoints from the good matches
                    obj2.push_back(kpObject2[ good_matches2[i].queryIdx ].pt);
                    scene2.push_back(kp_image2[ good_matches2[i].trainIdx ].pt);
                }

                H2 = findHomography(obj2, scene2, CV_RANSAC);

                perspectiveTransform(obj_corners2, scene_corners2, H2);

                //Draw lines between the corners (the mapped object in the scene image )
                line(img_matches2, scene_corners2[0] + Point2f(object2.cols, 0), scene_corners2[1] + Point2f(object2.cols, 0), Scalar(0, 255, 0), 4);
                line(img_matches2, scene_corners2[1] + Point2f(object2.cols, 0), scene_corners2[2] + Point2f(object2.cols, 0), Scalar(0, 255, 0), 4);
                line(img_matches2, scene_corners2[2] + Point2f(object2.cols, 0), scene_corners2[3] + Point2f(object2.cols, 0), Scalar(0, 255, 0), 4);
                line(img_matches2, scene_corners2[3] + Point2f(object2.cols, 0), scene_corners2[0] + Point2f(object2.cols, 0), Scalar(0, 255, 0), 4);
                x = 2;
            }


            else if (good_matches3.size() >= thresholdGoodMatches) {
                drawMatches(object3, kpObject3, image, kp_image3, good_matches3, img_matches3, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


                //Display that the object is found
                putText(img_matches3, "No Parking", cvPoint(250, 50), FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(0, 0, 250), 1, CV_AA);
                for (unsigned int i = 0; i < good_matches3.size(); i++) {
                    //Get the keypoints from the good matches
                    obj3.push_back(kpObject3[ good_matches3[i].queryIdx ].pt);
                    scene3.push_back(kp_image3[ good_matches3[i].trainIdx ].pt);
                }

                H3 = findHomography(obj3, scene3, CV_RANSAC);

                perspectiveTransform(obj_corners3, scene_corners3, H3);

                //Draw lines between the corners (the mapped object in the scene image )
                line(img_matches3, scene_corners3[0] + Point2f(object3.cols, 0), scene_corners3[1] + Point2f(object3.cols, 0), Scalar(0, 255, 0), 4);
                line(img_matches3, scene_corners3[1] + Point2f(object3.cols, 0), scene_corners3[2] + Point2f(object3.cols, 0), Scalar(0, 255, 0), 4);
                line(img_matches3, scene_corners3[2] + Point2f(object3.cols, 0), scene_corners3[3] + Point2f(object3.cols, 0), Scalar(0, 255, 0), 4);
                line(img_matches3, scene_corners3[3] + Point2f(object3.cols, 0), scene_corners3[0] + Point2f(object3.cols, 0), Scalar(0, 255, 0), 4);
                x = 3;
            }

            else if (good_matches4.size() >= thresholdGoodMatches) {

                drawMatches(object4, kpObject4, image, kp_image4, good_matches4, img_matches4, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

                //Display that the object is found
                putText(img_matches4, "STOP", cvPoint(250, 50), FONT_HERSHEY_COMPLEX_SMALL, 2, cvScalar(0, 0, 250), 1, CV_AA);
                for (unsigned int i = 0; i < good_matches4.size(); i++) {
                    //Get the keypoints from the good matches
                    obj4.push_back(kpObject4[ good_matches4[i].queryIdx ].pt);
                    scene4.push_back(kp_image4[ good_matches4[i].trainIdx ].pt);
                }

                H4 = findHomography(obj4, scene4, CV_RANSAC);

                perspectiveTransform(obj_corners4, scene_corners4, H4);

                //Draw lines between the corners (the mapped object in the scene image )
                line(img_matches4, scene_corners4[0] + Point2f(object4.cols, 0), scene_corners4[1] + Point2f(object4.cols, 0), Scalar(0, 255, 0), 4);
                line(img_matches4, scene_corners4[1] + Point2f(object4.cols, 0), scene_corners4[2] + Point2f(object4.cols, 0), Scalar(0, 255, 0), 4);
                line(img_matches4, scene_corners4[2] + Point2f(object4.cols, 0), scene_corners4[3] + Point2f(object4.cols, 0), Scalar(0, 255, 0), 4);
                line(img_matches4, scene_corners4[3] + Point2f(object4.cols, 0), scene_corners4[0] + Point2f(object4.cols, 0), Scalar(0, 255, 0), 4);
                x = 4;
            } else {
                putText(img_matches4, "ss", cvPoint(10, 50), FONT_HERSHEY_COMPLEX_SMALL, 3, cvScalar(0, 0, 250), 1, CV_AA);
            }

            if(x==1){
                imshow("Good Matches", img_matches);
                
            }
            else if(x==2){
                imshow("Good Matches", img_matches2);
                
            }
            else if(x==3){
                imshow("Good Matches", img_matches3);
            }
            else if(x==4){
                imshow("Good Matches", img_matches4);
            }
            else{
                
            }










            escapeKey = cvWaitKey(10);
            //imwrite("C:/School/Image Processing/bookIP3.jpg", img_matches);

            if (frameCount > 10)
                escapeKey = 'q';

            if (x == 4) {
                continue;
            }














        }

        //average frames per second
        if (true) {
            t = ((double) getTickCount() - t) / getTickFrequency();
            cout << t << " " << frameCount / t << endl;
            cvWaitKey(0);
        }

        frameCount = 0;
        escapeKey = 'a';
    }

    //Release camera and exit
    cap.release();
    return 0;
}


