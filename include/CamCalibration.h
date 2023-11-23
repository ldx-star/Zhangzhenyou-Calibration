//
// Created by ldx on 23-11-16.
//

#ifndef ZHANGZHENYOU_CALIBRATION_CAMCALIBRATION_H
#define ZHANGZHENYOU_CALIBRATION_CAMCALIBRATION_H

#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>

class CamCalibration {
public:

public:
    CamCalibration(const std::string &chessBoard_path, const int row, const int col, const double square_size,
                   const int img_num);

    void Calibrate();

    cv::Point2f reProjectPoint(const cv::Point3f &point_3d, const cv::Mat &R,const cv::Mat &t,const cv::Mat &K, const double k1,const double k2);


private:
    bool readImages();

    bool getKeyPoints();

    void CalcH();

    void CalcK();

    void CalcRT();

    void CalDistCoeff();

    double CalcRepjErr();

    void Normalize(const std::vector<cv::Point2f> &points, std::vector<cv::Point2f> &normal_points, cv::Mat &normT);


private:
    std::string chessBoard_path_;
    int row_;
    int col_;
    double square_size_;
    int img_num_;
    cv::Mat K_;
    cv::Mat dist_coeff_;


    std::vector<cv::Mat> chessBoards_;
    std::vector<std::vector<cv::Point2f>> points_3d_vec_; // z = 0;
    std::vector<std::vector<cv::Point2f>> points_2d_vec_;
    std::vector<cv::Mat> H_vec_;
    std::vector<cv::Mat> R_vec_;
    std::vector<cv::Mat> t_vec_;
};


#endif //ZHANGZHENYOU_CALIBRATION_CAMCALIBRATION_H
