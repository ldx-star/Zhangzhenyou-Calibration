//
// Created by ldx on 23-11-16.
//
#include "../include/CamCalibration.h"

/**
 * 构造函数
 * @param img_path 棋盘格路径
 * @param col 列数
 * @param row 行数
 * @param square_size 每个棋盘格大小
 * @param img_num 图片个数
 */
CamCalibration::CamCalibration(const std::string &chessBoard_path, const int col, const int row,
                               const float square_size, const int img_num)
        : chessBoard_path_(chessBoard_path), col_(col), row_(row), square_size_(square_size), img_num_(img_num) {}

/**
 *  相机标定
 */
void CamCalibration::Calibrate() {
    std::cout << "start Calibrate" << std::endl;

    if(readImages() && getKeyPoints()){

    }

    std::cout << "Calibrate succeed" << std::endl;
}

/**
 * 读取棋盘格图片
 * @return 读取成功返回true，失败返回false
 */
bool CamCalibration::readImages() {
    const int img_num = img_num_;
    const std::string &chessBoard_path = chessBoard_path_;
    std::vector<cv::Mat> &chessBoards = chessBoards_;
    for (int i = 0; i < img_num; i++) {
        std::string img_path = chessBoard_path + "/" + std::to_string(i + 100000) + ".png";
        cv::Mat img = cv::imread(img_path, 0);
        chessBoards.push_back(img);
        if (img.empty()) {
            std::cerr << "read chessBoards failed:" << chessBoard_path;
            return false;
        }
    }
    std::cout << "chessBoards image read succeed" << std::endl;
    return true;
}

bool CamCalibration::getKeyPoints() {
    auto chessBoards = chessBoards_;
    const float square_size = square_size_;
    auto &points_3d_vec = points_3d_vec_;
    auto &points_2d_vec = points_2d_vec_;

    const int row = row_;
    const int col = col_;
    //采集世界坐标
    for (int i = 0; i < chessBoards.size(); i++) {
        std::vector<cv::Point2f> points_3d; // 一张图的世界坐标
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                cv::Point2f point;
                point.x = r * square_size;
                point.y = c * square_size;
                points_3d.push_back(point);
            }
        }
        points_3d_vec.push_back(points_3d);
    }
    //采集像素坐标,使用opencv库提取角点
    for (auto img: chessBoards) {
        std::vector<cv::Point2f> points_2d;

        bool found_flag = cv::findChessboardCorners(img, cv::Size(col, row), points_2d, cv::CALIB_CB_ADAPTIVE_THRESH +
                                                                                        cv::CALIB_CB_NORMALIZE_IMAGE); //cv::Size(col,row)
        if (!found_flag) {
            std::cerr << "found chess board corner failed";
            return false;
        }
        //指定亚像素计算迭代标注
        cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 40,0.001);
        cv::cornerSubPix(img, points_2d, cv::Size(5, 5), cv::Size(-1, -1), criteria);

        //display
//        cv::drawChessboardCorners(img,cv::Size(col,row),points_2d,found_flag);
//        cv::imshow("corner img", img);
//        cv::waitKey(300);

        points_2d_vec.push_back(points_2d);
    }
    std::cout << "getKeyPoints succeed" << std::endl;
    return true;
}