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

    if (readImages() && getKeyPoints()) {
        CalcH();
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

/**
 * 获取世界坐标和像素坐标
 * @return 成功返回true,失败返回false
 */
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
        cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 40, 0.001);
        cv::cornerSubPix(img, points_2d, cv::Size(5, 5), cv::Size(-1, -1), criteria);

        //display
//        cv::cvtColor(img,img,cv::COLOR_GRAY2BGR);
//        cv::drawChessboardCorners(img, cv::Size(col, row), points_2d, found_flag);
//        cv::namedWindow("corner img", cv::WINDOW_NORMAL);
//        cv::resizeWindow("corner img", img.cols / 2, img.rows / 2);
//        cv::imshow("corner img", img);
//        cv::waitKey(300);

        points_2d_vec.push_back(points_2d);
    }
    std::cout << "getKeyPoints succeed" << std::endl;
    return true;
}

/**
 * 计算单应性矩阵
 */
void CamCalibration::CalcH() {
    const auto &points_3d_vec = points_3d_vec_;
    const auto &points_2d_vec = points_2d_vec_;
    for (int i = 0; i < points_2d_vec.size(); i++) {
        //每一张图的世界坐标和像素坐标
        const auto &points_3d = points_3d_vec[i];
        const auto &points_2d = points_2d_vec[i];
        std::vector<cv::Point2f> normal_points_3d, normal_points_2d;
        cv::Mat normT_3d, normT_2d;
        Normalize(points_3d, normal_points_3d, normT_3d);
        Normalize(points_2d, normal_points_2d, normT_2d);

        cv::Mat H = cv::Mat::eye(3, 3, CV_32F);

        int corner_size = normal_points_2d.size();
        if (corner_size < 4) {
            std::cerr << "corner size < 4";
            exit(-1);
        }

        cv::Mat A(corner_size * 2, 9, CV_32F, cv::Scalar(0));
        for (int i = 0; i < corner_size; i++) {
            cv::Point2f point_3d = points_3d[i];
            cv::Point2f point_2d = points_2d[i];
            A.at<float>(i * 2, 0) = point_3d.x;
            A.at<float>(i * 2, 1) = point_3d.y;
            A.at<float>(i * 2, 2) = 1;
            A.at<float>(i * 2, 3) = 0;
            A.at<float>(i * 2, 4) = 0;
            A.at<float>(i * 2, 5) = 0;
            A.at<float>(i * 2, 6) = -point_2d.x * point_3d.x;
            A.at<float>(i * 2, 7) = -point_2d.x * point_3d.y;
            A.at<float>(i * 2, 8) = -point_2d.x;

            A.at<float>(i * 2 + 1, 0) = 0;
            A.at<float>(i * 2 + 1, 1) = 0;
            A.at<float>(i * 2 + 1, 2) = 0;
            A.at<float>(i * 2 + 1, 3) = point_3d.x;
            A.at<float>(i * 2 + 1, 4) = point_3d.y;
            A.at<float>(i * 2 + 1, 5) = 1;
            A.at<float>(i * 2 + 1, 6) = -point_2d.y * point_3d.x;
            A.at<float>(i * 2 + 1, 7) = -point_2d.y * point_3d.y;
            A.at<float>(i * 2 + 1, 8) = -point_2d.y;
        }
        cv::Mat U, W, VT;                                                    // A =UWV^T
        cv::SVD::compute(A, W, U, VT,
                         cv::SVD::MODIFY_A | cv::SVD::FULL_UV); // Eigen 返回的是V,列向量就是特征向量, opencv 返回的是VT，所以行向量是特征向量
        H = VT.row(8).reshape(0, 3);
        // H = inv_N2 * H * N3
        cv::Mat normT_2d_inv;
        cv::invert(normT_2d, normT_2d_inv);
        H = normT_2d_inv * H * normT_3d;
        H_vec_.push_back(H);
    }
}

/**
 * Z-Score 标准化（均值为0，方差为1）
 * @param points 原始数据点
 * @param normal_points 输出型参数，标准化后的数据点
 * @param normT 输出型参数，归一化矩阵的转置
 */
void CamCalibration::Normalize(const std::vector<cv::Point2f> &points, std::vector<cv::Point2f> &normal_points,
                               cv::Mat &normT) {
    //求均值
    float mean_x = 0;
    float mean_y = 0;
    for (const auto &point: points) {
        mean_x += point.x;
        mean_y += point.y;
    }
    mean_x /= points.size();
    mean_y /= points.size();
    //求方差
    for (const auto &point: points) {
        mean_x += point.x;
        mean_y += point.y;
    }
    float variance_x = 0;
    float variance_y = 0;
    for (const auto &point: points) {
        float tmp_x = pow(point.x - mean_x, 2);
        float tmp_y = pow(point.y - mean_y, 2);
        variance_x += tmp_x;
        variance_y += tmp_y;
    }
    variance_x = sqrt(variance_x);
    variance_y = sqrt(variance_y);

    for (const auto &point: points) {
        cv::Point2f p;
        p.x = (point.x - mean_x) / variance_x;
        p.y = (point.y - mean_y) / variance_y;
        normal_points.push_back(p);
    }
    normT = cv::Mat::eye(3, 3, CV_32F);
    normT.at<float>(0, 0) = 1 / variance_x;
    normT.at<float>(0, 2) = -mean_x / variance_x;
    normT.at<float>(1, 1) = 1 / variance_y;
    normT.at<float>(1, 2) = -mean_y / variance_y;
}

