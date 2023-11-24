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
                               const double square_size, const int img_num)
        : chessBoard_path_(chessBoard_path), col_(col), row_(row), square_size_(square_size), img_num_(img_num) {
    K_ = cv::Mat::eye(3, 3, CV_64F);
}

/**
 *  相机标定
 */
void CamCalibration::Calibrate() {
    std::cout << "start Calibrate" << std::endl;

    if (readImages() && getKeyPoints()) {
        CalcH();
        CalcK();
        CalcRT();
        CalDistCoeff();
        double repjErr = CalcRepjErr();
        std::cout << "优化前重投影误差： " << repjErr << std::endl;
        Optimize();
        repjErr = CalcRepjErr();
        std::cout << "优化后重投影误差： " << repjErr << std::endl;

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
    const double square_size = square_size_;
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
        cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 0.001);
        cv::cornerSubPix(img, points_2d, cv::Size(11, 11), cv::Size(-1, -1), criteria);

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

        cv::Mat H = cv::Mat::eye(3, 3, CV_64F);

        int corner_size = normal_points_2d.size();
        if (corner_size < 4) {
            std::cerr << "corner size < 4";
            exit(-1);
        }

        cv::Mat A(corner_size * 2, 9, CV_64F, cv::Scalar(0));
        for (int i = 0; i < corner_size; i++) {
            cv::Point2f point_3d = normal_points_3d[i];
            cv::Point2f point_2d = normal_points_2d[i];
            A.at<double>(i * 2, 0) = point_3d.x;
            A.at<double>(i * 2, 1) = point_3d.y;
            A.at<double>(i * 2, 2) = 1;
            A.at<double>(i * 2, 3) = 0;
            A.at<double>(i * 2, 4) = 0;
            A.at<double>(i * 2, 5) = 0;
            A.at<double>(i * 2, 6) = -point_2d.x * point_3d.x;
            A.at<double>(i * 2, 7) = -point_2d.x * point_3d.y;
            A.at<double>(i * 2, 8) = -point_2d.x;

            A.at<double>(i * 2 + 1, 0) = 0;
            A.at<double>(i * 2 + 1, 1) = 0;
            A.at<double>(i * 2 + 1, 2) = 0;
            A.at<double>(i * 2 + 1, 3) = point_3d.x;
            A.at<double>(i * 2 + 1, 4) = point_3d.y;
            A.at<double>(i * 2 + 1, 5) = 1;
            A.at<double>(i * 2 + 1, 6) = -point_2d.y * point_3d.x;
            A.at<double>(i * 2 + 1, 7) = -point_2d.y * point_3d.y;
            A.at<double>(i * 2 + 1, 8) = -point_2d.y;
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
//    //求均值
//    double mean_x = 0;
//    double mean_y = 0;
//    for (const auto &point: points) {
//        mean_x += point.x;
//        mean_y += point.y;
//    }
//    mean_x /= points.size();
//    mean_y /= points.size();
//    //求方差
//    for (const auto &point: points) {
//        mean_x += point.x;
//        mean_y += point.y;
//    }
//    double variance_x = 0;
//    double variance_y = 0;
//    for (const auto &point: points) {
//        double tmp_x = pow(point.x - mean_x, 2);
//        double tmp_y = pow(point.y - mean_y, 2);
//        variance_x += tmp_x;
//        variance_y += tmp_y;
//    }
//    variance_x = sqrt(variance_x);
//    variance_y = sqrt(variance_y);
//
//    for (const auto &point: points) {
//        cv::Point2f p;
//        p.x = (point.x - mean_x) / variance_x;
//        p.y = (point.y - mean_y) / variance_y;
//        normal_points.push_back(p);
//    }
//    normT = cv::Mat::eye(3, 3, CV_64F);
//    normT.at<double>(0, 0) = 1 / variance_x;
//    normT.at<double>(0, 2) = -mean_x / variance_x;
//    normT.at<double>(1, 1) = 1 / variance_y;
//    normT.at<double>(1, 2) = -mean_y / variance_y;


    normT = cv::Mat::eye(3, 3, CV_64F);
    double mean_x = 0;
    double mean_y = 0;
    for (const auto &p: points) {
        mean_x += p.x;
        mean_y += p.y;
    }
    mean_x /= points.size();
    mean_y /= points.size();
    double mean_dev_x = 0;
    double mean_dev_y = 0;
    for (const auto &p: points) {
        mean_dev_x += fabs(p.x - mean_x);
        mean_dev_y += fabs(p.y - mean_y);
    }
    mean_dev_x /= points.size();
    mean_dev_y /= points.size();
    double sx = 1.0 / mean_dev_x;
    double sy = 1.0 / mean_dev_y;
    normal_points.clear();
    for (const auto &p: points) {
        cv::Point2f p_tmp;
        p_tmp.x = sx * p.x - mean_x * sx;
        p_tmp.y = sy * p.y - mean_y * sy;
        normal_points.push_back(p_tmp);
    }
    normT.at<double>(0, 0) = sx;
    normT.at<double>(0, 2) = -mean_x * sx;
    normT.at<double>(1, 1) = sy;
    normT.at<double>(1, 2) = -mean_y * sy;
}

/**
 * 求内参矩阵
 */
void CamCalibration::CalcK() {
    const auto &H_vec = H_vec_;
    cv::Mat A(H_vec.size() * 2, 6, CV_64F, cv::Scalar(0));

    for (int i = 0; i < H_vec.size(); i++) {
        cv::Mat H = H_vec[i];
        double h11 = H.at<double>(0, 0);
        double h12 = H.at<double>(1, 0);
        double h13 = H.at<double>(2, 0);
        double h21 = H.at<double>(0, 1);
        double h22 = H.at<double>(1, 1);
        double h23 = H.at<double>(2, 1);

        cv::Mat v11 = (cv::Mat_<double>(1, 6) << h11 * h11, h11 * h12 + h12 * h11, h12 * h12, h13 * h11 + h11 * h13,
                h13 * h12 + h12 * h13, h13 * h13);
        cv::Mat v12 = (cv::Mat_<double>(1, 6) << h11 * h21, h11 * h22 + h12 * h21, h12 * h22, h13 * h21 + h11 * h23,
                h13 * h22 + h12 * h23, h13 * h23);
        cv::Mat v22 = (cv::Mat_<double>(1, 6) << h21 * h21, h21 * h22 + h22 * h21, h22 * h22, h23 * h21 + h21 * h23,
                h23 * h22 + h22 * h23, h23 * h23);
        v12.copyTo(A.row(2 * i));
        cv::Mat v_tmp = (v11 - v22);
        v_tmp.copyTo(A.row(2 * i + 1));
    }
    cv::Mat U, W, VT;
    cv::SVD::compute(A, W, U, VT);
//    std::cout << "A:\n" << A << std::endl;

    cv::Mat B = VT.row(5);
//    std::cout << "B:\n" << B << std::endl;
    double B11 = B.at<double>(0, 0);
    double B12 = B.at<double>(0, 1);
    double B22 = B.at<double>(0, 2);
    double B13 = B.at<double>(0, 3);
    double B23 = B.at<double>(0, 4);
    double B33 = B.at<double>(0, 5);

    double v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 * B12);
    double lambda = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11;
    double alpha = sqrt(lambda / B11);
    double beta = sqrt(lambda * B11 / (B11 * B22 - B12 * B12));
    double gamma = -B12 * alpha * alpha * beta / lambda;
    double u0 = gamma * v0 / beta - B13 * alpha * alpha / lambda;

    gamma = 0;
    K_.at<double>(0, 0) = alpha;
    K_.at<double>(0, 1) = gamma;
    K_.at<double>(0, 2) = u0;
    K_.at<double>(1, 1) = beta;
    K_.at<double>(1, 2) = v0;
    std::cout << "K:\n" << K_ << std::endl;
}

void CamCalibration::CalcRT() {
    const auto &K = K_;
    const auto &H_vec = H_vec_;
    auto &R_vec = R_vec_;
    auto &t_vec = t_vec_;

    cv::Mat K_inverse;
    cv::invert(K, K_inverse);

    for (const auto &H: H_vec) {
        cv::Mat M = K_inverse * H;

        cv::Vec3d r1(M.at<double>(0, 0), M.at<double>(1, 0), M.at<double>(2, 0));
        cv::Vec3d r2(M.at<double>(0, 1), M.at<double>(1, 1), M.at<double>(2, 1));
        cv::Vec3d r3 = r1.cross(r2);
        cv::Mat Q = cv::Mat::eye(3, 3, CV_64F);

        Q.at<double>(0, 0) = r1(0);
        Q.at<double>(1, 0) = r1(1);
        Q.at<double>(2, 0) = r1(2);
        Q.at<double>(0, 1) = r2(0);
        Q.at<double>(1, 1) = r2(1);
        Q.at<double>(2, 1) = r2(2);
        Q.at<double>(0, 2) = r3(0);
        Q.at<double>(1, 2) = r3(1);
        Q.at<double>(2, 2) = r3(2);
        cv::Mat normQ;
        cv::normalize(Q, normQ);

        cv::Mat U, W, VT;
        cv::SVD::compute(normQ, W, U, VT, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
        cv::Mat R = U * VT;

        R_vec.push_back(R);
        cv::Mat t = cv::Mat::eye(3, 1, CV_64F);
        M.col(2).copyTo(t.col(0));
        t_vec.push_back(t);
    }
}

void CamCalibration::CalDistCoeff() {
    std::vector<double> r2_vec;
    std::vector<cv::Point2f> ideal_point_vec;

    //用一副图进行计算
    const cv::Mat &K = K_;
    const cv::Mat &R = R_vec_[0];
    const cv::Mat &t = t_vec_[0];
    auto points_3d = points_3d_vec_[0];
    auto &dist_coeff = dist_coeff_;
    for (const auto &point_3d: points_3d) {
        cv::Mat p_3d = (cv::Mat_<double>(3, 1) << point_3d.x, point_3d.y, 0);
        //世界坐标转相机坐标
        cv::Mat p_cam = R * p_3d + t;
        //转换成欧式坐标
        p_cam.at<double>(0, 0) = p_cam.at<double>(0, 0) / p_cam.at<double>(2, 0);
        p_cam.at<double>(1, 0) = p_cam.at<double>(1, 0) / p_cam.at<double>(2, 0);
        p_cam.at<double>(2, 0) = 1;
        double x = p_cam.at<double>(0, 0);
        double y = p_cam.at<double>(1, 0);
        double r2 = x * x + y * y;
        r2_vec.push_back(r2);

        //相机坐标转像素坐标
        cv::Mat p_pix = K * p_cam;
        ideal_point_vec.emplace_back(p_pix.at<double>(0, 0), p_pix.at<double>(1, 0));
    }
    const std::vector<cv::Point2f> &dist_point_vec = points_2d_vec_[0];
    double u0 = K.at<double>(0, 2);
    double v0 = K.at<double>(1, 2);

    cv::Mat D = cv::Mat::eye(ideal_point_vec.size() * 2, 2, CV_64F);
    cv::Mat d = cv::Mat::eye(ideal_point_vec.size() * 2, 1, CV_64F);
    for (int i = 0; i < ideal_point_vec.size(); i++) {
        double r2 = r2_vec[i];
        const auto &ideal_p = ideal_point_vec[i];
        const auto &dist_p = dist_point_vec[i];
        D.at<double>(i * 2, 0) = (ideal_p.x - u0) * r2;
        D.at<double>(i * 2, 1) = (ideal_p.x - u0) * r2 * r2;
        D.at<double>(i * 2 + 1, 0) = (ideal_p.y - v0) * r2;
        D.at<double>(i * 2 + 1, 1) = (ideal_p.y - v0) * r2 * r2;
        d.at<double>(2 * i, 0) = dist_p.x - ideal_p.x;
        d.at<double>(2 * i + 1, 0) = dist_p.y - ideal_p.y;

    }
    cv::Mat DT;
    cv::transpose(D, DT);
//    std::cout << "D:" << D << std::endl;

    cv::Mat DTD_inverse;
    cv::invert(DT * D, DTD_inverse);
//    std::cout << "DTD_inv:" << DTD_inverse << std::endl;

    dist_coeff = DTD_inverse * DT * d;
    std::cout << "distort coeff: " << dist_coeff.at<double>(0, 0) << ", " << dist_coeff.at<double>(1, 0) << std::endl;
}

/**
 * 计算重投影误差
 * @return
 */
double CamCalibration::CalcRepjErr() {
    const auto &points_3d_vec = points_3d_vec_;
    const auto &points_2d_vec = points_2d_vec_;
    double repjErr = 0;
    int p_num = 0;

    for (int i = 0; i < points_3d_vec.size(); i++) {
        for (int j = 0; j < points_3d_vec[i].size(); j++) {
            cv::Point3f p;
            p.x = points_3d_vec[i][j].x;
            p.y = points_3d_vec[i][j].y;
            p.z = 0;

            cv::Point2f repj_p = reProjectPoint(p, R_vec_[i], t_vec_[i], K_, dist_coeff_.at<double>(0, 0),
                                                dist_coeff_.at<double>(1, 0));
            const cv::Point2f origin_p = points_2d_vec[i][j];
            repjErr += sqrt((origin_p.x - repj_p.x) * (origin_p.x - repj_p.x) +
                            (origin_p.y - repj_p.y) * (origin_p.y - repj_p.y));
            p_num++;
        }
    }
    repjErr /= p_num;
    return repjErr;
}

/**
 *  获得重投影点（将世界坐标用求出的参数转成二维坐标）
 * @param point_3d 世界坐标
 * @param R 旋转矩阵
 * @param t 平移矩阵
 * @param K 相机内参
 * @param k1 畸变系数
 * @param k2 畸变系数
 * @return 像素坐标
 */
cv::Point2f CamCalibration::reProjectPoint(const cv::Point3f &point_3d, const cv::Mat &R, const cv::Mat &t,
                                           const cv::Mat &K, const double k1, const double k2) {
    //世界坐标系
    cv::Mat p = (cv::Mat_<double>(3, 1) << point_3d.x, point_3d.y, point_3d.z);
    //相机坐标系
    cv::Mat p_cam = R * p + t;
    //转成欧式坐标
    double x = p_cam.at<double>(0, 0) / p_cam.at<double>(2, 0);
    double y = p_cam.at<double>(1, 0) / p_cam.at<double>(2, 0);
    double r2 = x * x + y * y;

    double x_dist = x * (1 + k1 * r2 + k2 * r2 * r2);
    double y_dist = y * (1 + k1 * r2 + k2 * r2 * r2);
    double u_dist = x_dist * K.at<double>(0, 0) + K.at<double>(0, 2);
    double v_dist = y_dist * K.at<double>(1, 1) + K.at<double>(1, 2);

    cv::Point2f p_dist;
    p_dist.x = u_dist;
    p_dist.y = v_dist;
    return p_dist;
}

struct CamCalibration::ReprojErr {
public:
    ReprojErr(const cv::Point2f &pixPoint_2d, const cv::Point2f &worldPoint_3d)
            : pixPoint_2d_(pixPoint_2d), worldPoint_3d_(worldPoint_3d) {}

    template<class T>
    // const 优先修饰左边，左边没有修饰右边
    // const int 、 int const 修饰int 内容不能被修改
    // int *const 修饰* 指针不能被修改
    bool operator()(const T *const Rt, const T *const K, const T *const dist_coeff, T *residual) const {
        T p_3d[3] = {static_cast<T>(worldPoint_3d_.x, worldPoint_3d_.y, 0)};
        T p[3];
        //Rt[0,1,2]是旋转向量
        ceres::AngleAxisRotatePoint(Rt, p_3d, p); // p_3d 通过旋转向量 得到 p
        //Rt[3,4,5]是平移向量
        p[0] += Rt[3];
        p[1] = Rt[4];
        p[2] = Rt[5];
        //转成欧式坐标
        T x = p[0] / p[2];
        T y = p[1] / p[2];
        T r2 = x * x + y * y;

        const T &alpha = K[0];
        const T &beta = K[1];
        const T &u0 = K[2];
        const T &v0 = K[3];

        T x_dist = x * (static_cast<T> (1) + dist_coeff[0] * r2 + dist_coeff[1] * r2*r2);
        T y_dist = y * (static_cast<T> (1) + dist_coeff[0] * r2 + dist_coeff[1] * r2*r2);

        const T u_dist = alpha * x_dist +u0;
        const T v_dist = beta * y_dist +v0;

        residual[0] = u_dist - static_cast<T>(pixPoint_2d_.x);
        residual[1] = v_dist - static_cast<T>(pixPoint_2d_.y);
        return true;
    }

private:
    const cv::Point2f worldPoint_3d_;
    const cv::Point2f pixPoint_2d_;
};

void CamCalibration::Optimize() {
    const auto &points_3d_vec = points_3d_vec_;
    const auto &points_2d_vec = points_2d_vec_;
    const auto &K = K_;
    const auto &dist_coeff = dist_coeff_;
    const auto &R_vec = R_vec_;
    const auto &t_vec = t_vec_;
    ceres::Problem problem;
    int pic_num = points_3d_vec.size();
    double *K_param = new double[4];
    *(K_param + 0) = K.at<double>(0, 0);
    *(K_param + 1) = K.at<double>(1, 1);
    *(K_param + 2) = K.at<double>(0, 2);
    *(K_param + 3) = K.at<double>(1, 2);
    double *dist_coeff_param = new double[2];
    *(dist_coeff_param) = dist_coeff.at<double>(0, 0);
    *(dist_coeff_param + 1) = dist_coeff.at<double>(1, 0);

    double *Rt_param = new double[6 * pic_num];
    for (int i = 0; i < pic_num; i++) {
        const cv::Mat &R = R_vec[i];
        const cv::Mat &t = t_vec[i];
        cv::Mat angle_axis;
        cv::Rodrigues(R, angle_axis); //旋转矩阵转为旋转向量
        *(Rt_param + 6 * i + 0) = angle_axis.at<double>(0, 0);
        *(Rt_param + 6 * i + 1) = angle_axis.at<double>(1, 0);
        *(Rt_param + 6 * i + 2) = angle_axis.at<double>(2, 0);
        *(Rt_param + 6 * i + 3) = t.at<double>(0, 0);
        *(Rt_param + 6 * i + 4) = t.at<double>(1, 0);
        *(Rt_param + 6 * i + 5) = t.at<double>(2, 0);
    }
    for (int i = 0; i < pic_num; i++) {
        const auto &points_3d = points_3d_vec[i];
        const auto &points_2d = points_2d_vec[i];
        for (int j = 0; j < points_3d.size(); j++) {
            double *Rt_param_start = Rt_param + 6 * i;
//            ceres::CostFunction *costFunction =
        }
    }

}