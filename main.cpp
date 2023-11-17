//
// Created by ldx on 23-11-15.
//
#include<iostream>
#include"include/CamCalibration.h"
int main(){
    std::string chessBoard_path = "../chess_board_img";
    CamCalibration camCalibration(chessBoard_path,8,11,0.02,41);
    camCalibration.Calibrate();

    return 0;
}