#include"imageBaseOP.h"
#include<vector>
#include"locate.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;
int main()
{
	Mat src = QF_ReadImage("C:/images/measure/20.bmp");
	if (src.empty()) {
		cout << "\033[31m invalid path of the input image \033[0m";
		return -1;
	}
	Mat dst;
	//medianBlur(src, src, 19);
	GaussianBlur(src, dst, cv::Size(5, 5), 3, 3);
	cvtColor(src, src, COLOR_BGR2GRAY);
	//blur(src, src, Size(5, 5));
	Canny(src, dst, 150, 100);
	imwrite("C:/images/ans/gauss5_20.bmp", dst);

	//QF_ShowImage(dst, "01", 0);
	
	return 0;
}