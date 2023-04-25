#include"locate.h"
#include<vector>
#include"imageBaseOP.h"

using namespace std;
void QF_findLine(Mat src)
{
	//转换为灰度图
	Mat src_gray;
	Mat dst;
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	//分割图像,平均取图像行数的1/10作为点集
	int height = src_gray.rows;
	int width = src_gray.cols;
	int rowSet_size = height / 10;//此处调整点集大小

	Mat kernel = (Mat_<int>(1, 3) << 1, 0, -1);//线的方向:从左往右
	Mat convl;//保存卷积结果
	//填充矩阵 将convl填充为和src一样大小的矩阵
	Mat add = Mat::zeros(rowSet_size - 1, width, CV_8UC1);
	//遍历每十行做一次卷积
	for (int i = 0; i < height; i += rowSet_size)
	{
		Mat tmp_row = src_gray.row(i);//取一行
		Mat tmp_con;//保存一行的卷积结果
		filter2D(tmp_row, tmp_con, src_gray.depth(), kernel);
		//将一行的卷积结果加到总的上去
		vconcat(convl, tmp_con, convl);
		vconcat(convl, add, convl);//填充
	}
	//如果行数不是10的倍数的话最后会多几行,所以要删除掉
	//但是多出的元素都是零,不影响最后的拟合结果,先看看行不行
	while (convl.rows != height)
	{

	}
	vector<Point2f> line_pix;//创建一个保存梯度极值点的数组
	for (int i = 0; i < height; i++)//遍历dst将极值点保存在line_pix中
	{
		for (int j = 0; j < width; j++)
		{
			if (dst.at<uchar>(i, j) == 255)
			{
				Point pt = Point(j, i);
				line_pix.push_back(pt);
			}
		}
	}
	Vec4f line_para;
	fitLineRansac(line_pix, line_para, 1000, 1., -7., 7.);
	//获取点斜式的点和斜率
	cv::Point point0;
	point0.x = line_para[2];
	point0.y = line_para[3];

	double k = line_para[1] / line_para[0];

	//计算直线的端点(y = k(x - x0) + y0)
	cv::Point point1, point2;
	point1.x = 0;
	point1.y = k * (0 - point0.x) + point0.y;
	point2.x = 640;
	point2.y = k * (640 - point0.x) + point0.y;

	cv::line(src, point1, point2, cv::Scalar(0, 255, 0), 20);

}
//Ransac拟合
void fitLineRansac(const std::vector<cv::Point2f>& points,
	cv::Vec4f& line,
	int iterations = 1000,
	double sigma = 1.,
	double k_min = -7.,
	double k_max = 7.)
{
	unsigned int n = points.size();

	if (n < 2)
	{
		return;
	}

	cv::RNG rng;
	double bestScore = -1.;
	for (int k = 0; k < iterations; k++)
	{
		int i1 = 0, i2 = 0;
		while (i1 == i2)
		{
			i1 = rng(n);
			i2 = rng(n);
		}
		const cv::Point2f& p1 = points[i1];
		const cv::Point2f& p2 = points[i2];

		cv::Point2f dp = p2 - p1;//直线的方向向量
		dp *= 1. / norm(dp);
		double score = 0;

		if (dp.y / dp.x <= k_max && dp.y / dp.x >= k_min)
		{
			for (int i = 0; i < n; i++)
			{
				cv::Point2f v = points[i] - p1;
				double d = v.y * dp.x - v.x * dp.y;//向量a与b叉乘/向量b的摸.||b||=1./norm(dp)
				//score += exp(-0.5*d*d/(sigma*sigma));//误差定义方式的一种
				if (fabs(d) < sigma)
					score += 1;
			}
		}
		if (score > bestScore)
		{
			line = cv::Vec4f(dp.x, dp.y, p1.x, p1.y);
			bestScore = score;
		}
	}
}

//直接梯度找边
void myLocate(Mat src0, Mat dst)
{
	
	Mat src = src0.clone();
	cvtColor(src, src, COLOR_BGR2GRAY);
	//构造sobel算子
	Mat kernel = (Mat_<int>(3, 3) << -1, 0, 1, -2, 0, 2, -3, 0, 3);
	//卷积
	filter2D(src, dst, src.depth(), kernel);
	int row = dst.rows;
	int col = dst.cols;
	//对dst进行二值化 大于100的全部赋值为255,其余的为0
	//threshold(dst, dst, 100, 255, THRESH_BINARY);
	//vector<Point> line_pix;//创建一个保存梯度极值点的数组
	//for (int i = 0; i < row; i++)//遍历dst将极值点保存在line_pix中
	//{
	//	for (int j = 0; j < col; j++)
	//	{
	//		if (dst.at<uchar>(i, j) == 255)
	//		{
	//			Point pt = Point(j, i);
	//			line_pix.push_back(pt);
	//		}
	//	}
	//}
	////在彩色图像上绘制 处理的时候只能用灰度图,用彩色图会产生bug
	////绘制的时候如果是灰度图看不出来
	//Mat src_bgr = src.clone();
	//cvtColor(src_bgr, src_bgr, COLOR_GRAY2BGR);
	////遍历line_pix并连接每两个像素点以绘制边缘
	//for (int i = 0; i < line_pix.size() - 1; i++)
	//{
	//	//cout << line_pix[i] << " ";
	//	line(src_bgr, line_pix[i], line_pix[i + 1], Scalar(0, 0, 255), 1);
	//}
	////Rect rec(100, 100, 800, 500);
	////rectangle(src_bgr, rec, Scalar(0, 0, 255), 30);
	//QF_ShowImage(src_bgr, "01", 0);
}

//改进的canny边缘识别算子
void my_canny(Mat src, Mat dst)
{
	cvtColor(src, src, COLOR_BGR2GRAY);
	
}