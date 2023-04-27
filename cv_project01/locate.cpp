#include"locate.h"
#include<vector>
#include"imageBaseOP.h"
#define pi 3.14159

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
void cannyEdgeDetection(Mat img, Mat& result,int guaSize, 
    double hightThres, double lowThres) 
{
    // 高斯滤波->中值滤波
    Rect rect; // IOU区域
    cv::Mat filterImg = cv::Mat::zeros(img.rows, img.cols, CV_64FC1);
    img.convertTo(img, CV_64FC1);
    result = cv::Mat::zeros(img.rows, img.cols, CV_64FC1);
    int guassCenter = guaSize / 2; // 高斯核的中心 // (2* guassKernelSize +1) * (2*guassKernelSize+1)高斯核大小
    double sigma = 1;   // 方差大小
    cv::Mat guassKernel = cv::Mat::zeros(guaSize, guaSize, CV_64FC1);
    for (int i = 0; i < guaSize; i++) {
        for (int j = 0; j < guaSize; j++) {
            guassKernel.at<double>(i, j) = (1.0 / (2.0 * pi * sigma * sigma)) *
                (double)exp(-(((double)pow((i - (guassCenter + 1)), 2) + (double)pow((j - (guassCenter + 1)), 2)) / (2.0 * sigma * sigma)));
            // std::cout<<guassKernel.at<double>(i, j) << " ";
        }
        // std::cout<<std::endl;
    }
    cv::Scalar sumValueScalar = cv::sum(guassKernel);
    double sum = sumValueScalar.val[0];
    std::cout << sum << std::endl;
    guassKernel = guassKernel / sum;
    //    for(int i = 0; i< guaSize; i++){
    //        for(int j = 0; j < guaSize; j++){
    //            std::cout<<guassKernel.at<double>(i, j) << " ";
    //        }
    //        std::cout<<std::endl;
    //    }
    for (int i = guassCenter; i < img.rows - guassCenter; i++) {
        for (int j = guassCenter; j < img.cols - guassCenter; j++) {
            rect.x = j - guassCenter;
            rect.y = i - guassCenter;
            rect.width = guaSize;
            rect.height = guaSize;
            filterImg.at<double>(i, j) = cv::sum(guassKernel.mul(img(rect))).val[0];
            // std::cout<<filterImg.at<double>(i,j) << " ";
        }
        // std::cout<<std::endl;
    }
    cv::Mat guassResult;
    filterImg.convertTo(guassResult, CV_8UC1);
    cv::imshow("guass-result", guassResult);
    // std::cout<<cv::sum(guassKernel).val[0]<<std::endl;
    // 计算梯度,用sobel算子
    cv::Mat gradX = cv::Mat::zeros(img.rows, img.cols, CV_64FC1); // 水平梯度
    cv::Mat gradY = cv::Mat::zeros(img.rows, img.cols, CV_64FC1); // 垂直梯度
    cv::Mat grad = cv::Mat::zeros(img.rows, img.cols, CV_64FC1);  // 梯度幅值
    cv::Mat thead = cv::Mat::zeros(img.rows, img.cols, CV_64FC1); // 梯度角度
    cv::Mat locateGrad = cv::Mat::zeros(img.rows, img.cols, CV_64FC1); //区域
    // x方向的sobel算子
    cv::Mat Sx = (cv::Mat_<double>(3, 3) << -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
        );
    // y方向sobel算子
    cv::Mat Sy = (cv::Mat_<double>(3, 3) << 1, 2, 1,
        0, 0, 0,
        -1, -2, -1
        );
    // 计算梯度赋值和角度
    for (int i = 1; i < img.rows - 1; i++) {
        for (int j = 1; j < img.cols - 1; j++) {
            // 卷积区域 3*3
            rect.x = j - 1;
            rect.y = i - 1;
            rect.width = 3;
            rect.height = 3;
            cv::Mat rectImg = cv::Mat::zeros(3, 3, CV_64FC1);
            filterImg(rect).copyTo(rectImg);
            // 梯度和角度
            gradX.at<double>(i, j) += cv::sum(rectImg.mul(Sx)).val[0];
            gradY.at<double>(i, j) += cv::sum(rectImg.mul(Sy)).val[0];
            grad.at<double>(i, j) = sqrt(pow(gradX.at<double>(i, j), 2) + pow(gradY.at<double>(i, j), 2));
            thead.at<double>(i, j) = atan(gradY.at<double>(i, j) / gradX.at<double>(i, j));
            // 设置四个区域
            if (0 <= thead.at<double>(i, j) <= (pi / 4.0)) {
                locateGrad.at<double>(i, j) = 0;
            }
            else if (pi / 4.0 < thead.at<double>(i, j) <= (pi / 2.0)) {
                locateGrad.at<double>(i, j) = 1;
            }
            else if (-pi / 2.0 <= thead.at<double>(i, j) <= (-pi / 4.0)) {
                locateGrad.at<double>(i, j) = 2;
            }
            else if (-pi / 4.0 < thead.at<double>(i, j) < 0) {
                locateGrad.at<double>(i, j) = 3;
            }
        }
    }
    // debug
    cv::Mat tempGrad;
    grad.convertTo(tempGrad, CV_8UC1);
    imshow("grad", tempGrad);
    // 梯度归一化
    double gradMax;
    cv::minMaxLoc(grad, &gradMax); // 求最大值
    if (gradMax != 0) {
        grad = grad / gradMax;
    }
    // debug
    cv::Mat tempGradN;
    grad.convertTo(tempGradN, CV_8UC1);
    imshow("gradN", tempGradN);

    // 双阈值确定
    cv::Mat caculateValue = cv::Mat::zeros(img.rows, img.cols, CV_64FC1); // grad变成一维
    cv::resize(grad, caculateValue, cv::Size(1, (grad.rows * grad.cols)));
    // caculateValue.convertTo(caculateValue, CV_64FC1);
    cv::sort(caculateValue, caculateValue, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING); // 升序
    long long highIndex = img.rows * img.cols * hightThres;
    double highValue = caculateValue.at<double>(highIndex, 0); // 最大阈值
    // debug
    // std::cout<< "highValue: "<<highValue<<" "<<  caculateValue.cols << " "<<highIndex<< std::endl;

    double lowValue = highValue * lowThres; // 最小阈值
    // 3.非极大值抑制， 采用线性插值
    for (int i = 1; i < img.rows - 1; i++) {
        for (int j = 1; j < img.cols - 1; j++) {
            // 八个方位
            double N = grad.at<double>(i - 1, j);
            double NE = grad.at<double>(i - 1, j + 1);
            double E = grad.at<double>(i, j + 1);
            double SE = grad.at<double>(i + 1, j + 1);
            double S = grad.at<double>(i + 1, j);
            double SW = grad.at<double>(i - 1, j - 1);
            double W = grad.at<double>(i, j - 1);
            double NW = grad.at<double>(i - 1, j - 1);
            // 区域判断，线性插值处理
            double tanThead; // tan角度
            double Gp1; // 两个方向的梯度强度
            double Gp2;
            // 求角度，绝对值
            tanThead = abs(tan(thead.at<double>(i, j)));
            switch ((int)locateGrad.at<double>(i, j)) {
            case 0:
                Gp1 = (1 - tanThead) * E + tanThead * NE;
                Gp2 = (1 - tanThead) * W + tanThead * SW;
                break;
            case 1:
                Gp1 = (1 - tanThead) * N + tanThead * NE;
                Gp2 = (1 - tanThead) * S + tanThead * SW;
                break;
            case 2:
                Gp1 = (1 - tanThead) * N + tanThead * NW;
                Gp2 = (1 - tanThead) * S + tanThead * SE;
                break;
            case 3:
                Gp1 = (1 - tanThead) * W + tanThead * NW;
                Gp2 = (1 - tanThead) * E + tanThead * SE;
                break;
            default:
                break;
            }
            // NMS -非极大值抑制和双阈值检测
            if (grad.at<double>(i, j) >= Gp1 && grad.at<double>(i, j) >= Gp2) {
                //双阈值检测
                if (grad.at<double>(i, j) >= highValue) {
                    grad.at<double>(i, j) = highValue;
                    result.at<double>(i, j) = 255;
                }
                else if (grad.at<double>(i, j) < lowValue) {
                    grad.at<double>(i, j) = 0;
                }
                else {
                    grad.at<double>(i, j) = lowValue;
                }

            }
            else {
                grad.at<double>(i, j) = 0;
            }
        }
    }
    // NMS 和算阈值检测后的梯度图
    cv::Mat tempGradNMS;
    grad.convertTo(tempGradNMS, CV_8UC1);
    imshow("gradNMS", tempGradNMS);

    // 4.抑制孤立低阈值点 3*3. 找到高阈值就255
    for (int i = 1; i < img.rows - 1; i++) {
        for (int j = 1; j < img.cols - 1; j++) {
            if (grad.at<double>(i, j) == lowValue) {
                // 3*3区域找强梯度
                rect.x = j - 1;
                rect.y = i - 1;
                rect.width = 3;
                rect.height = 3;
                for (int i1 = 0; i1 < 3; i1++) {
                    for (int j1 = 0; j1 < 3; j1++) {
                        if (grad(rect).at<double>(i1, j1) == highValue) {
                            result.at<double>(i, j) = 255;
                            std::cout << result.at<double>(i, j);
                            break;
                        }
                    }
                }
            }
        }
    }
    // 结果
    result.convertTo(result, CV_8UC1);
    imshow("result", result);


}