#include <iostream>

#define _USE_MATH_DEFINES
#include <cmath>

#include <opencv2\opencv.hpp>
#include <vector>

#include<algorithm>


using namespace std;

// Matrix Matmul Function 
vector<vector<double> >changeMatrix(vector<vector<double> >A, vector<vector<double> >B)
{
	vector<vector<double> >answer;
	answer.resize(A.size());
	for (int i = 0; i < answer.size(); i++) {

		answer[i].resize(B[0].size());

	}

	for (int i = 0; i < A.size(); i++) {

		for (int j = 0; j < B[0].size(); j++) {

			for (int k = 0; k < A[0].size(); k++) {

				answer[i][j] += A[i][k] * B[k][j];

			}

		}
	}

	return answer;
}


cv::Mat problem_a_rotate_forward(cv::Mat img, double angle) {

	cv::Mat output;

	//////////////////////////////////////////////////////////////////////////////
	//                         START OF YOUR CODE                               //
	//////////////////////////////////////////////////////////////////////////////

	// (1) Define Original Pixel Value

	int origin_width = img.cols; int origin_height = img.rows;

	vector<vector<double>>origin(origin_width*origin_height, vector<double>(2, 0));


	double matrix_num = 0;

	for (int i = 0; i < origin.size(); i++) {

		if ((i % 225 == 0) && (i != 0))

		{

			matrix_num += 1;

		}

		origin[i][0] = matrix_num;

	}

	double matrix_num2 = 0;

	for (int i = 0; i < origin.size(); i++) {

		if ((i % 225 == 0) && (i != 0))

		{

			matrix_num2 = 0;

		}

		origin[i][1] = matrix_num2;

		matrix_num2 += 1;

	}

	// (2) Rotation Matrix Vector 
	
	double theta = (M_PI) / (180 / angle);

	vector<vector<double>> v;

	v = {

		{cos(theta),-sin(theta)},

		{sin(theta),cos(theta)}
	};

	// Result Vector 

	vector<vector<double>> result(origin_width * origin_height, vector<double>(2, 0));

	// (3) Matrix Matmul (Change the Pixel Value)

	double xmin = 0; double xmax = 0; double ymin = 0; double ymax = 0;

	for (int i = 0; i < result.size(); i++) {

		vector<vector<double> >A{ {origin[i][0],origin[i][1]} };
		vector<vector<double> > rotation_result = changeMatrix(A, v);

		result[i][0] = rotation_result[0][0];
		result[i][1] = rotation_result[0][1];

		xmin = min(xmin, result[i][0]);
		xmax = max(xmax, result[i][0]);

		ymin = min(ymin, result[i][1]);
		ymax = max(ymax, result[i][1]);
	

	}

	// (4) Get the New Images 

	int change_width = ceil(xmax - xmin);
	int change_height = ceil(ymax - ymin);

	for (int i = 0; i < result.size(); i++) {// 좌표 이동

		if (xmin < 0) {

			result[i][0] = result[i][0] + (-xmin);
		}

		if (ymin < 0) {

			result[i][1] = result[i][1] + (-ymin);
		
		}

	}

	output = cv::Mat::zeros(change_height, change_width, img.type());  // 새로 생성된 빈 이미지 

	int count = 0;
	
	
	while (count < result.size()) {

		for (int c = 0; c < img.channels(); c++) {

			int transition = img.at<cv::Vec3b>(origin[count][1], origin[count][0])[c]; //pixel값 (색상)
			output.at<cv::Vec3b>(result[count][1], result[count][0])[c] = transition;

		}

		count += 1;

	}
	
	//////////////////////////////////////////////////////////////////////////////
	//                          END OF YOUR CODE                                //
	//////////////////////////////////////////////////////////////////////////////

	cv::imshow("a_output", output);
	cv::waitKey(0);

	return output;

}

cv::Mat problem_b_rotate_backward(cv::Mat img, double angle) {

	cv::Mat output;

	//////////////////////////////////////////////////////////////////////////////
	//                         START OF YOUR CODE                               //
	//////////////////////////////////////////////////////////////////////////////

	////// Forward Step

	int origin_width = img.cols; int origin_height = img.rows;

	vector<vector<double>>origin(origin_width * origin_height, vector<double>(2, 0));


	double matrix_num = 0;

	for (int i = 0; i < origin.size(); i++) {

		if ((i % 225 == 0) && (i != 0))

		{

			matrix_num += 1;

		}

		origin[i][0] = matrix_num;

	}

	double matrix_num2 = 0;

	for (int i = 0; i < origin.size(); i++) {

		if ((i % 225 == 0) && (i != 0))

		{

			matrix_num2 = 0;

		}

		origin[i][1] = matrix_num2;

		matrix_num2 += 1;

	}

	// (2) Rotation Matrix Vector

	double theta = (M_PI) / (180 / angle);

	vector<vector<double>> v;

	v = {

		{cos(theta),-sin(theta)},

		{sin(theta),cos(theta)}
	};

	// Result Vector 

	vector<vector<double>> result(origin_width * origin_height, vector<double>(2, 0));

	// (3) Matrix Matmul (Change the Pixel Value)

	double xmin = 0; double xmax = 0; double ymin = 0; double ymax = 0;

	for (int i = 0; i < result.size(); i++) {

		vector<vector<double> >A{ {origin[i][0],origin[i][1]} };
		vector<vector<double> > rotation_result = changeMatrix(A, v);

		result[i][0] = rotation_result[0][0];
		result[i][1] = rotation_result[0][1];

		xmin = min(xmin, result[i][0]);
		xmax = max(xmax, result[i][0]);

		ymin = min(ymin, result[i][1]);
		ymax = max(ymax, result[i][1]);


	}

	// (4) Get the New Images 

	int change_width = ceil(xmax - xmin);
	int change_height = ceil(ymax - ymin);

	for (int i = 0; i < result.size(); i++) {// 좌표 이동

		if (xmin < 0) {

			result[i][0] = result[i][0] + (-xmin);
		}

		if (ymin < 0) {

			result[i][1] = result[i][1] + (-ymin);

		}

	}

	output = cv::Mat::zeros(change_height, change_width, img.type());  // 새로 생성된 빈 이미지 

	int x_center = change_width / 2; int y_center = change_height / 2;

	int x_diff = (change_width - origin_width) / 2;
	int y_diff = (change_height - origin_height) / 2;


	////// Backward Step


	double origin_x; double origin_y;


	for (int x = 0; x < change_width; x++) {

		for (int y = 0; y < change_height; y++) {

			double expand_y = (change_height - 1) - y - y_center; //y만 좌표 이동 

			double expand_x = x - x_center; // 좌표 변환 

			// 역행렬 곱하기

			origin_x = ((cos(theta)) * expand_x) + ((sin(theta)) * expand_y);
			origin_y = (-(sin(theta)) * expand_x) + ((cos(theta)) * expand_y);


			origin_x = origin_x + x_center - x_diff;// y
			origin_y = (origin_height - 1) - (origin_y + y_center - y_diff);  // x 

			if ((origin_x < 0) || (origin_x > (origin_width - 1)) || (origin_y < 0) || (origin_y > (origin_height - 1))) {

				continue;

			}

			for (int c = 0; c < img.channels(); c++) {

				output.at<cv::Vec3b>(y, x)[c] = img.at<cv::Vec3b>((int)origin_y, (int)origin_x)[c];

			}
		}

	}

	//////////////////////////////////////////////////////////////////////////////
	//                          END OF YOUR CODE                                //
	//////////////////////////////////////////////////////////////////////////////

	cv::imshow("b_output", output);
	cv::waitKey(0);

	return output;

}

cv::Mat problem_c_rotate_backward_interarea(cv::Mat img, double angle) {
	cv::Mat output;

	//////////////////////////////////////////////////////////////////////////////
	//                         START OF YOUR CODE                               //
	//////////////////////////////////////////////////////////////////////////////

	////// Forward Step
	int origin_width = img.cols; int origin_height = img.rows;

	vector<vector<double>>origin(origin_width * origin_height, vector<double>(2, 0));


	double matrix_num = 0;

	for (int i = 0; i < origin.size(); i++) {

		if ((i % 225 == 0) && (i != 0))

		{

			matrix_num += 1;

		}

		origin[i][0] = matrix_num;

	}

	double matrix_num2 = 0;

	for (int i = 0; i < origin.size(); i++) {

		if ((i % 225 == 0) && (i != 0))

		{

			matrix_num2 = 0;

		}

		origin[i][1] = matrix_num2;

		matrix_num2 += 1;

	}

	// (2) Rotation Matrix Vector (45 degrees)

	double theta = (M_PI) / (180 / angle);

	vector<vector<double>> v;

	v = {

		{cos(theta),-sin(theta)},

		{sin(theta),cos(theta)}
	};

	// Result Vector 

	vector<vector<double>> result(origin_width * origin_height, vector<double>(2, 0));

	// (3) Matrix Matmul (Change the Pixel Value)

	double xmin = 0; double xmax = 0; double ymin = 0; double ymax = 0;

	for (int i = 0; i < result.size(); i++) {

		vector<vector<double> >A{ {origin[i][0],origin[i][1]} };
		vector<vector<double> > rotation_result = changeMatrix(A, v);

		result[i][0] = rotation_result[0][0];
		result[i][1] = rotation_result[0][1];

		xmin = min(xmin, result[i][0]);
		xmax = max(xmax, result[i][0]);

		ymin = min(ymin, result[i][1]);
		ymax = max(ymax, result[i][1]);


	}

	// (4) Get the New Images 

	int change_width = ceil(xmax - xmin);
	int change_height = ceil(ymax - ymin);

	for (int i = 0; i < result.size(); i++) {// 좌표 이동

		if (xmin < 0) {

			result[i][0] = result[i][0] + (-xmin);
		}

		if (ymin < 0) {

			result[i][1] = result[i][1] + (-ymin);

		}

	}


	////// Backward Step

	output = cv::Mat::zeros(change_height, change_width, img.type());  // 새로 생성된 빈 이미지 

	double x_center = change_width / 2; double y_center = change_height / 2;

	double x_diff = (change_width - origin_width) / 2;
	double y_diff = (change_height - origin_height) / 2;


	output = cv::Mat::zeros(change_height, change_width, img.type());


	double origin_x; double origin_y;

	for (int x = 0; x < change_width; x++) {

		for (int y = 0; y < change_height; y++) {

			double expand_y = (change_height - 1) - y - y_center; //y만 좌표 이동 

			double expand_x = x - x_center; // 좌표 변환 

			// 역행렬 곱하기

			origin_x = ((cos(theta)) * expand_x) + ((sin(theta)) * expand_y);
			origin_y = (-(sin(theta)) * expand_x) + ((cos(theta)) * expand_y);

			origin_x = origin_x + x_center - x_diff;
			origin_y = (origin_height - 1) - (origin_y + y_center - y_diff);  

			if ((origin_x < 0) || (origin_x > (origin_width - 1)) || (origin_y < 0) || (origin_y > (origin_height - 1))) {

				continue;
			}

			////// Bilinear Interpolation Step

			// (1) Get the four near points 

			int convert_min_x = floor(origin_x); int convert_max_x = ceil(origin_x); int convert_min_y = floor(origin_y); int convert_max_y = ceil(origin_y);

			// (2) alpha & beta // p & q

			double alpha = origin_y - convert_min_y;
			double beta = 1 - alpha;

			double p = origin_x - convert_min_x;
			double q = 1 - p;


			for (int c = 0; c < img.channels(); c++) {

				output.at<cv::Vec3b>(y, x)[c] += q*beta*(img.at<cv::Vec3b>(convert_min_y, convert_min_x)[c]);//A
				output.at<cv::Vec3b>(y, x)[c] += q * alpha * (img.at<cv::Vec3b>(convert_max_y, convert_min_x)[c]);//B
				output.at<cv::Vec3b>(y, x)[c] += p * beta * (img.at<cv::Vec3b>(convert_min_y, convert_max_x)[c]);//D
				output.at<cv::Vec3b>(y, x)[c] += p * alpha * (img.at<cv::Vec3b>(convert_max_y, convert_max_x)[c]);//C

			}
	
		}

	}
	//////////////////////////////////////////////////////////////////////////////
	//                          END OF YOUR CODE                                //
	//////////////////////////////////////////////////////////////////////////////

	cv::imshow("c_output", output); cv::waitKey(0);

	return output;
}

cv::Mat Example_change_brightness(cv::Mat img, int num, int x, int y) {
	/*
	img : input image
	num : number for brightness (increase or decrease)
	x : x coordinate of image (for square part)
	y : y coordinate of image (for square part)

	*/
	cv::Mat output = img.clone();
	int size = 100;
	int height = (y + 100 > img.cols) ? img.cols : y + 100;
	int width = (x + 100 > img.rows) ? img.rows : x + 100;

	for (int i = x; i < width; i++)
	{
		for (int j = y; j < height; j++)
		{
			for (int c = 0; c < img.channels(); c++)
			{
				int t = img.at<cv::Vec3b>(i, j)[c] + num;
				output.at<cv::Vec3b>(i, j)[c] = t > 255 ? 255 : t < 0 ? 0 : t;
			}
		}

	}
	cv::imshow("output1", img);
	cv::imshow("output2", output);
	cv::waitKey(0);
	return output;
}

int main(void) {

	double angle = 45.0f;

	cv::Mat input = cv::imread("lena.jpg");
	//Fill problem_a_rotate_forward and show output

	problem_a_rotate_forward(input, angle);
	//Fill problem_b_rotate_backward and show output

	problem_b_rotate_backward(input, angle);
	//Fill problem_c_rotate_backward_interarea and show output

	problem_c_rotate_backward_interarea(input, angle);
	//Example how to access pixel value, change params if you want

	Example_change_brightness(input, 100, 50, 125);


}