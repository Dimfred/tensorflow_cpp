#include <objdet.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define THRESHOLD 0.8


int main(int argc, char* argv[])
{
	if (argc != 4)
	{
		std::cout << "Error! Usage: <path/to_saved_model> <path/to_input/image.jpg> <path/to/output/image.jpg>" << std::endl;
		return 1;
	}

	const string model_path = argv[1];
	const string test_image_file  = argv[2];
	const string test_prediction_image = argv[3];

	// load and predict
	Model model(model_path);
	auto preds = model(test_image_file);

	// draw results
	cv::Mat img = cv::imread(test_image_file, cv::IMREAD_COLOR);
	if (img.empty()) {
		std::cout << "CVImageRead: failed" << std::endl;
		exit(1);
	}

	cv::Size size = img.size();
	int height = size.height;
	int width = size.width;

	for (Prediction pred : preds)
	{
		if (pred.score < THRESHOLD)
			continue;

		int xmin = (int)(pred.x * width);
		int ymin = (int)(pred.y * height);
		int w = (int)(pred.w * width) - xmin;
		int h = (int)(pred.h * height) - ymin;
		cv::Rect rect(xmin, ymin, w, h);
		cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 2);
	}

	imwrite(test_prediction_image, img);
}
