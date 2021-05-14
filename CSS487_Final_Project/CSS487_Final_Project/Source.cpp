// Source code for CSS 487's final project
// Author: Carl Howing, Max Nguyen
//
// Implementing computer vision concepts and OpenCV, our program accesses 
// the user’s webcam to implement face detections, then allows users to 
// interact with the GUI to add a filter in real-time.

#include <iostream>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
//#include "opencv2/face.hpp"
#include <vector>
#include <fstream>

using namespace std;
using namespace cv;
//using namespace cv::face;

void overlayFilter(Mat& sourceImage, Mat& overlay, int row, int col, double scale, Rect area);
bool eyeDetection(CascadeClassifier classifier, Mat& frame, const Mat& resizedFrame, const Rect& area, const double& scale, Point& center);
void threshold(Mat& source, Mat& output, int commonVal, int maxVal);
void overlayTest(Mat& source, Mat& overlay, Mat& frame);
uchar checkNN(Mat& source, Mat& output, int pixelRow, int pixelCol, int size);
Rect guiButton(Mat frame, string text, Point btnPos);
void mouseEventHandler(int event, int x, int y, int, void*);

// GUI buttons
Rect addGlacierButton;
bool activateGlacier = false;
Rect headbandButton;
bool activateHeadband = false;
Rect sunglassesButton;
bool activateSunglasses = false;

// main method
// Preconnditions: three input images exists in the same directory of this source file
//					and are correctly formatted JPG images
// Postcondition: 
int main()
{
	// Read the Snapchat-like filters
	Mat overlay = imread("headband_overlay.jpg");
	Mat sunglasses = imread("sunglasses_overlay_2.jpg");

	// Scale factor for face and eye detection
	double scale = 2.0;

	// Initialize the classifiers
	CascadeClassifier faceCascade, leftEyeCascade, rightEyeCascade;

	// Load classifiers for face detection
	try {
		//faceCascade.load("C:\\OpenCV\\etc\\haarcascades\\haarcascade_frontalface_alt.xml");		// for cj
		//leftEyeCascade.load("C:\\OpenCV\\etc\\haarcascades\\haarcascade_mcs_lefteye.xml");
		//rightEyeCascade.load("C:\\OpenCV\\etc\\haarcascades\\haarcascade_righteye_2splits.xml");
		faceCascade.load("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml");	// for max
		leftEyeCascade.load("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_lefteye_2splits.xml");
		rightEyeCascade.load("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_righteye_2splits.xml");

		//// For eye classifiers comparision
		// CascadeClassifier eyeCascade, eyeCascade2;
		//eyeCascade.load("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml");
		//eyeCascade2.load("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml");
	}

	// display error message if file did not load
	catch (Exception e) {
		if (faceCascade.empty()) {
			cout << "Could not load classifiers" << endl;
			exit(1);
		}
	}

	// Sets the Video Capture Object
	VideoCapture cap(0);

	//Perminent Loop
	for (;;)
	{
		//Mat objects
		Mat frame;
		Mat greyscale;
		Mat resizedFrame;

		//Reads in each Image to Frame
		cap >> frame;

		// For testing with static image
		//frame = imread("static_input_image.jpg");

		//Converts the Image frame to grayscale
		cvtColor(frame, greyscale, COLOR_BGR2GRAY);

		//Resizes the frame to increase performance
		resize(greyscale, resizedFrame, Size(greyscale.size().width / scale, greyscale.size().height / scale));

		// To improve the contrast and brightneses of image
		equalizeHist(resizedFrame, resizedFrame);

		//Vector of Faces
		vector<Rect> faces;

		//Calls the face cascade classifier
		faceCascade.detectMultiScale(resizedFrame, faces, 1.1, 3, 0, Size(30, 30));

		//Loops through all the faces in the vector
		for (Rect area : faces)
		{
			//cout << area.x << '\t';
			//cout << area.y;
			//cout << endl;

			//Sets the color of the bounding box
			Scalar drawColor = (Scalar(0, 255, 0));

			//Draws the bounding box on the frame
			rectangle(frame, Point(cvRound(area.x * scale), cvRound(area.y * scale)),
				Point(cvRound((area.x + area.width - 1) * scale), cvRound((area.y + area.height - 1) * scale)), drawColor);


			//********************** Start Eye Detection **********************\\
			//*****************************************************************\\

			// Store the face as a matrix
			Mat faceImg = resizedFrame(area);

			/*
			// Attempt to crop face image for faster eye detection
			int leftCol = cvRound(faceImg.cols * 0.1f);
			int topRow = cvRound(faceImg.rows * 0.25f);
			int widthCol = cvRound(faceImg.cols * 0.4f);
			int heightRow = cvRound(faceImg.rows * 0.3f);
			int rightCol = cvRound(faceImg.cols * 0.5f);


			//resize(topLeftFace, topLeftFace, Size(frame.cols, frame.rows), 1);
			Mat topLeftFace = faceImg(Rect(leftCol, topRow, widthCol, heightRow));
			//imshow("left", topLeftFace);
			Mat topRightFace = faceImg(Rect(rightCol, topRow, widthCol, heightRow));
			//imshow("right", topRightFace);
			*/

			// Initialize the center of each eye
			Point leftEye, rightEye;

			//bool leftEyeDetected = eyeDetection(eyeCascade, frame, faceImg, area, scale, leftEye, faceImg);
			bool leftEyeDetected = eyeDetection(leftEyeCascade, frame, faceImg, area, scale, leftEye);
			bool rightEyeDetected = eyeDetection(rightEyeCascade, frame, faceImg, area, scale, leftEye);

			// If both eyes are detected and user clicks on "Add Sunglasses" button
			if (leftEyeDetected && rightEyeDetected && activateSunglasses) {

				// Resize the sunglasses image
				Mat resizedFilter;
				resize(sunglasses, resizedFilter, Size(faceImg.rows, faceImg.cols) * 2, 3);

				// Add the overlay effect according to the coordinates of both eyes detected
				//overlayFilter(frame, resizedFilter, (leftEye.y + rightEye.y) / 2 + 20, (leftEye.x + rightEye.x) / 2 - 95, 1, area);
				overlayFilter(frame, resizedFilter, (leftEye.y + rightEye.y) / 2 + 17, (leftEye.x + rightEye.x) / 2 - 40, scale, area);
			}

			//Mat resizedFace = greyscale;
			//Mat gBlur;
			//resize(frame, resizedFace, Size(frame.cols / 5, frame.rows / 5), 3);
			//GaussianBlur(resizedFace, gBlur, Size(31, 31), 4);
			//resize(gBlur, resizedFace, Size(frame.cols, frame.rows), 3);
			//imshow("GB", resizedFace);			

			// If user clicks on "Add Background" button
			if (activateGlacier) {

				Mat backgroundImage = imread("glacierbay.jpg");

				// Resize the background image
				Mat resizedBackground;
				resize(backgroundImage, resizedBackground, Size(frame.cols, frame.rows), 3);

				// Add a virtual background in realtime
				overlayTest(greyscale, resizedBackground, frame);
			}

			// If user clicks on "Add Headband" button
			if (activateHeadband) {

				// Resize the overlay to fit the face
				Mat resized;
				resize(overlay, resized, Size(area.width * scale, area.height * scale), 3);

				// Add the headband overlay effect according the coordinate of the face detected
				overlayFilter(frame, resized, area.y, area.x, scale, area);
			}

			// Draw buttons
			addGlacierButton = guiButton(frame, "Add Background", Point(8, 8));
			headbandButton = guiButton(frame, "Add Headband", Point(addGlacierButton.x, addGlacierButton.y + addGlacierButton.height));
			sunglassesButton = guiButton(frame, "Add Sunglasses", Point(8, headbandButton.y + headbandButton.height));

			//faceRecognition(faceImg);

		}

		//faceRecognition(faceImg);

		imshow("Webcam Frame", frame);

		// Get mouse click event
		setMouseCallback("Webcam Frame", mouseEventHandler, 0);

		if (waitKey(30) >= 0)
		{
			break;
		}


	}

	return 0;

}

// -------------------------------------- Background Overlay ---------------------------------------
//overlayTest
//overlayTest using a thresholding technique for segmentation
//the threshold is calculated using nearest neighbors
//The result of the image creates a mask, the mask is used 
//for overlaying the background
void overlayTest(Mat& source, Mat& overlay, Mat& frame)
{

	threshold(source, source, 100, 255);
	for (int i = 0; i < frame.rows - 1; i++)
	{
		for (int j = 0; j < frame.cols - 1; j++)
		{
			if (int(source.at<uchar>(i, j)) == 0)
			{
				//int overlayR = i % overlay.rows;
				//int overlayL = j % overlay.cols;
				//frame.at<Vec3b>(i, j) = overlay.at<uchar>(overlayR, overlayL);
				frame.at<Vec3b>(i, j) = overlay.at<Vec3b>(i, j);
			}
		}
	}

}

//threshold
//threshold is a function that uses commonVal and the maxVal
//if a pixel value is less then the commonVal, the pixel is 
//set to maxval.
//The result of the output image is a binary image
void threshold(Mat& source, Mat& output, int commonVal, int maxVal)
{
	for (int row = 0; row < source.rows; row++)
	{
		for (int col = 0; col < source.cols; col++)
		{
			uchar temp = checkNN(source, output, row, col, 3);
			if (temp < uchar(commonVal))
			{
				source.at<uchar>(row, col) = uchar(maxVal);

			}
			else
			{
				source.at<uchar>(row, col) = uchar(0);
			}

		}
	}
}

//checkNN
//checkNN checks the neighbouring pixels to determine what the value of the current
//pixel should be. The neighboring pixels are summed and then divided by the size to 
//get an average.
uchar checkNN(Mat& source, Mat& output, int pixelRow, int pixelCol, int size)
{
	int total = 0;
	for (int row = 0; row < size; row++)
	{
		for (int col = 0; col < size; col++)
		{
			if (pixelRow + row < 0 || pixelRow + row > source.rows - 1 || pixelCol + col == 0 || pixelCol + col > source.cols - 1)
			{
				total += 0;
			}
			else
			{
				total += int(source.at<uchar>(pixelRow + row, pixelCol + col));
			}

		}
	}
	return uchar(total / (size * size));

}


//GraySub
//GraySub computes the difference of two grayscale image
//this function attempts to create a mask based on the differences
//of two images.
void graySub(Mat& source, Mat& background, Mat& mask)
{
	//loop through the rows
	for (int rows = 0; rows < source.rows; rows++)
	{
		//loop through the cols
		for (int cols = 0; cols < source.cols; cols++)
		{
			//variables for containing the two pixel values
			uchar backgroundPixel = background.at<uchar>(rows, cols);
			uchar sourcePixel = source.at<uchar>(rows, cols);
			//checks the difference and thresholds
			if (abs(backgroundPixel - sourcePixel < 5))
			{
				mask.at<uchar>(rows, cols) = 0;
			}
			else
			{
				mask.at<uchar>(rows, cols) = 255;
			}


		}
	}
}

//backgroundSub
//backgroundSub computes the difference of two color image
//this function attempts to create a mask based on the differences
//of two images.
void backgroundSub(Mat& source, Mat& background, Mat& mask)
{
	//loop through the rows
	for (int rows = 0; rows < source.rows; rows++)
	{
		//loop through the cols
		for (int cols = 0; cols < source.cols; cols++)
		{
			//variables containing the BGR values
			Vec3b backgroundPixel = background.at<Vec3b>(rows, cols);
			Vec3b sourcePixel = source.at<Vec3b>(rows, cols);
			//Subtracts the values of the source and background image
			if (abs(backgroundPixel[0] - sourcePixel[0]) < 5 &&
				abs(backgroundPixel[1] - sourcePixel[1]) < 5 &&
				abs(backgroundPixel[2] - sourcePixel[2]) < 5)
			{
				//Sets the value of the mask
				mask.at<uchar>(rows, cols) = 0;
			}
			else
			{
				//Sets the value of the mask
				mask.at<uchar>(rows, cols) = 255;
			}


		}
	}
}


// --------------------------------------- Snapchat-like Filter Overlay --------------------------------
//overlayFilter
//overlayFilter takes the source image and the overlay image
//the row and col is used make the calculations for the scaling of 
//the overlay. The rect object is used to determine the dimensions of the face.
void overlayFilter(Mat& sourceImage, Mat& overlay, int row, int col, double scale, Rect area)
{
	int scaleY = cvRound(row * scale);
	int scaleX = cvRound(col * scale);
	//if (row > overlay.rows)
	//{
	for (int i = 0; i < overlay.rows; i++)
	{
		for (int j = 0; j < overlay.cols; j++)
		{
			Vec3b newPixel = overlay.at<Vec3b>(i, j);
			if (newPixel[0] < 254 || newPixel[1] < 254 || newPixel[2] < 254)
			{
				sourceImage.at<Vec3b>(abs(scaleY - area.height) + i, scaleX + j) = newPixel;
			}
		}
	}
	//}

}


// -------------------------------------- Eye Detection --------------------------------------------------

// eyeDetection
// eyeDetection uses the specified classifer to detect the eyes based on the 
// the results of the face detection. The face detector returns a Rect object
// the Rect object creates the new search space for the classifier to detect the eyes.
// the scale value is used to scale up the results, since the face was scaled down for 
// computational effeciency.
// Preconditions: Cascade Classifier used for eye detection, the real-time window frame,
//					the resized window frame, the face image, the scale factor, 
//					and the center point of the eye needed to be detected
// Postcondition: Implements eye detection and mark it with a red circle,
//					returns a boolean of whether an eye is detected
bool eyeDetection(CascadeClassifier classifier, Mat& frame, const Mat& resizedFrame, const Rect& area, const double& scale, Point& center) {

	//imshow("Resized", resizedFrame);

	// Boolean for detected eyes
	bool eyeDetected = false;

	//Initialize eyeVector
	vector<Rect> eyesVector;

	// Radius of the eye detected
	int radius;

	//detects eyes using haarcascade 
	classifier.detectMultiScale(resizedFrame, eyesVector, 1.1, 7, 0 | CASCADE_SCALE_IMAGE, Size(20, 20));

	for (Rect eye : eyesVector) {
		//double centerP = (eye.width + eye.height) * .5;

		// Calculate the center coordinate and radius of the eye detected
		center.x = cvRound((area.x + eye.x + eye.width * 0.5) * scale);
		center.y = cvRound((area.y + eye.y + eye.height * 0.5) * scale);
		radius = cvRound((eye.width + eye.height) * .25);

		// Draw red circle around the eye
		circle(frame, center, radius, Scalar(0, 0, 255), 2);

		// Return true if an eye is detected
		if (eye.width > 0)
			eyeDetected = true;
	}

	return eyeDetected;
}


// --------------------------------------- GUI Buttons and Mouse Event --------------------------------------

// addText
// Preconditions: Real-time window, bottom-left corner of the text, color of text,
//					default font scale factor, thickness of lines to render text, font
// Postcondition: Adds text to real-time window on the top-left by default, 
//					and returns the rectangle boundary around text
Rect addText(Mat frame, string text, Point textOrg, Scalar color, float fontScale = 0.6f, int thickness = 1, int fontFace = FONT_HERSHEY_SIMPLEX)
{

	// Calcaluate the text size & baseline.
	int baseline = 0;
	Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
	baseline += thickness;

	if (textOrg.y >= 0) {
		// Move text coordinate down by one row
		textOrg.y += textSize.height;
	}
	// If y-coordinate input is negative, adjust the text coordinate from the bottom
	else
		textOrg.y += frame.rows - baseline + 1;

	// Get the rectangle boundary around the text
	Rect boundary = Rect(textOrg.x, textOrg.y - textSize.height, textSize.width, baseline + textSize.height);

	// Draw the text string
	putText(frame, text, textOrg, fontFace, fontScale, color, thickness, LINE_AA);

	// Let the user know how big their text is, in case they want to arrange things.
	return boundary;
}


// guiButton
// Pre-conditions: the frame of the image, text inside the button,
//					position of the upper left corner of the button
// Post-condition: a button on the real-time frame
Rect guiButton(Mat frame, string text, Point btnCoord) {

	int border = 8;

	// calculate the position of the text
	Point textPos = Point(btnCoord.x + border, btnCoord.y + border);

	// get the boundary surrounding the text
	Rect boundary = addText(frame, text, textPos, CV_RGB(0, 0, 0));

	// draw a rectangle around text
	Rect fillBox = Rect(boundary.x - border, boundary.y - border, boundary.width + 2 * border, boundary.height + 2 * border);

	// fill button with semi-transparent background
	Mat background = frame(boundary);
	background += CV_RGB(90, 90, 90);

	// draw a gray border
	rectangle(frame, boundary, CV_RGB(200, 200, 200), 1, LINE_AA);

	// display text on button
	addText(frame, text, textPos, CV_RGB(0, 0, 0));

	return boundary;
}

// isButtonClicked
// Preconditions: the point where the user clicked on left mouse,
//					the position of the button
// Postcondition: returns true if the coordinate of where the user left-clicked
//					is inside the bound of the button
bool isButtonClicked(const Point clickPt, const Rect button)
{
	if (clickPt.x >= button.x && clickPt.x <= (button.x + button.width - 1))
		if (clickPt.y >= button.y && clickPt.y <= (button.y + button.height - 1))
			return true;

	return false;
}

// mouseEventHandler
// Preconditions: user's mouse event
//					coordinate of user's mouse click
// Postcondition: activate events according to which button the user lef-click on
void mouseEventHandler(int event, int x, int y, int, void*) {

	// if the user doesn't left-click, exit current function
	if (event != EVENT_LBUTTONDOWN)
		return;

	// if user clicks on "Add Background" button
	Point clickPoint = Point(x, y);
	if (isButtonClicked(clickPoint, addGlacierButton))
		activateGlacier = !activateGlacier;

	// if use clicks on "Add Headband" button
	if (isButtonClicked(clickPoint, headbandButton))
		activateHeadband = !activateHeadband;

	// if use clicks on "Add Sunglasses" button
	if (isButtonClicked(clickPoint, sunglassesButton))
		activateSunglasses = !activateSunglasses;

}

