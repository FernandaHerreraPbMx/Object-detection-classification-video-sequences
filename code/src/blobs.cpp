/* Applied Video Analysis of Sequences (AVSA)
 *
 *	LAB2: Blob detection & classification
 *	Lab2.0: Sample Opencv project
 *
 *
 * Authors: José M. Martínez (josem.martinez@uam.es), Paula Moral (paula.moral@uam.es), Juan C. San Miguel (juancarlos.sanmiguel@uam.es)
 * Maria Fernanda Herrera, David Savary
 */

#include "blobs.hpp"

/**
 *	Draws blobs with different rectangles on the image 'frame'. All the input arguments must be
 *  initialized when using this function.
 *
 * \param frame Input image
 * \param pBlobList List to store the blobs found
 * \param labelled - true write label and color bb, false does not wirite label nor color bb
 *
 * \return Image containing the draw blobs. If no blobs have to be painted
 *  or arguments are wrong, the function returns a copy of the original "frame".
 *
 */
 Mat paintBlobImage(cv::Mat frame, std::vector<cvBlob> bloblist, bool labelled)
{
	cv::Mat blobImage;
	//check input conditions and return original if any is not satisfied
	//...
	frame.copyTo(blobImage);

	//required variables to paint
	//...

	//paint each blob of the list
	for(int i = 0; i < bloblist.size(); i++)
	{
		cvBlob blob = bloblist[i]; //get ith blob
		//...
		Scalar color;
		std::string label="";
		switch(blob.label){
		case PERSON:
			color = Scalar(255,0,0);
			label="PERSON";
			break;
		case CAR:
					color = Scalar(0,255,0);
					label="CAR";
					break;
		case OBJECT:
					color = Scalar(0,0,255);
					label="OBJECT";
					break;
		default:
			color = Scalar(255, 255, 255);
			label="UNKOWN";
		}

		Point p1 = Point(blob.x, blob.y);
		Point p2 = Point(blob.x+blob.w, blob.y+blob.h);

		rectangle(blobImage, p1, p2, color, 1, 8, 0);
		if (labelled)
			{
			rectangle(blobImage, p1, p2, color, 1, 8, 0);
			putText(blobImage, label, p1, FONT_HERSHEY_SIMPLEX, 0.5, color);
			}
			else
				rectangle(blobImage, p1, p2, Scalar(255, 255, 255), 1, 8, 0);
	}

	//destroy all resources (if required)
	//...

	//return the image to show
	return blobImage;
}


/**
 *	Blob extraction from 1-channel image (binary). The extraction is performed based
 *	on the analysis of the connected components. All the input arguments must be 
 *  initialized when using this function.
 *
 * \param fgmask Foreground/Background segmentation mask (1-channel binary image) 
 * \param bloblist List with found blobs
 *
 * \return Operation code (negative if not succesfull operation) 
 */


int extractBlobs(cv::Mat fgmask, std::vector<cvBlob> &bloblist, int connectivity)
{																											
	Mat aux; 																								// auxiliary variables
	cvBlob blob; 			
	Rect blob_rect;
	int myPixel;
	int currentBlob = 0;
	fgmask.convertTo(aux,CV_32SC1);																			

	bloblist.clear();																						// clear blob list
									
	for(int i=0;i<aux.rows;i++){
		for(int j=0;j<aux.cols;j++){
			myPixel = aux.at<int>(i,j);																		// search for a foreground pixel
			if(255==myPixel){
				floodFill(aux, Point(j,i), 1, &blob_rect, 0, 0, connectivity);								// search pixels connected to the current foreground pixel
				blob = initBlob(currentBlob, blob_rect);													// initialize a new blob for connected foreground pixels
				bloblist.push_back(blob);																	// add detected blob to blob list
				currentBlob++;
			}
		}
	}
	return 1;
}


int removeSmallBlobs(std::vector<cvBlob> bloblist_in, std::vector<cvBlob> &bloblist_out, int min_width, int min_height)
{
	bloblist_out.clear();																					// clear blob list
	for(int i = 0; i < bloblist_in.size(); i++)	{
		cvBlob blob_in = bloblist_in[i];																    // get ith blob
		if((blob_in.w>min_width)&(blob_in.h>min_height)){													// filter blobs by size
			bloblist_out.push_back(blob_in); 																// add proper blobs to the new blob list
		}
	}
	return 1;
}



 /**
  *	Blob classification between the available classes in 'Blob.hpp' (see CLASS typedef). All the input arguments must be
  *  initialized when using this function.
  *
  * \param frame Input image
  * \param fgmask Foreground/Background segmentation mask (1-channel binary image)
  * \param bloblist List with found blobs
  *
  * \return Operation code (negative if not succesfull operation)
  */

 // ASPECT RATIO MODELS
#define MEAN_PERSON 0.3950
#define STD_PERSON 0.1887

#define MEAN_CAR 1.4736
#define STD_CAR 0.2329

#define MEAN_OBJECT 1.2111
#define STD_OBJECT 0.4470

// end ASPECT RATIO MODELS
// distances
float ED(float val1, float val2)
{	return sqrt(pow(val1-val2,2));}

float WED(float val1, float val2, float std)
{	return sqrt(pow(val1-val2,2)/pow(std,2));}
//end distances

 int classifyBlobs(std::vector<cvBlob> &bloblist)
 {
	float ratio;																							// auxiliary variables
	float edcheckP;
	float wedcheckP;
	float edcheckC;
	float wedcheckC;
	float edcheckO;
	float wedcheckO;

	CLASS mylabel;
 	//check input conditions and return -1 if any is not satisfied
 	//required variables for classification
 	//classify each blob of the list

 	for(int i = 0; i < bloblist.size(); i++){	
 		cvBlob blob = bloblist[i]; 																			// get i-th blob
		ratio = (float)blob.w/(float)blob.h;																// get aspect ratio of that blob

		edcheckP = ED(ratio,MEAN_PERSON);																	// distance from current blob to person blob model 
		wedcheckP = WED(ratio,MEAN_PERSON,STD_PERSON);														// weighted distance from current blob to person blob model 
		
		edcheckC = ED(ratio,MEAN_CAR);																		// distance from current blob to car blob model 
		wedcheckC = WED(ratio,MEAN_CAR,STD_CAR);															// weighted distance from current blob to car blob model 
		
		edcheckO = ED(ratio,MEAN_OBJECT);																	// distance from current blob to object blob model 
		wedcheckO = WED(ratio,MEAN_OBJECT,STD_OBJECT);														// weighted distance from current blob to object blob model 

		if((wedcheckP<wedcheckC)&&(wedcheckP<wedcheckO)){													// assign person label if weighted distance to person blob model is smaller
			mylabel = PERSON; // type of blob   
		}else if((wedcheckC<wedcheckP)&&(wedcheckC<wedcheckO)){												// assign person label if weighted distance to car blob model is smaller
			mylabel =  CAR;
		}else if((wedcheckO<wedcheckP)&&(wedcheckO<wedcheckC)){												// assign person label if weighted distance to object blob model is smaller
			mylabel =  OBJECT;
		}
		bloblist[i].label = mylabel;
 	}
 	return 1;
 }

//stationary blob extraction function
 /**
  *	Stationary FG detection
  *
  * \param fgmask Foreground/Background segmentation mask (1-channel binary image)
  * \param fgmask_history Foreground history counter image (1-channel integer image)
  * \param sfgmask Foreground/Background segmentation mask (1-channel binary image)
  * \return Operation code (negative if not succesfull operation)
  * Based on: Stationary foreground detection for video-surveillance based on foreground and motion history images, D.Ortego, J.C.SanMiguel, AVSS2013
  */

#define FPS 25 																				//check in video - not really critical
#define SECS_STATIONARY 5																	
#define I_COST 1 																			// increment cost for stationarity detection
#define D_COST 4 																			// decrement cost for stationarity detection
#define STAT_TH 0.4

int extractStationaryFG (Mat fgmask, Mat &fgmask_history, Mat &sfgmask)
{
	Mat aux; 																				//foreground bits
	Mat notAux;																				//background bits
	Mat fgmask_history_norm; 																//normalized foreground history image

	int numframes4static=(int)(FPS*SECS_STATIONARY);

	threshold(fgmask, aux, 127, 1,0);														//Mark foregorund pixels from fgmask
	threshold(fgmask, notAux, 127, 1,1);													//Mark background pixels from fgmask

	addWeighted(fgmask_history, 1, aux, I_COST, 0.0, fgmask_history, CV_32FC1);				//Increment score for foreground pixels in fgmask history
	addWeighted(fgmask_history, 1, notAux, -D_COST, 0.0, fgmask_history, CV_32FC1);			//Decrement score for backgrounf pixels in fgmask history
	threshold(fgmask_history, fgmask_history, 0, 255,3);									//Set to 0 all negative values in fgmask history

	fgmask_history_norm = fgmask_history/(numframes4static);
	threshold(fgmask_history_norm, fgmask_history_norm, 1, 1,2);							//Normalize fgmask history
	threshold(fgmask_history_norm, sfgmask, STAT_TH, 255, 0);								//Create static fgmask from the history fgmask using the STAT_TH threshold
	sfgmask.convertTo(sfgmask, CV_8UC1);
	return 1;
}


