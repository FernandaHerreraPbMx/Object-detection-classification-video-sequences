# Object-detection-classification-video-sequences

Blob extraction was implemented using the Grass-Fire algorithm and a simple size based filter. The implemented blob classification routine is based on the aspect ratio of the extracted blobs and a statistical Gaussian pre-trained classifier based on mean and variance. Lastly, the static foreground extractor is based on a foreground motion history image that is able to detect stationary blobs. 
