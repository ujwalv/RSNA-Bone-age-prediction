# KAGGLE RSNA-Bone-age-prediction
Predict bone age of a person using the provided XRay image dataser

https://www.kaggle.com/kmader/rsna-bone-age

This is a simple example to run ML model on XRAY image dataset, 

<h3>Few point to note:<h3>

Actual DATASET is loaded directly from KAGGLE


Data is mainly reused in most of the places to save memory


Image is transformed into 320x240 , to have uniformity and also to reduce each execution from 1:30 hr to 20minutes 


This example is the very basic model to process image data, with mean error of 35%
<br>
<br>
<br>
Context:
    At RSNA 2017 there was a contest to correctly identify the age of a child from an X-ray of their hand. This is the dataset on Kaggle      making it easier to experiment with and do educational demos. Additionally maybe there are some new ideas for building smarter models for handling X-ray images.

Content:
A number of folders full of images (digital and scanned) with a CSV containing the age (what is to be predicted) and the gender (useful additional information)

Acknowledgements:
The dataset was originally published on CloudApp as an RSNA challenge.

Original Dataset Acknowledgements:
The Radiological Society of North America (RSNA) Radiology Informatics Committee (RIC) Pediatric Bone Age Machine Learning Challenge Organizing Committee:

Kathy Andriole, Massachusetts General Hospital
Brad Erickson, Mayo Clinic
Adam Flanders, Thomas Jefferson University
Safwan Halabi, Stanford University
Jayashree Kalpathy-Cramer, Massachusetts General Hospital
Marc Kohli, University of California - San Francisco
Luciano Prevedello, The Ohio State University
Data sets used in the Pediatric Bone Age Challenge have been contributed by Stanford University, the University of Colorado and the University of California - Los Angeles.

The MedICI platform (built CodaLab) used for the challenge is provided by Jayashree Kalpathy-Cramer, supported through NIH grants (U24CA180927) and a contract from Leidos.

Inspiration
Can you predict with better than 4.2 months accuracy?
Is identifying the joints an important step?
What algorithms work best?
What do the algorithms focus on?
Is gender a necessary piece of information or can it be automatically derived from the image?
