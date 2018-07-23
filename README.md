# Wide-angle Image Correction

This package is an attempt to process images taken with a wide-angle 
lens and deduce the approximate angle of distortion using a genetic 
algorithm. Once the angle of distortion is deduced then the image can 
be corrected using a conformal mapping.

At a high level the approach works as follows:

1. From initial image, perform a Hough Transform to detect all 
straight lines
2. Create an initial set of line pairs by randomly shuffling all 
detected lines and pairing neighbors together
3. Compute the slope of each line, and then compute the angle 
between slopes of each pair (loss function)
4. Create parent sets of line pairs determined by the pairs with 
the smallest angles
5. From the set of parents, create children pairs with random 
mutations of the start/stop points
6. Compute the new loss function
7. Recurse on steps 4 - 6
    
The interpretation of the entire algorithm is that once recursion has 
terminated then all of the pairs of lines are either parallel or 
quasi-parallel.  

If the original image had a wide-angle distortion 
then the final pairs of lines will not be exactly parallel, and will 
instead have a slight angular separation between them.  This average 
angular separation is interpreted as the angle of distortion from the 
photo lens.  Once this angle is determined then the original image 
can have the distortion removed by some conformal mapping (TBD).

**Note**: This is intended as an experiment and learning experience 
with genetic algorithms and is not intended as a definite solution to 
determining arbitrary wide-angle lens distortions.  If you would like 
to contribute to this project or offer insight please open a ticket or 
contact me directly.  **Any contributions are greatly appreciated**.


## Setup

Checkout the package and install dependencies by running
```bash
git clone https://github.com/brainsqueeze/Image_correction.git
cd ${HOME}/Image_correction
pip install -r requirements.txt
```

## Run algorithm on demo image

The learning algorithm can be run on the provided sample image by 
running
```bash
python -m src.find_parallel
```


## To do

1. The final conformal mapping on the original image is not yet 
implemented.  
2. There are improvements needed for pruning poorly 
performing species during each generational epoch.