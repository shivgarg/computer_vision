%reads in the image, converts it to grayscale, and converts the intensities
%from uint8 integers to doubles. (Brightness must be in 'double' format for
%computations, or else MATLAB will do integer math, which we don't want.)
dark = double(rgb2gray(imread('u2dark.png')));

%%%%%% Your part (a) code here: calculate statistics
min_val=min(min(dark))
max_val=max(max(dark))
average_val=mean(mean(dark))

%%%%%% Your part (b) code here: apply offset and scaling
fixedimg = [];
a=255.0/(max_val-min_val)
b=-a*min_val
fixedimg=a*dark+b

%displays the image
imshow(uint8(fixedimg));
imshow(uint8((dark-min_val)*255.0/(max_val-min_val)))

%%%%%% Your part (c) code here: apply the formula to increase contrast,
% and display the image
contrasted = uint8(2*(fixedimg-128)+128)

