function D = SpatialPyramidDist( I1, I2, nbins )
%SpatialPyramidDist
%   Compute the spatial pyramid measure for the two given image
%   patches.
%
%Input:
%   I1: image patch 1
%   I2: image patch 2
%   nbins: number of bins for color histograms. Note this is unrelated to 
%   spatial partitioning of the image.
%
%Output:
%   D: spatial pyramid measure (a real number)
%
if nargin == 2
    nbins = 20;
end

numLevel = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
%                                YOUR CODE HERE                                %
%                You should fill in D with the weighted distance               %
%                   between histograms of two entire images.                   %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D = HistIntersectDist(I1,I2)/(2**numLevel);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
%                               END OF YOUR CODE                               %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for l = 1 : numLevel,
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
%                                YOUR CODE HERE                                %
%             You should fill in numCells with the number of cells             %
%                           along x and y directions.                          %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    numCells = 2**l;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
%                               END OF YOUR CODE                               %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1 : numCells,
        for j = 1 : numCells,
            
%               You should fill in x_lo1, x_hi1, y_lo1, y_hi1 to               %
%                    extract one cell of I1 in the pyramid.                    %
            x_lo1 = floor((i-1)*size(I1,2)/numCells+1);
            x_hi1 = floor(min(i*size(I1,2)/numCells,size(I1,2)));
            y_lo1 = floor((j-1)*size(I1,1)/numCells+1);
            y_hi1 = floor(min(j*size(I1,1)/numCells,size(I1,1)));
            x_lo2 = floor((i-1)*size(I2,2)/numCells+1);
            x_hi2 = floor(min(i*size(I2,2)/numCells,size(I2,2)));
            y_lo2 = floor((j-1)*size(I2,1)/numCells+1);
            y_hi2 = floor(min(j*size(I2,1)/numCells,size(I2,1)));
            img1 = I1(y_lo1:y_hi1,x_lo1:x_hi1);
            img2 = I2(y_lo2:y_hi2,x_lo2:x_hi2);
            D = D + HistIntersectDist(img1,img2,nbins)/(2**(numLevel+l+1));
%               You should fill in x_lo2, x_hi2, y_lo2, y_hi2 to               %
%                    extract one cell of I2 in the pyramid.                    %
%You should increment D by the weighted distances between patches in this cell.%
        end
    end
end
end

