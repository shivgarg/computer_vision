function warped_img = ComputeFlowWarpFrame(img0, img1, u0, v0, t)
% Use forward flow warping to compute an interpolated frame.
%
% INPUTS
% img0 - Array of size h x w x 3 containing pixel data for the starting
%        frame.
% img1 - Array of the same size as img0 containing pixel data for the
%        ending frame.
% u0 - Horizontal component of the optical flow from img0 to img1. In
%      particular, an array of size h x w where u0(y, x) is the horizontal
%      component of the optical flow at the pixel img0(y, x, :).
% v0 - Vertical component of the optical flow from img0 to img1. In
%      particular, an array of size h x w where v0(y, x) is the vertical
%      component of the optical flow at the pixel img0(y, x, :).
% t - Parameter in the range [0, 1] giving the point in time at
%     which to compute the interpolated frame. For example, 
%     t = 0 corresponds to img0, t = 1 corresponds to img1, and
%     t = 0.5 is equally spaced in time between img0 and img1.
%
% OUTPUTS
% img - Array of size h x w x 3 containing pixel data for the interpolated
% frame.

    height = size(img0, 1);
    width = size(img1, 2);
    
    % Use forward warping to estimate the velocities at time t.
    [ut, vt] = WarpFlow(img0, img1, u0, v0, t);
    
    warped_img = zeros(size(img0));
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
%                                YOUR CODE HERE:                               %
%            Use backward warping to compute the interpolated frame.           %
%                 Store the result in the warped_img variable.                 %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	for i=1:height
		for j=1:width
			y=round(i-t*vt(i,j));
			x=round(j-t*ut(i,j));
			if x<=0 | x > width | y<=0 | y> height
				continue;
			endif
			warped_img(i,j)=img0(y,x);
		endfor
	endfor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
%                                 END YOUR CODE                                %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    warped_img = uint8(warped_img);
end

function [ut, vt] = WarpFlow(img0, img1, u0, v0, t)
% Use forward warping to transform a flow field.
%
% INPUTS
% img0 - Array of size h x w x 3 containing pixel data for the starting
%        frame.
% img1 - Array of the same size as img0 containing pixel data for the
%        ending frame.
% u0 - Horizontal component of the optical flow from img0 to img1. In
%      particular, an array of size h x w where u0(y, x) is the horizontal
%      component of the optical flow at the pixel img0(y, x, :).
% v0 - Vertical component of the optical flow from img0 to img1. In
%      particular, an array of size h x w where v0(y, x) is the vertical
%      component of the optical flow at the pixel img0(y, x, :).
% t - Parameter in the range [0, 1] giving the point in time at
%     which to compute the interpolated flow field. For example,
%     t = 0 corresponds to img0, t = 1 corresponds to img1, and
%     t = 0.5 is equally spaced in time between img0 and img1.
%
% OUTPUTS
% ut - Horizontal component of the warped flow field; this is an array of
%      the same size as u0.
% vt - Vertical component of the warped flow field; this is an array of the
%      same size as v0.

    height = size(img0, 1);
    width = size(img1, 2);

    ut = zeros(size(u0));
    vt = zeros(size(v0));
	size(u0)
	size(img0)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
%                                YOUR CODE HERE:                               %
%            Use forward warping to compute the velocities ut and vt           %
%               using the procedure described in the problem set.              %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    check = zeros(size(img0))+intmax;
    for i=1:height
        for j=1:width
			y=i+t*v0(i,j);
			x=j+t*u0(i,j);
			x=[max(0,floor(x)) min(width,ceil(x))];
			y=[max(0,floor(y)) min(height,ceil(y))];
			if (x(1) > x(2)) | (y(1) > y(2))
				continue;
			endif
			for l=linspace(x(1),x(2),1)
				for k=linspace(y(1),y(2),1)
					p=round(i+v0(i,j));
					q=round(j+u0(i,j));
					if (p<=0) | (p>height) | (q<=0) | (q>width)
						continue;
					endif
					n=norm(double(reshape(img0(i,j,:)-img1(p,q,:),[1,3])));
					if n < check(k,l)
						ut(k,l)=u0(i,j);
						vt(k,l)=v0(i,j);
						check(k,l) = n;
					endif

				endfor
			endfor
        endfor
    endfor


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                              %
%                                 END YOUR CODE                                %
%                                                                              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    % Use linear interpolation to fill in any unassigned elements of the
    % warped velocities.
    ut = FillInHoles(ut);
    vt = FillInHoles(vt);
end