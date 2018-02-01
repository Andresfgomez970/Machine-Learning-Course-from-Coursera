function [h, display_array] = displayData(X, example_width)
%DISPLAYDATA Display 2D data in a nice grid
%   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
%   stored in X in a nice grid. It returns the figure handle h and the 
%   displayed array if requested.

% Set example_width automatically if not passed in
%   So, ~ is the not operator an the option var maked for the 
%   funtion only to change if it that name exist as a variable
%   || is the or operator
%   isempty return True if the input is a 0 matrix

if ~exist('example_width', 'var') || isempty(example_width)
	%This suppose that the image is an square
	%  take care if your image is not because you would
	%  get then an error.
	example_width = round(sqrt(size(X, 2)));
end

% Gray Image (To set this in the plotting)
colormap(gray);

% Compute rows, cols
[m n] = size(X);
%As n is a number that could represent the area of the image
% in some arbitary units, the height is obtained as follow in 
% that units too  
example_height = (n / example_width); 

% Compute number of items to display
% In general this is to show mxm element
% or mx(m+1) because ceil(m / display_rows)
% is like to have ceil(m / floor( sqrt(m)) )
% and so the numerator is bigger or equal to the 
% denominator and the result is therefore 
% sqrt(m)+1 or sqrt(m).
display_rows = floor(sqrt(m));
display_cols = ceil(m / display_rows);

% Between images padding
pad = 1;

% Setup blank display
display_array = - ones(pad + display_rows * (example_height + pad), ...
                       pad + display_cols * (example_width + pad));



% Copy each example into a patch on the display array
curr_ex = 1;
for j = 1:display_rows
	for i = 1:display_cols
		if curr_ex > m, 
			break; 
		end
		% Copy the patch
		
		% Get the max value of the patch to divide by this one later 
		%  making this one to represent the most dark one in a given 
		%  image; in some way this set the values of each image
		%  in a same scale; here we are telling to the program where to 
		%  put our example images in each row of X.
		max_val = max(abs(X(curr_ex, :))); 
		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		             pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
			reshape(X(curr_ex, :), example_height, example_width) / max_val;
		curr_ex = curr_ex + 1;
	end
	if curr_ex > m, 
		break; 
	end
end

% Display Image
% imagesc is the code to plor a matrix as an image
h = imagesc(display_array, [-1 1]); 

% Do not show axis
axis image off

drawnow;

end


