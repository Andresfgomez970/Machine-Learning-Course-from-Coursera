function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
% We know that J = 1/(2m) sum (i=1,i=m) [h(x_i)-y_i]**2 
% That is  1/(2m) sum (i=1,i=m) [theta.*x_i - y_i]**2
% but theta'*x_i is the i-th component of X*theta' and 
% the same happens for y_i, so organizing the values of the form 
% theta'*x_i - y_i in an array we have that this  X*theta' - y
% and so the cost funtion is:  1/(2m) sum([X*theta - y]**2)

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

J = 1/(2*m) * sum( (X*theta-y).**2 );


% =========================================================================

end
