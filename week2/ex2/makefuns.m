function funs = makefuns
  funs.plotData=@plotData;
  funs.costFunction=@costFunction;
  funs.sigmoid=@sigmoid;
  funs.predict=@predict;
  funs.mapFeature=@mapFeature;
  funs.costFunctionReg=@costFunctionReg;
  funs.plotDecisionBoundary= @plotDecisionBoundary;
end

% ============================================
% function plotData(X, y)
% ====================== ======================

function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% ====================== CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% Create New Figure
figure; hold on;

pos = y==1;
neg = y==0;

plot(X(pos,1),X(pos,2),'+r')
plot(X(neg,1),X(neg,2),'markerfacecolor','k','o')

% =========================================================================

hold off;

end


% ============================================
% function sigmoid(X, y)
% ====================== ======================

function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 

% ====================== CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% =============================================================

g = 1./(1+exp(-z));

end


% ============================================
% function costFunction(X, y)
% ====================== ======================

function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.


% ====================== CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
h = sigmoid(X*theta);

J = - 1/m*( y'*log(h) + (1-y)'*log(1-h) );
grad = 1/m*X'*(h-y);


% =============================================================

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check for this one to include it here 
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ============================================
% function plotDecisionBoundary(theta, X, y);
% ====================== ======================

function plotDecisionBoundary(theta, X, y,degree)
%PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
%the decision boundary defined by theta
%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
%   positive examples and o for the negative examples. X is assumed to be 
%   a either 
%   1) Mx3 matrix, where the first column is an all-ones column for the 
%      intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones

% Plot Data
plotData(X(:,2:3), y);
hold on

if size(X, 2) <= 3
    % Only need 2 points to define a line, so choose two endpoints
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % Calculate the decision boundary line
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y)
    
    % Legend, specific for the exercise
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    axis([30, 100, 30, 100])
else
    % Here is the grid range
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j),degree)*theta;
        end
    end
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end
hold off

end



% ============================================
%function p = predict(theta, X)
% ====================== ======================

function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

% ====================== CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%


% You need to return the following variables correctly
p = sigmoid(X*theta); %zeros(m, 1);
con1 = p>=0.5 ;
con2 = p<0.5;
p(con1) = 1; 
p(con2) = 0;

% =========================================================================

end



% ============================================
%function out = mapFeature(X1, X2)
% ====================== ======================

function out = mapFeature(X1, X2,degree)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%

degree = degree;
out = ones( size( X1(:,1) ) );
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

end

% ============================================
%function [J, grad] = costFunctionReg(theta, X, y, lambda)
% ====================== ======================



function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
h = sigmoid(X*theta);


J = - 1/m*( y'*log(h) + (1-y)'*log(1-h) ) + lambda*1/(2*m)*theta(2:n)'*theta(2:n);
grad(1) = 1/m*X'(1,:)*(h - y ) ;
grad(2:n) = 1/m*X'(2:n,:)*(h - y ) + lambda/m*theta(2:n);


% =============================================================

end


