function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

% Some useful variables
m = size(X, 1);
n = size(X, 2);
niter = 50;
alpha = 0.1;

% Add ones to the X data matrix
X = [ones(m, 1) X];

all_theta = zeros(num_labels, n + 1);
J = zeros(num_labels,niter);

%{
%---   way 1 ----%

for i = 1:num_labels
    a = y==i;
    for j  = 1:niter
        [J(i,j) grad] = lrCostFunction(all_theta(i,:)', X, a, lambda);
        all_theta(i,:) = all_theta(i,:) - alpha*grad'; 
    end
end
%}


%---  way 2 ---%

%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);

for i = 1:num_labels
    a = y==i;
    % Set options for fminunc
    options = optimset('GradObj', 'on', 'MaxIter', niter);    
    [all_theta(i,:)] = ...
        fmincg (@(t) ( lrCostFunction(t, X, a, lambda) ), ...
        all_theta(i,:)', options); 
end

%To get the values of the J organized like in the way 1
% change line 82-85 by this 
%{
    [all_theta(i,:),new] = ...
        fmincg (@(t) ( lrCostFunction(t, X, a, lambda) ), ...
        all_theta(i,:)', options); 
        J(i,:) = new'; 
%This is not necessary: all_theta = all_theta';
% because if a=[1,2;3,4] and a(1,:)=[5;6] then 
% a = [5,6;3,4], in J(i,:) = new' niether but is writen 
% for clarity purposes.

%}



% =========================================================================


end
