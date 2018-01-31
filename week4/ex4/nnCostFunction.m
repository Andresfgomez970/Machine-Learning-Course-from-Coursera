function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

Theta1 = reshape( nn_params(1:hidden_layer_size * (input_layer_size + 1) ), ...
                 hidden_layer_size, (input_layer_size + 1) );


Theta2 = reshape( nn_params( 1 + (hidden_layer_size * (input_layer_size + 1))  :end), ...
                 num_labels, (hidden_layer_size + 1) );

% Setup some useful variables
m = size(X, 1);

X = [ones(m,1) X];

a2 = sigmoid( Theta1*X' );
a2 = [ones(1,m) ; a2];
a3 =  sigmoid( Theta2*a2 );

ys = zeros(num_labels,m);
for i = 1:m 
    ys( y(i), i ) = 1;
end 

J = -1/m*( ys.*log(a3)  + (1-ys).*log( 1-a3 ) );
J = sum(sum(J));


Delta1 = zeros(hidden_layer_size,input_layer_size+1);
Delta2 = zeros(num_labels,hidden_layer_size+1);

%{
This code could be useful if the X array is very big for the memory. 
Proves have not been done to know what number could be consider 
large. 
%}

%{
for i = 1:m 
    a1 = X(i,:)';
    z2 = Theta1*a1;
    a2 =  [1;sigmoid(z2)] ;
    a3 = sigmoid( Theta2*a2 );
    delta3 = a3 - ys(:,i);
    delta2 = (Theta2'*delta3).*[1;sigmoidGradient(z2)];  
    Delta1 = Delta1 + delta2(2:end)*a1';
    Delta2 = Delta2 + delta3*a2';
end 
%}

for i = 1:m 
    delta3 = a3(:,i) - ys(:,i);
    delta2 = ( Theta2'*delta3 ).*(a2(:,i).*(1-a2(:,i) )) ;
    Delta2 = Delta2 + delta3*a2(:,i)';
    Delta1 = Delta1 + delta2(2:end)*X(i,:);
end 

CTheta1 = Theta1;
CTheta2 = Theta2;
CTheta1(:,1) = 0;
CTheta2(:,1) = 0;
Delta1 = 1/m* (Delta1 + lambda*CTheta1 );
Delta2 = 1/m* (Delta2+ lambda*CTheta2);

grad = [Delta1(:) ; Delta2(:)];
J = J +  1./(2*m)*lambda*( sum(sum(CTheta1.*CTheta1)) ...
        +sum(sum(CTheta2.*CTheta2)) );

% -------------------------------------------------------------

% =========================================================================

end
