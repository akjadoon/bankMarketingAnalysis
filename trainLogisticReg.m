function [theta] = trainLogisticReg(X, y, lambda)
%Trains logistic regression given (X,y) and regularization parameter


% Initialize theta (Assumes X already includes x0)
initial_theta = zeros(size(X, 2), 1);


% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
    fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);


end