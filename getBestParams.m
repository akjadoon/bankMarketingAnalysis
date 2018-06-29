% function [theta, lambda] = getBestParams(XTrain, yTrain, XCV, yCV, initial_theta)

best_lambda = 0;
best_score = 0;
best_theta = [];
lambda = 0.01;

for i=1:8
    % Set Options
    options = optimset('GradObj', 'on', 'MaxIter', 400);

    % Optimize
    [theta, J, exit_flag] = ...
        fminunc(@(t)(costFunctionReg(t, XTrain, yTrain, lambda*(3^i))), initial_theta, options);

    p = predict(theta, XCV);
    fprintf('Train Accuracy: for lambda %d is %f\n',i ,mean(double(p == yCV)) * 100);
    if (mean(double(p == yCV)) > best_score)
        best_lambda = lambda *(3^i);
        best_theta = theta;
    endif

end
theta = best_theta;
lambda  = best_lambda;