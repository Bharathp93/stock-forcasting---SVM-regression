X = csvread('HistoricalQuotes.csv');

Xtrain=X(91:190,1:end-1);
Ytrain=X(91:190,end);
Xtest=X(196:235,1:end-1);
Ytest=X(196:235,end);


% rbf_in=fitrsvm(Xtrain,Ytrain,'KernelFunction','RBF', 'KernelScale', 2 * exp(3) ,'BoxConstraint', 2 * exp(15));
% %CVSVMModel = crossval(rbf_in)
% rbf_out=predict(rbf_in,Xtest);
% %rbf_out=kfoldPredict(CVSVMModel);
% mse = mean((Ytest - rbf_out).^2);
% avg_rmse = sqrt(mse);
% %avg_rmse = mean(sqrt(mse));

C = [(2 * exp(-3)) (2 * exp(0)) (2 * exp(3)) (2 * exp(6)) (2 * exp(9))];
gamma = [(2 * exp(-9)) (2 * exp(-6)) (2 * exp(-3)) (2 * exp(0)) (2 * exp(3))];
all_combo = combvec(C,gamma);
all_combo = all_combo';

avrg_rmse = zeros(size(all_combo, 1), 1);

K = 10;
cv = cvpartition(numel(Ytrain), 'kfold',K);
for i = 1:size(all_combo, 1)
    mse = zeros(K,1);
    for k=1:K
        % training/testing indices for this fold
        trainIdx = cv.training(k);
        testIdx = cv.test(k);

        % box constraint = c; kernel scale = gamma
        rbf_in=fitrsvm(Xtrain(trainIdx,:), Ytrain(trainIdx),'KernelFunction','RBF', 'KernelScale', all_combo(i,2) ,'BoxConstraint', all_combo(i,1));

        % predict regression output
        Y_hat = predict(rbf_in, Xtrain(testIdx,:));

        % compute mean squared error
        mse(k) = mean((Ytrain(testIdx) - Y_hat).^2);
    end

    % average RMSE across k-folds
    avrg_rmse(i) = mean(sqrt(mse));
end

[M, I] = min(avrg_rmse);
best_c = all_combo(I,1);
best_gamma = all_combo(I,2);

fun=fitrsvm(Xtrain, Ytrain,'KernelFunction','RBF', 'KernelScale', best_gamma ,'BoxConstraint', best_c);
Y_test_pred = predict(fun, Xtest);

plot(Y_test_pred,'r')
hold on;
plot(Ytest,'b')
legend('predicted','actual','Location','southeast');
ylabel('closing price');
xlabel('days');

% x = [0.0818947604570001;0.0637366218143446;0.0637385763090755;0.0637385421433271;0.357967662462945;0.0818815759275310;0.0637332367430879;0.0637364248936212;0.0637364557419345;0.436522571477193;0.0730948977035996;0.0556255540072058;0.0556306059990861;0.0556305683268427;0.874402977341581;0.0557534782990184;0.0121654238359171;0.0107739545782558;0.0107593069532615;0.0107512925084818;0.393893181552321;0.331817275797272;0.00357377505978363;0.00338863065013305;0.00338852517684341]
% [M, I] = min(x);
