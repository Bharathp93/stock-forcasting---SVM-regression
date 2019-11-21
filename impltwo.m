X1 = csvread('HistoricalQuotes.csv');
X=X1(:,end);
A = zeros(size(X,1),5);
B = zeros(size(X,1),1);
p_new = zeros(size(X,1),1);
%Xtrain=X(96:195,end);
%Xtest=X(196:235,end);

EMA15 = tsmovavg(X,'e',15,1);
p_dash = tsmovavg(X,'e',3,1);
for j = 1:(size(X,1) - 5)
    p_new(j) = X(j+5) - X(j);
end
p_five_dash =  tsmovavg(p_new,'e',3,1);
%p_new =  tsmovavg(X,'e',3,1);
for i = 21:(size(X,1)- 5)
    A(i,1) = X(i) - EMA15(i);
    A(i,2) = (X(i) - X(i-5))/X(i-5)*100;
    A(i,3) = (X(i) - X(i-10))/X(i-10)*100;
    A(i,4) = (X(i) - X(i-15))/X(i-15)*100;
    A(i,5) = (X(i) - X(i-20))/X(i-20)*100;
    B(i) = p_five_dash(i)/p_dash(i) * 100;
end


Xtrain=A(91:190,1:end);
Ytrain=X(91:190);
Xtest=X1(196:235,1:end-1);
%Ytest=X1(196:235,end);
Ytest=X(196:235);


C = [(2 * exp(-3)) (2 * exp(0)) (2 * exp(3)) (2 * exp(6)) (2 * exp(9))];
gamma = [(2 * exp(-9)) (2 * exp(-6)) (2 * exp(-3)) (2 * exp(0)) (2 * exp(3))];
all_combo = combvec(C,gamma);
all_combo = all_combo';
X_train=X1(91:190,1:end-1);
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

fun=fitrsvm(X_train, Ytrain,'KernelFunction','RBF', 'KernelScale', best_gamma ,'BoxConstraint', best_c);
Y_test_pred = predict(fun, Xtest);

plot(Y_test_pred,'r')
hold on;
plot(Ytest,'b')
legend('predicted','actual','Location','southeast');
ylabel('closing price');
xlabel('days');
