%% LeastSquare
clc, close all, clear all
dataset = 'wallpaper';
[train_featureVector, train_labels, test_featureVector, test_labels] = loadDataset(dataset);
N = size(train_featureVector,1);          % Number of data points
c_k = categories(train_labels);           % Class Labels
x_tilda = [train_featureVector,ones(N,1)];% Add bias to train data = x_tilda 
t = ones(N,numel(c_k));                   % 1 of K coding scheme
for j=1:length(c_k)
    for i=1:N
        if j == double(train_labels(i,1))
            t(i,j)=t(i,j);
        else
            t(i,j)=0;
        end
    end
end
W = inv(x_tilda'*x_tilda)*x_tilda'; 
W = W*t;
% W = x_tilda \ t;             % Alternately Least square fit for system of linear equations y = w^t*x
%  import bioma.data.*
% DMobj = DataMatrix(W);
   

trainlabel_prob = x_tilda*W;
[~,train_pred] = max(trainlabel_prob,[],2);
train_pred = categorical(train_pred);
train_confmat = confusionmat(train_labels,train_pred);
% train_classmat = train_confmat./meshgrid(countcats(train_labels))';
% train_acc = mean(diag(train_classmat));
% train_std = std(diag(train_classmat));
% figure(1)
 % plotconfusion(train_labels,train_pred)
%test
xtest = [test_featureVector,ones((size(test_featureVector,1)),1)];
testlabel_prob = xtest*W;
[~,test_pred] = max(testlabel_prob,[],2);
test_pred = categorical(test_pred);
test_confmat = confusionmat(test_labels,test_pred);
% test_classmat = train_confmat./meshgrid(countcats(test_labels))';
% test_acc = mean(diag(test_classmat));
% test_std = std(diag(test_classmat));
% figure(2)
% plotconfusion(test_labels,test_pred)
% visualize
% figure(3)
 [x1,x2] = ndgrid( ...
 linspace(min(train_featureVector(:,2)),max(train_featureVector(:,2)),130),...
 linspace(min(train_featureVector(:,1)),max(train_featureVector(:,1)),130)...
 );
 x3 = reshape((double(test_labels)),size(x1));
 contour(x1,x2,reshape((double(test_labels)'),size(x1)),'color','k');
 hold on;
 scatter(test_featureVector(:,1), test_featureVector(:,2), [], c, 'filled');
% visualizeBoundaries(W,test_featureVector,test_labels,1,2)
% y = -W(3)/W(2)-(W(1)/W(2))*xtest;
% plot(xtest(:, 1), xtest(:, 2), 'bo'), hold on
% plot(xtest, y, 'r');
% axis equal
%% Fisher
clc, clear,close all, 
dataset = 'wallpaper';
[train_featureVector, train_labels, test_featureVector, test_labels] = loadDataset(dataset);
[N,D] = size(train_featureVector);          %N = number of training data, D = dimension
[c ,ia,ic] = unique(train_labels) ;
c_k = double(unique(train_labels));       % Class Labels

% D-dimensional mean vector
Mu = []; V = []; 
id = [ia;N+1];     
for i = 1:length(ia)
    for j = 1:D
        Mu(i,j) = mean(train_featureVector(id(i):id(i+1)-1,j));
        V(i,j) = var(train_featureVector(id(i):id(i+1)-1,j));
    end
end
mu_tot = mean(train_featureVector);             % overall feature mean
v_tot = var(train_featureVector);

% Within class matrix: SW & Between class matrix: SB
    SB=zeros(D,D);              
    SW=zeros(D,D);    
    for k=1:length(double(c))
        classk = find(ic==double(c(k)));        % separate feature data for each class
        xk = train_featureVector(classk,:);        
        muk = mean(xk);                         % Mean of seperated class
        xk = xk - repmat(muk,length(classk),1); % xk - muk
        SW = SW + xk'*xk;                         
        SB = SB + length(classk)*(muk - mu_tot)'*(muk - mu_tot); 
    end
A = SW \ SB;                                % SW^(-1)*SB                 

% Eigen values + Eigen vectors
[W,lamda] = eig(A);                        % W = eigenvectors, lamda = eigenvalues
[eig_vals,ind] = sort(diag(lamda),'descend'); % d = eigenvalues in descending order; ind = indices of d
lamda_s = lamda(ind,ind);
W_s = W(:,ind);                            % reordered eigenvectors
e1 = norm(A*W-W*lamda);e2 = norm(A*W_s-W_s*lamda_s);
e = abs(e1-e2);                            % must be close to 0.
% disp('eigenvalues in decreasing order:')
% disp(eig_vals)
eig_vals_final = eig_vals(eig_vals>=0.001);
eig_vecs_final = W_s(:,1:length(eig_vals_final));
x_train = train_featureVector*eig_vecs_final;

%  [~,train_pred] = max(abs(x_train),[],2);
%  train_pred = categorical(train_pred,c_k);
% train_confmat = confusionmat(train_labels,train_pred);
% train_classmat = train_confmat./meshgrid(countcats(train_labels))';
% train_acc = mean(diag(train_classmat));
% train_std = std(diag(train_classmat));
% figure(1)
% plotconfusion(train_labels,train_pred)
% 
% x_test = test_featureVector*eig_vecs_final;
% [~,test_pred] = max(abs(x_test),[],2);
% test_pred = categorical(test_pred,c_k);
% test_confmat = confusionmat(test_labels,test_pred);
% test_classmat = train_confmat./meshgrid(countcats(test_labels))';
% test_acc = mean(diag(test_classmat));
% test_std = std(diag(test_classmat));
% figure(2)
% plotconfusion(test_labels,test_pred)

% Classification
% for s = 1:length(N)
%         ck(s) = find(ic(s)==ia(s));        % separate feature data for each class
%         x_t(s) = x_train(ck,:);        
% end
% for r = 1:length(ia)
%     P(r) = ((id(r)-id(r+1))/N);
% end
w0 = (mu_tot.^2/(2*v_tot.^2));
y1 = x_train(:,1)*(mu_tot/v_tot)-w0; 
y2 = x_train(:,2)*(mu_tot/v_tot)-w0;
y = [y1 y2];

scatter(x_train(:,1),x_train(:,2))
%% generate toy dataset
clc, clear
% how many points and classes
n = 300;
k = 3;

% randomly choose class labels (integers from 1 to k)
c = randi(k, n, 1);

% convert labels to binary indicator vectors
% Y(i,j) = 1 if point i in class j, else 0
Y = full(sparse((1:n)', c, 1));

% mean of input points in each class
mu = [
    0, 0;
    4, 0;
    0, 4
];

% sample 2d input points from gaussian distributions
% w/ class-specific means
X = randn(n, 2) + mu(c, :);

% add a column of ones
X = [X, ones(n,1)];

% fit weights using least squares
W = X \ Y;

% out-of-sample prediction

% generate new test points on a grid covering the training points
[xtest2, xtest1] = ndgrid( ...
    linspace(min(X(:,2)), max(X(:,2)), 501), ...
    linspace(min(X(:,1)), max(X(:,1)), 501) ...
);
X_test = [xtest1(:), xtest2(:)];

% add a column of ones
X_test = [X_test, ones(size(X_test,1), 1)];

% project test points onto weights
A_test = X_test * W;

% predict class for each test point
% choose class w/ maximal projection
[~, c_test] = max(A_test, [], 2);
xtest3 = reshape(c_test, size(xtest1));
% plot

% plot decision boundary
% using contour plot of predicted class labels at grid points
figure;
contour(xtest1, xtest2, xtest3, 'color', 'k');

% plot training data colored by true class label
hold on;
scatter(X(:,1), X(:,2), [], c, 'filled');