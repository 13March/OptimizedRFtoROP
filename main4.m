clear; close all; clc;

% 数据准备（替换为实际的训练和测试数据）
% XTrain, YTrain - 训练数据和目标
% XTest, YTest - 测试数据和目标
filename = 'NewData.xlsx';               
data     = readmatrix(filename); % [DVER, WOBA, RPMA, MFIA, Size, SPPA, MDOA, I, ROPA];
data(:,end) = data(:,1) ./ data(:,end);

% 不同地层拟合时，准备内容
ForBoundary  = [0,520; 520,1143; 1143,1256; 1256,1893; 1893,3128; 3128,3629; 3629,3826]; % 不同地层的分界面(最后为储层)
Cleaned_DVER = data(:,1); % 提取数据清洗后的垂深
[a,~]        = size(ForBoundary);
Cleaned_Data = [];
for i = 2 : a
    ForRows       = any(Cleaned_DVER > ForBoundary(i,1) & Cleaned_DVER <= ForBoundary(i,2), 2); 
    FormationData = data(ForRows,:); 
    Dep_data      = DataClean_DBSCAN(FormationData);   % 处理异常数据
    Cleaned_Data  = [Cleaned_Data;Dep_data];
end

Cleaned_Data  = sortrows(Cleaned_Data, 1);            % 按照垂深从小到大排序

% 提取每层数据
FormationData = Cleaned_Data;

% 提取垂深和实际钻速
Deep        = FormationData(:,1);
ROP_actual  = FormationData(:,end);
FormationData(:, end) = [];    % 删除实际钻速的数据 

% 将数据拆分为训练集和测试集（80%训练，20%测试）
cv      = cvpartition(size(FormationData, 1), 'HoldOut', 0.2);
idx     = cv.test;
XTrain = FormationData(~idx, :);
YTrain = ROP_actual(~idx);
XTest  = FormationData(idx, :);
YTest  = ROP_actual(idx);

% 定义随机森林超参数的上下界
lb = [10, 1, 1, 1];   % 树的数量下界, 最大深度下界, 最小分割样本数下界, 最小叶节点大小下界
ub = [500, 30, 10, 20]; % 树的数量上界, 最大深度上界, 最小分割样本数上界, 最小叶节点大小上界

% 定义适应度函数的句柄，适应度函数修改为包括4个参数
fitnessFunc = @(params) fitnessFunction(params, XTrain, YTrain);

% 初始化全局变量
global bestValues meanValues
bestValues = []; % 用于存储每次迭代的最佳值
meanValues = []; % 用于存储每次迭代的平均值

% 设置粒子群优化算法（PSO）选项
options = optimoptions('particleswarm', ...
    'MaxIterations', 50, ...        % 最大迭代次数
    'SwarmSize', 10, ...            % 粒子群规模
    'Display', 'iter', ...          % 显示优化过程
    'UseParallel', true, ...        % 启用并行计算
    'OutputFcn', @psoOutputFcn);    % 指定自定义输出函数

% 执行粒子群优化算法
[bestParams, bestFitness] = particleswarm(fitnessFunc, 4, lb, ub, options);

% 从最佳参数中提取超参数
nTreesBest = round(bestParams(1));
maxDepthBest = round(bestParams(2));
minSamplesSplitBest = round(bestParams(3));
minLeafSizeBest = round(bestParams(4));

% 用最佳超参数训练最终模型
finalModel = TreeBagger(nTreesBest, XTrain, YTrain, ...
                        'Method', 'regression', ...
                        'MinLeafSize', minLeafSizeBest, ...
                        'MaxNumSplits', maxDepthBest, ...
                        'MinLeafSize', minLeafSizeBest, ...
                        'OOBPrediction', 'On');

% 对测试数据进行预测
YPred = predict(finalModel, XTest);

% 计算均方误差（MSE）
MSE = mean((YPred - YTest).^2);

% 显示最佳参数和MSE
disp(['最佳树的数量: ', num2str(nTreesBest)]);
disp(['最佳最大深度: ', num2str(maxDepthBest)]);
disp(['最佳最小分割样本数: ', num2str(minSamplesSplitBest)]);
disp(['最佳最小叶节点大小: ', num2str(minLeafSizeBest)]);
disp(['测试集上的MSE: ', num2str(MSE)]);

% 保存最优模型到文件
save('BestRandomForestModel.mat', 'finalModel');

% 绘制每次迭代的最佳值和平均值
figure;
plot(bestValues, 'b-', 'LineWidth', 2); % 最佳值
hold on;
plot(meanValues, 'r--', 'LineWidth', 2); % 平均值
xlabel('迭代次数');
ylabel('适应度值');
legend('最佳值（Best）', '平均值（Mean）');
title('PSO优化过程中的最佳值和平均值');
grid on;

% 自定义输出函数，用于记录每次迭代的最佳值和平均值
function stop = psoOutputFcn(options, state, flag)
    % 声明全局变量来存储每次迭代的最佳值和平均值
    global bestValues meanValues

    % 在每次迭代时，记录最佳值和平均值
    bestValues(end+1) = options.bestfval;       % 记录每次迭代的最优值
    meanValues(end+1) = options.meanfval;         % 记录每次迭代的平均值

    % 设置 stop 为 false，以便 PSO 不会在每次迭代后停止
    stop = false;
end

% 定义适应度函数
function fitness = fitnessFunction(params, X, Y)
    % 提取超参数
    nTrees = round(params(1));          % 树的数量
    maxDepth = round(params(2));        % 最大深度
    minSamplesSplit = round(params(3)); % 最小分割样本数
    minLeafSize = round(params(4));     % 最小叶节点大小
    
    % 构建随机森林模型
    RFModel = TreeBagger(nTrees, X, Y, ...
                         'Method', 'regression', ...
                         'MinLeafSize', minLeafSize, ...
                         'MaxNumSplits', maxDepth, ...
                         'MinLeafSize', minLeafSize, ...
                         'OOBPrediction', 'On', ...
                         'OOBPredictorImportance', 'On');
    
    % 使用OOB误差作为适应度值
    oobErrorValue = oobError(RFModel, 'Mode', 'cumulative');
    fitness = mean(oobErrorValue);
end

% DBSCAN 数据清洗函数
function X_cleaned = DataClean_DBSCAN(X)
% 设置DBSCAN参数
epsilon = 15; % 邻域半径
MinPts = 5; % 最小点数

% 使用DBSCAN进行聚类
[labels, isNoise] = dbscan(X, epsilon, MinPts);

% 检测噪声点（异常点）
outliers = (labels == -1); % 在DBSCAN中，标签为-1的点被标记为噪声点

% 删除异常数据点
X_cleaned = X(~outliers, :);

end
