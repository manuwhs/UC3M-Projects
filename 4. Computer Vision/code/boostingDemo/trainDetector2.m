% clear all
close all

% Load parameters
parameters

% Load precomputed features for training and test
load (dataFile)

% Total number of samples (objects and background)
Nsamples = length(data.class);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Split training and test
% First, select images use for training:
rand('seed', 0)

n = unique(data.image);
n = n(randperm(length(n)));
trainingImages = n(1:numTrainImages);
trainingSamples = find(ismember(data.image, trainingImages));

% Plot number of background and object training samples
figure
hist(data.class(trainingSamples), [-1 1]);
title('Number of background and object training samples')
drawnow

featuresTrain = data.features(trainingSamples, :)';
classesTrain  = data.class(trainingSamples)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TRAINING THE DETECTOR:  THIS JUST OBTAINS THE MODEL !!

NweakClassifiers =6; 
Nh = 150;   % Number of hidden layer values
classifier = AdaBoostELM_train(featuresTrain.', classesTrain.', NweakClassifiers, Nh);
% classifier = gentleBoost(featuresTrain, classesTrain, NweakClassifiers);

data.detector2 = classifier;
data.trainingSamples = trainingSamples;
data.trainingImages = trainingImages;

save (dataFile, 'data')
