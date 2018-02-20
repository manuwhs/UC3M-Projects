function [Score, boundingBox, boxScores] = singleScaleBoostedDetectorPrev(img, data, NweakClassifiers)
%
% This runs the detector at single scale.

% Number of weak detectors:
if nargin < 3
    NweakClassifiers = length(data.detector2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Precomputed Test 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rows = size(img,1);
cols = size(img,2);

% Compute all features for test
FeaturesTest = zeros(rows*cols, size(data.dictionary.filter,2));   %Matrix with the features of the test image ONLY ONE IMAGE 256x256 points (vectors) by 640 features!!

disp('Obtaining image features')
for k=1:size(data.dictionary.filter,2)  % For every possible feature
    feature = convCrossConv(img, data.dictionary.filter(k), data.dictionary.patch(k), data.dictionary.location(k));
    % Feature is a 256x256 array of points (image) with the output of
    % performing the path to every possible point in the image !! We
    % trandformit into a column array since it is a feature for every
    % point.
    feature = feature(:);
%     disp(size(feature))
%     disp(size(FeaturesTest(:,k)))
    FeaturesTest(:,k) = feature;
end

% Once we have the features we just run the boosting. 

[Y] = AdaBoostELM_run(FeaturesTest,data.detector2);

Score = vec2mat(Y,rows).'; % Turn vector back into matrix of points

% Look at local maximum of output score and output a set of detected object
% bounding boxes.
s = double(Score>0);
s = conv2(hamming(35),hamming(35),s,'same');

BW = imregionalmax(s);
[y, x] = find(BW.*s);

D = dist([x y]'); D = D + 1000*eye(size(D));
while min(D(:))<10
    N = length(x);
    [i,j] = find(D==min(D(:)));
    x(i(1)) = round((x(i(1)) + x(j(1)))/2);
    y(i(1)) = round((y(i(1)) + y(j(1)))/2);
    x = x(setdiff(1:N,j(1)));
    y = y(setdiff(1:N,j(1)));
    D = dist([x y]'); D = D + 1000*eye(size(D));
end

nDetections = length(x);
boundingBox = repmat(data.averageBoundingBox, [nDetections 1]) + [x x y y];
ind = sub2ind(size(s), y, x);
boxScores = s(ind);
