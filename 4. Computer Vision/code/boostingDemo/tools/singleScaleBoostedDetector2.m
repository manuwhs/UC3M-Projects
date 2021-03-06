function [Score, boundingBox, boxScores] = singleScaleBoostedDetector2(indx, data,FeaturesTest_img,rows, NweakClassifiers)
%
% This runs the detector at single scale.

% Number of weak detectors:
% if nargin < 3
%     NweakClassifiers = length(data.detector2);
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Precomputed Test 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Once we have the features we just run the boosting. 

[Y] = AdaBoostELM_run(FeaturesTest_img{indx},data.detector2);



Score = vec2mat(Y,rows).'; % Turn vector back into matrix of points

% Look at local maximum of output score and output a set of detected object
% bounding boxes.
s = double(Score>0);
% figure()
% imshow(s);
% pause;
s = conv2(hamming(35),hamming(35),s,'same');
% figure()
% imshow(s);
% pause

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
