% clear all
close all

parameters
testImageSize = [256 256];

% Load detector parameters:
load (dataFile)
NweakClassifiers = length(data.detector);
%NweakClassifiers = [30 120]; % put a list if you want to compare performances with different number of weak learners.

[Dc, jc]  = LMquery(data.databaseStruct, 'object.name', objects);

% remove images used to create the dictionary:
testImages = setdiff(jc, [data.trainingImages; data.dictionary.imagendx']);
NtestImages = length(testImages);

% Define variables used for the precision-recall curve
scoreAxis = linspace(0, 100, 20); RET = []; REL = []; RETREL = [];

% Loop on test images

% NtestImages
for i = 6:N_test_precomputed
    disp(['Computing Feature Vector of Test Image ' , int2str(i),'/',int2str(NtestImages) ]);
    
    % Read image and ground truth
    Img = LMimread(data.databaseStruct, testImages(i), HOMEIMAGES);
    annotation = data.databaseStruct(testImages(i)).annotation;

    % Normalize image:
    [newannotation, newimg, crop, scaling, err, msg] = LMcookimage(annotation, Img, ...
        'objectname', objects, 'objectsize', normalizedObjectSize, 'objectlocation', 'original', 'maximagesize', testImageSize);

    img = double(mean(newimg,3));  % Image to give to the covCrossConv
    
    rows = size(img,1);
    cols = size(img,2);

    % Compute all features for test
    FeaturesTest = zeros(rows*cols, size(data.dictionary.filter,2));   %Matrix with the features of the test image ONLY ONE IMAGE 256x256 points (vectors) by 640 features!!

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
    
    FeaturesTest_img{i} = FeaturesTest;  % Structure with the feature matrixes of test images.
 
end