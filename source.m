% Load the ground truth data
data = load('gTruth.mat');
gTruth = data.gTruth; % Access the ground truth object

% Access the data source from the gTruth object
imageFilePaths = gTruth.DataSource.Source; % File paths to the images

% Access the pixel label data from gTruth
pixelLabelData = gTruth.LabelData; % This contains the labeled masks

% Verify the extracted data
disp(imageFilePaths);
disp(pixelLabelData);

% Number of images
numImages = numel(imageFilePaths);

% Randomize indices
idx = randperm(numImages);

% Define split ratio
splitRatio = 0.8;
numTrain = round(splitRatio * numImages);

% Split file paths and labels into training and testing sets
trainImagePaths = imageFilePaths(idx(1:numTrain));
testImagePaths = imageFilePaths(idx(numTrain+1:end));

trainLabels = pixelLabelData(idx(1:numTrain), :);
testLabels = pixelLabelData(idx(numTrain+1:end), :);

% Extract file paths from train and test labels tables
trainLabelPaths = trainLabels{:, 1}; % Assuming the paths are in the first column
testLabelPaths = testLabels{:, 1};   % Adjust column index if necessary

% Load training images and their corresponding masks
trainImages = cellfun(@imread, trainImagePaths, 'UniformOutput', false);
trainMasks = cellfun(@imread, trainLabelPaths, 'UniformOutput', false);

% Load testing images and their corresponding masks
testImages = cellfun(@imread, testImagePaths, 'UniformOutput', false);
testMasks = cellfun(@imread, testLabelPaths, 'UniformOutput', false);

% Define augmentation parameters
augmenter = imageDataAugmenter('RandRotation', [-10, 10], ...  % Random rotation between -10 and 10 degrees
                               'RandXReflection', true, ...    % Random horizontal flip
                               'RandYReflection', true, ...    % Random vertical flip
                               'RandXTranslation', [-5, 5], ...% Random translation along x-axis
                               'RandYTranslation', [-5, 5]);   % Random translation along y-axis

% Augment training images and masks
augmentedTrainImages = cellfun(@(img) augment(augmenter, img), trainImages, 'UniformOutput', false);
augmentedTrainMasks = cellfun(@(mask) augment(augmenter, mask), trainMasks, 'UniformOutput', false);

% Initialize arrays to store features and labels
features = [];
labels = [];

% Loop through each training mask to extract features
for i = 1:numel(trainMasks)
    % Read the current mask
    mask = trainMasks{i};

    % Convert mask to binary if necessary
    binaryMask = imbinarize(mask);

    % Extract features using regionprops
    props = regionprops(binaryMask, 'Area', 'Perimeter', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength');

    % Convert structure to an array of feature vectors
    for j = 1:length(props)
        featureVector = [props(j).Area, props(j).Perimeter, props(j).Eccentricity, ...
                         props(j).MajorAxisLength, props(j).MinorAxisLength];
        features = [features; featureVector];
        
        % Assign a label (1 for crack, 0 for non-crack)
        labels = [labels; 1]; % Adjust this according to your data annotations
    end
end

% Train an SVM classifier using the extracted features and labels
svmModel = fitcsvm(features, labels, 'KernelFunction', 'linear', 'Standardize', true);

% Predict on the training data to evaluate performance
predictedLabels = predict(svmModel, features);

% Calculate accuracy
accuracy = sum(predictedLabels == labels) / numel(labels);
fprintf('Training Accuracy: %.2f%%\n', accuracy * 100);

% Example of extracting features from testing masks
testFeatures = [];
testLabels = [];

for i = 1:numel(testMasks)
    mask = testMasks{i};
    binaryMask = imbinarize(mask);
    props = regionprops(binaryMask, 'Area', 'Perimeter', 'Eccentricity', 'MajorAxisLength', 'MinorAxisLength');

    for j = 1:length(props)
        featureVector = [props(j).Area, props(j).Perimeter, props(j).Eccentricity, ...
                         props(j).MajorAxisLength, props(j).MinorAxisLength];
        testFeatures = [testFeatures; featureVector];
        testLabels = [testLabels; 1]; % Adjust according to your annotations
    end
end

% Predict on the testing features
testPredictions = predict(svmModel, testFeatures);

% Calculate testing accuracy
testAccuracy = sum(testPredictions == testLabels) / numel(testLabels);
fprintf('Testing Accuracy: %.2f%%\n', testAccuracy * 100);

% Calculate the confusion matrix
confusionMatrix = confusionmat(testLabels, testPredictions);

% Calculate precision, recall, and F1-score
precision = confusionMatrix(1, 1) / sum(confusionMatrix(:, 1));
recall = confusionMatrix(1, 1) / sum(confusionMatrix(1, :));
f1Score = 2 * (precision * recall) / (precision + recall);

fprintf('Precision: %.2f%%\n', precision * 100);
fprintf('Recall: %.2f%%\n', recall * 100);
fprintf('F1-Score: %.2f%%\n', f1Score * 100);

% Perform k-fold cross-validation (e.g., 5 folds)
cvModel = crossval(svmModel, 'KFold', 5);
crossValAccuracy = 1 - kfoldLoss(cvModel);
fprintf('Cross-Validation Accuracy: %.2f%%\n', crossValAccuracy * 100);

% Load the ground truth data
data = load('gTruth.mat');
gTruth = data.gTruth; % Access the ground truth object

% Extract image file paths and pixel label data
imageFilePaths = gTruth.DataSource.Source;
pixelLabelData = gTruth.LabelData;

% Split the data into training and testing sets
numImages = numel(imageFilePaths);
idx = randperm(numImages);
splitRatio = 0.8;
numTrain = round(splitRatio * numImages);
testImagePaths = imageFilePaths(idx(numTrain+1:end));
testLabelPaths = pixelLabelData{idx(numTrain+1:end), 1};

% Load testing images and masks
testImages = cellfun(@imread, testImagePaths, 'UniformOutput', false);
testMasks = cellfun(@imread, testLabelPaths, 'UniformOutput', false);

% Processing and displaying the images step-by-step
figure;
for i = 1:numel(testImages)
    % Step 1: Input Image
    inputImage = testImages{i};
    subplot(1, 5, 1);
    imshow(inputImage);
    title('Input');
    
    % Step 2: Thresholding (adaptive thresholding for crack segmentation)
    grayImage = rgb2gray(inputImage);
    thresholdedImage = imbinarize(grayImage, 'adaptive', 'Sensitivity', 0.5);
    subplot(1, 5, 2);
    imshow(thresholdedImage);
    title('Thresholded');
    
    % Step 3: Morphological Processing (remove noise and refine cracks)
    morphProcessed = imopen(thresholdedImage, strel('disk', 2)); % Morphological opening
    morphProcessed = imclose(morphProcessed, strel('disk', 2)); % Morphological closing
    subplot(1, 5, 3);
    imshow(morphProcessed);
    title('Morph. Processed');
    
    % Step 4: Filtering (cleaning up using connected components)
    filteredImage = bwareaopen(morphProcessed, 50); % Remove small objects (tune size threshold)
    subplot(1, 5, 4);
    imshow(filteredImage);
    title('Filtered');
    
    % Step 5: Ground Truth
    groundTruthMask = imbinarize(testMasks{i}); % Load the ground truth mask
    groundTruthMask = imresize(groundTruthMask, size(filteredImage)); % Resize if necessary
    subplot(1, 5, 5);
    imshow(groundTruthMask);
    title('Ground Truth');
    
    % Pause to review each set of results
    pause(2);
end
