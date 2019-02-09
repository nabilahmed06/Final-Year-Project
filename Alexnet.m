% Group - 12 
% Kevin Setiawan ID: 25895710
% Nabil Ahmed ID: 25364170
% AlexNet
% Training and Testing of Retinal Images classifying the 3 classs (no DR, mild DR, severe DR)
% Reference: Math Works

%import dataset
imds = imageDatastore('D:\Group12\Retinal\123','IncludeSubfolders',true,'LabelSource','foldernames');

%split dataset into 2 part. Validation and training. 70% and 30%
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

%plot images
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end

%import alexnet
net = alexnet;

%analysize network
analyzeNetwork(net)

%set global variable for input size
inputSize = net.Layers(1).InputSize


layersTransfer = net.Layers(1:end-3);


numClasses = numel(categories(imdsTrain.Labels))
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%data augmentation
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

%validation part
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

%training part
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',20, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');


netTransfer = trainNetwork(augimdsTrain,layers,options);


[YPred,scores] = classify(netTransfer,augimdsValidation);


idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end


YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)



