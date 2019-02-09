% Group - 12 
% Kevin Setiawan ID: 25895710
% Nabil Ahmed ID: 25364170
% GoogLeNet
% Training and Testing of Retinal Images classifying the 3 classs (no DR, mild DR, severe DR)
% Reference: Math Works

% upload dataset
imds = imageDatastore('D:\Group12\Retinal\123','IncludeSubfolders',true,'LabelSource','foldernames');

% seperate the dataset into 2 part. 70% for training and the rest for
% validation
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);

%import google net
net = googlenet;

%analyze the network
analyzeNetwork(net)
net.Layers(1)
inputSize = net.Layers(1).InputSize;


if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 


%find layer that will be replaced
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer]; 


numClasses = numel(categories(imdsTrain.Labels));

% update a new layer
if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end


lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

%get the classes
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

%ploting
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)

ylim([0,10])
layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

%data augmentation
pixelRange = [-30 30];
scaleRange = [0.9 1.1];

imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

%validation part
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

%training part
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',20, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false ,...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain,lgraph,options);

[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == imdsValidation.Labels);

idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end
