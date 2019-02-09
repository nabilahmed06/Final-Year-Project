% Group - 12 
% Kevin Setiawan ID: 25895710
% Nabil Ahmed ID: 25364170
% Testing of Retinal Images classifying the 3 classs (no DR, mild DR, severe DR)


%I = imread('D:\Group12\Dataset_Nabil\3\20051021_39314_0100_PP.tif');
%I = imread('D:\Group12\Dataset_Nabil\3\20051020_45068_0100_PP.tif');

%I = imread('D:\Group12\Dataset_Nabil\2\20051201_37462_0400_PP.tif');
%I = imread('D:\Group12\Dataset_Nabil\2\20051216_45619_0200_PP.tif');

%I = imread('D:\Group12\Dataset_Nabil\1\20051214_41358_0100_PP.tif');
%I = imread('D:\Group12\Dataset_Nabil\1\20060410_39994_0200_PP.tif');



%Classify using AlexNet

%load the training architecture
load net_transfer_alexnet.mat
Alexnet = netTransfer;

%get input size for alexnet and later resize images
inputSize = Alexnet.Layers(1).InputSize;
classNames = Alexnet.Layers(end).ClassNames;

I = imresize(I,inputSize(1:2));
[label,alexScores] = classify(Alexnet,I);
disp(label);
disp(alexScores);

disp("Alexnet classify as " + string(label) + ", " + num2str(100*alexScores(classNames == label),3) + "%");



%Classify using GoogleNet

%load the training architecture

load net_transfer_googlenet.mat
Googlenet = net;

%get input size for alexnet and later resize images
inputSize = Googlenet.Layers(1).InputSize;
classNames2 = Googlenet.Layers(end).ClassNames;

I = imresize(I,inputSize(1:2));
[label,googleScores] = classify(Googlenet,I);
disp(label);
disp(googleScores);


disp("GoogleNet classify as " + string(label) + ", " + num2str(100*googleScores(classNames2 == label),3) + "%");


%Late Fusion Technique Score Based

%From each class, we get a average and we take the max accuracy to get the
%final classification
max = 0;
maxIndex = 0;

for i = 1:3
    fusionScore = 100 * (googleScores(i) + alexScores(i))/2;
    if fusionScore > max
       max = fusionScore;
       maxIndex = i;
    end
end

disp("After Fusion,it classified as "  +string(maxIndex) +  ", " + num2str(max) + "%");
figure
imshow(I)



