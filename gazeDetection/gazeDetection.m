function [direction] = gazeDetection(im, detector, ii)
% This function takes an input face image and determines the gaze
% direction. It expects its inputs to be:
%   im:         An RGB face image with open eyes
%   detector:   System Object output of a Viola Jones eye pair detector

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                    Segment eye pair from face image                     %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A pair of eyes are segmented from the entire face image by using a
% pretrained Viola Jones detector.

% Generate a bounding box of a positive match for an eye pair
bbox = detector(im);

% Only crop the image if there was a positive match
% Image is cropped with buffer so extremities of eyes aren't cropped
if ~isempty(bbox)
    buffer = 100*(700/size(im,2));
    bbox = [bbox(1)-buffer, bbox(2)-buffer, bbox(3)+buffer, bbox(4)+buffer];
    im = imcrop(im, bbox);
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                      Separate left and right eye                        %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The left and right eyes are segemented from each other by looking at the
% total standard deviation of the images in the vertical direction. As in,
% within the local region of the eyes there should be a low standard
% deviation region between the eyes.

% Perform Gaussian blur with std of 5 to reduce std of hair, eyebrows and
% noise.
imGrey = imgaussfilt(rgb2gray(im),5);

% Find the local standard deviation of the image with a 3x3 neighbourhood
imStd = stdfilt(imGrey);
imStdx = sum(imStd, 1);

% Locate the minimal intra-class variance point
T = floor(otsuthresh(imStdx)*size(im, 2));

% Isolate left and right eye based on threshold location T
imLeft = im(:,1:T,:);
imRight = im(:,T:end,:);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                              Locate iris                                %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The iris is located by using the Hough circular transform. Preprocessing
% of the left and right images is done to maximise effectiveness and
% accuracy of the imfindcircles() function.

% Transform to greyscale because imfindcircles() requires grey
imLeft = rgb2gray(imLeft);
imRight = rgb2gray(imRight);

% Rescale so that iris' of each image are roughly the same size
scale = 300/size(imLeft,2);
imLeft = imresize(imLeft, scale);
    
scale = 300/size(imRight,2);
imRight = imresize(imRight, scale);

% Apply Gaussian filter to the image with std of 3, to remove spurious
% edges from hair, eyebrows and noise.
imL = imgaussfilt(imLeft,3);
imR = imgaussfilt(imRight, 3);


% Canny edge detection to accentuate iris edges
imL = edge(imL, 'canny', [0.15 0.45]);
imR = edge(imR, 'canny', [0.15 0.45]);

% Feed into imfindcircles to perform Circular Hough Transform
[centL, radL, metL] = imfindcircles(imL, [22 45], 'Sensitivity', 0.95);
centL = centL(1,:); radL = radL(1); metL = metL(1);

[centR, radR, metR] = imfindcircles(imR, [22 45], 'Sensitivity', 0.95);
centR = centR(1,:); radR = radR(1); metR = metR(1);
    
% Plot results
% figure('units', 'normalized', 'outerposition', [0.7 0 0.3 1])
% subplot(5,2,1), imshow(imLeft)
% viscircles(centL, radL,'EdgeColor','b');
% title(num2str(metL))
% subplot(5,2,3), imshow(imL)
% subplot(5,2,2), imshow(imRight)
% viscircles(centR, radR,'EdgeColor','b');
% title(num2str(metR))
% subplot(5,2,4), imshow(imR)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                         Isolate Eyes w/o eyebrows                       %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The location of the iris will be used to crop the image further to
% include only the eyes and remove eyebrows and any other facial features.
% If the above section failed to locate the iris correctly, further correct
% detection will fail here.

BuffW = 10;
BuffH = 4;
OffH = 20;
% Extract a bounding box based on located iris
bboxL = [centL(1)-BuffW/2*radL, centL(2)-radL-OffH, BuffW*radL, BuffH*radL];
imLeft = imcrop(imLeft, bboxL);

bboxR = [centR(1)-BuffW/2*radR, centR(2)-radR-OffH, BuffW*radR, BuffH*radR];
imRight = imcrop(imRight, bboxR);
% 
% subplot(5,2,5), imshow(imLeft)
% subplot(5,2,6), imshow(imRight)



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                         Locate Iris a second time                       %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This repeats the circular Hough transform to find the iris. The reasoning
% behind this is that the further cropped image will have less incorrect
% iris matches.

% Apply Gaussian filter to the image with std of 3, to remove spurious
% edges from hair, eyebrows and noise.
imL = imgaussfilt(imLeft,3);
imR = imgaussfilt(imRight, 3);

% Canny edge detection to accentuate iris edges
imL = edge(imL, 'canny', [0.15 0.45]);
imR = edge(imR, 'canny', [0.15 0.45]);

% Feed into imfindcircles to perform Circular Hough Transform
[centL, radL, metL] = imfindcircles(imL, [22 45], 'Sensitivity', 0.98);
centL = floor(centL(1,:)); radL = floor(radL(1)); metL = metL(1);

[centR, radR, metR] = imfindcircles(imR, [22 45], 'Sensitivity', 0.98);
centR = floor(centR(1,:)); radR = floor(radR(1)); metR = metR(1);

% Plot results
% subplot(5,2,7), imshow(imLeft)
% viscircles(centL, radL,'EdgeColor','b');
% title(num2str(metL))
% subplot(5,2,9), imshow(imL)
% subplot(5,2,8), imshow(imRight)
% viscircles(centR, radR,'EdgeColor','b');
% title(num2str(metR))
% subplot(5,2,10), imshow(imR)

Zero = zeros(size(imLeft));
St = strel('disk', radL, 8);
Zero(centL(2)-radL:centL(2)+radL-2, centL(1)-radL:centL(1)+radL-2) = St.Neighborhood;
Cent = Zero;
path = '../EYE_valout_out/';
imTrainL = cat(3, imL, imLeft, Cent);
%imTrainL = cat(3, imL, imLeft, imLeft);
imwrite(imTrainL,strcat(path, 'EyeLeftTest', num2str(ii),'.jpg'))

Zero = zeros(size(imRight));
St = strel('disk', radR, 8);
Zero(centR(2)-radR:centR(2)+radR-2, centR(1)-radR:centR(1)+radR-2) = St.Neighborhood;
Cent = Zero;

imTrainR = cat(3, imR, imRight, Cent);
imwrite(imTrainR,strcat(path, 'EyeRightTest', num2str(ii),'.jpg'))


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                         Fit Ellipse Test                                %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fit an ellipse using least mean squared error of the corner features.

% Filter to remove eyebrows
imL = ordfilt2(imLeft, 24, ones(2,12));
imR = ordfilt2(imRight, 24, ones(2,12));

% Blur from the edges in to make it less likely that corners are detection
% far away from iris.
imL = edgetaper(imL, fspecial('gaussian',30,60));
imR = edgetaper(imR, fspecial('gaussian',30,60));


% figure()
% subplot(2,2,3), imshow(imL);
% subplot(2,2,4), imshow(imR);

% Find corners using Shi-Tomasi corner detection algorithm
cL = detectMinEigenFeatures(imL, 'MinQuality', 0.03, 'FilterSize', 3);
cL = cL.selectStrongest(1000);
cR = detectMinEigenFeatures(imR, 'MinQuality', 0.03, 'FilterSize', 3);
cR = cR.selectStrongest(1000);

% Fit least square error ellipse about corner features
% subplot(2,2,1), imshow(imLeft); hold on;
% plot(cL.selectStrongest(100));
% subplot(5,2,7)
ellipse_L = fit_ellipse(cL.Location(:,1),cL.Location(:,2), gca);
% subplot(2,2,2), imshow(imRight); hold on;
% plot(cR.selectStrongest(100));
% subplot(5,2,8)
ellipse_R = fit_ellipse(cR.Location(:,1),cR.Location(:,2), gca);

% Check ellipse area
AreaL = pi*ellipse_L.long_axis * ellipse_L.short_axis/4;
AreaR = pi*ellipse_R.long_axis * ellipse_R.short_axis/4;

% Check iris area
IAreaL = pi*radL^2;
IAreaR = pi*radR^2;

% Confidence
ConfL = AreaL/3/IAreaL;
ConfR = AreaR/3/IAreaR;

if mean(cL.Location(:,1)) > centL(1) + 20
    direction{1} = 'Left';
elseif ellipse_L.X0_in < centL(1) - 20
    direction{1} = 'Right';
else
    direction{1} = 'Straight';
end
if mean(cR.Location(:,1)) > centR(1) + 20
    direction{2} = 'Left';
elseif ellipse_R.X0_in < centR(1) - 20
    direction{2} = 'Right';
else
    direction{2} = 'Straight';
end

% subplot(5,2,7), hold on
% line([mean(cL.Location(:,1)), mean(cL.Location(:,1))], [1, size(imLeft,1)]);
% 
% subplot(5,2,8), hold on
% line([mean(cR.Location(:,1)), mean(cR.Location(:,1))], [1, size(imRight,1)]);

% subplot(2,2,1), title(direction(1))
% subplot(2,2,2), title(direction(2))
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                           Canny Edge Test                               %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Only consider edge information which intersects mid point of iris

% imL = stdfilt(imL);
% imR = stdfilt(imR);
% imL = edge(imL, 'sobel', 0, 'horizontal', 'nothinning');
% imR = edge(imR, 'sobel', 0, 'horizontal', 'nothinning');
imL = imfilter(imL, [-1 -2 -1; 0 0 0; 1 2 1]);
imR = imfilter(imR, [-1 -2 -1; 0 0 0; 1 2 1]);
imL = abs(imL(2:end-1,2:end-1)); imR = abs(imR(2:end-1,2:end-1));
imL = imL - min(imL(:));
imR = imR - min(imR(:));
imL = imL./max(imL(:));
imR = imR./max(imR(:));

% subplot(2,2,3), imshow(double(imL > 0.2),[]);
% subplot(2,2,4), imshow(double(imR > 0.2),[]);




end