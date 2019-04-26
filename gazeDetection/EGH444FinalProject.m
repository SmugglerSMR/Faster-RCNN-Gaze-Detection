%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                    EGH444 - Gaze Detection Script                       %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clean up workspace, etc
clearvars; close all; clc;

%% Set Variables
% Set the number of input images
numIm = 15;

% Set Path for the input images
path = '../EYE_val/';

% Set the naming convention using for input images
name = 'EYE_va';

% Set file type
type = '.jpg';

%% Initialise Viola Jones detector
%  'EyesPairBig': Locate both eyes and remove other facial features
%  'MinSize', [50, 100]: Prevent small matches that aren't eyes
detector = vision.CascadeObjectDetector('EyePairBig', 'MinSize', [50, 100]);

%% Feed each image into gaze dection function

for ii = 1:numIm
    %% Load images
    % This assumes the naming convention follows 01, 02, 03 -> 10, 11, etc
    num = dec2base(ii,10) - '0';
    num = [0, num];
    num = strrep(num2str(num(end-1:end)), ' ', '');
    
    im = im2double(imread([path, name, num, type]));

    %% Feed into function
    [direction] = gazeDetection(im, detector, ii);


end

clearvars -except im direction