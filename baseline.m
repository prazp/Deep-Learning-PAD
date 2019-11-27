%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ASVspoof 2019
% Automatic Speaker Verification Spoofing and Countermeasures Challenge
%
% http://www.asvspoof.org/
%
% ============================================================================================
% Matlab implementation of spoofing detection baseline system based on:
%   - linear frequency cepstral coefficients (LFCC) features + Gaussian Mixture Models (GMMs)
%   - constant Q cepstral coefficients (CQCC) features + Gaussian Mixture Models (GMMs)
% ============================================================================================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%clear; close all; clc;

% add required libraries to the path
addpath(genpath('LFCC'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('GMM'));
addpath(genpath('bosaris_toolkit'));
addpath(genpath('tDCF_v1'));

% set here the experiment to run (access and feature type)
feature_type = 'CQCC'; % LFCC or CQCC

% set paths to the wave files and protocols

% TODO: in this code we assume that the data follows the directory structure:
%
% ASVspoof_root/
%   |- LA
%      |- ASVspoof2019_LA_dev_asv_scores_v1.txt
% 	   |- ASVspoof2019_LA_dev_v1/
% 	   |- ASVspoof2019_LA_protocols_v1/
% 	   |- ASVspoof2019_LA_train_v1/
%   |- PA
%      |- ASVspoof2019_PA_dev_asv_scores_v1.txt
%      |- ASVspoof2019_PA_dev_v1/
%      |- ASVspoof2019_PA_protocols_v1/
%      |- ASVspoof2019_PA_train_v1/

pathToDatabase = fullfile('..', 'wav');
trainProtocolFile = fullfile('..', 'CM_protocol', 'cm_train.trn');
devProtocolFile = fullfile('..', 'CM_protocol', 'cm_develop.ndx');
evalProtocolFile = fullfile('..', 'CM_protocol', 'cm_evaluation.ndx');

% read train protocol
fileID = fopen(trainProtocolFile);
protocol = textscan(fileID, '%s%s%s%s');
fclose(fileID);

% get file and label lists
folderlist = protocol{1};
filelist = protocol{2};
key = protocol{4};

% get indices of genuine and spoof files
genuineIdx = find(strcmp(key,'human'));
spoofIdx = find(strcmp(key,'spoof'));

data_size = 16000*5;

%% Feature extraction for training data

% extract features for GENUINE training data and store in cell array
disp('Extracting features for GENUINE training data...');
genuineFeatureCell = cell(size(genuineIdx));
for i=1:length(genuineIdx)
    filePath = fullfile(pathToDatabase,folderlist{genuineIdx(i)},filelist{genuineIdx(i)}+".wav");
    [x,fs] = audioread(filePath);
    if (length(x) < data_size)
        x = padarray(x, [data_size-length(x), 0], 'post', 'circular');
    end
    genuineFeatureCell{i} = cqcc(x(1:data_size), fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
end
disp('Done!');

% extract features for SPOOF training data and store in cell array
disp('Extracting features for SPOOF training data...');
spoofFeatureCell = cell(size(spoofIdx));
for i=1:length(spoofIdx)
    filePath = fullfile(pathToDatabase,folderlist{spoofIdx(i)},filelist{spoofIdx(i)}+".wav");
    [x,fs] = audioread(filePath);
    if (length(x) < data_size)
        x = padarray(x, [data_size-length(x), 0], 'post', 'circular');
    end
    spoofFeatureCell{i} = cqcc(x(1:data_size), fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
end
disp('Done!');

%% GMM training

% train GMM for GENUINE data
disp('Training GMM for GENUINE...');
[genuineGMM.m, genuineGMM.s, genuineGMM.w] = vl_gmm([genuineFeatureCell{:}], 512, 'verbose', 'MaxNumIterations',100);
disp('Done!');

% train GMM for SPOOF data
disp('Training GMM for SPOOF...');
[spoofGMM.m, spoofGMM.s, spoofGMM.w] = vl_gmm([spoofFeatureCell{:}], 512, 'verbose', 'MaxNumIterations',100);
disp('Done!');


%% Feature extraction and scoring of development data

% read development protocol
fileID = fopen(devProtocolFile);
protocol = textscan(fileID, '%s%s%s%s');
fclose(fileID);

% get file and label lists
folderlist = protocol{1};
filelist = protocol{2};
labels = protocol{4};

% process each development trial: feature extraction and scoring
scores = zeros(size(filelist));
disp('Computing scores for development trials...');
dev_data = cell(size(filelist));
for i=1:length(filelist)
    filePath = fullfile(pathToDatabase,folderlist{i},filelist{i}+".wav");
    [x,fs] = audioread(filePath);
    % featrue extraction
    if (length(x) < data_size)
        x = padarray(x, [data_size-length(x), 0], 'post', 'circular');
    end
    x_cqcc = cqcc(x(1:data_size), fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    
    dev_data{i} = cqcc(x(1:data_size), fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    
    %score computation
    llk_genuine = mean(compute_llk(x_cqcc,genuineGMM.m,genuineGMM.s,genuineGMM.w));
    llk_spoof = mean(compute_llk(x_cqcc,spoofGMM.m,spoofGMM.s,spoofGMM.w));
    %compute log-likelihood ratio
    scores(i) = llk_genuine - llk_spoof;
end
disp('Done!');

% compute performance
[Pmiss,Pfa] = rocch(scores(strcmp(labels,'human')),scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100; 
fprintf('Development EER is %.2f\n', EER);

%% Feature extraction and scoring of evaluation data

% % read development protocol
% fileID = fopen(evalProtocolFile);
% protocol = textscan(fileID, '%s%s%s%s');
% fclose(fileID);
% 
% % get file and label lists
% folderlist = protocol{1};
% filelist = protocol{2};
% labels = protocol{4};
% 
% % process each development trial: feature extraction and scoring
% scores = zeros(size(filelist));
% disp('Computing scores for evaluation trials...');
% eval_data = cell(size(filelist));
% for i=1:length(filelist)
%     filePath = fullfile(pathToDatabase,folderlist{i},filelist{i}+".wav");
%     [x,fs] = audioread(filePath);
%     % featrue extraction
%     if (length(x) < data_size)
%         x = padarray(x, [data_size-length(x), 0], 'post', 'circular');
%     end
%     x_cqcc = cqcc(x(1:data_size), fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
%     eval_data{i} = cqcc(x(1:data_size), fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
%     
%     %score computation
%     llk_genuine = mean(compute_llk(x_cqcc,genuineGMM.m,genuineGMM.s,genuineGMM.w));
%     llk_spoof = mean(compute_llk(x_cqcc,spoofGMM.m,spoofGMM.s,spoofGMM.w));
%     %compute log-likelihood ratio
%     scores(i) = llk_genuine - llk_spoof;
% end
% disp('Done!');

% read development protocol
fileID = fopen(evalProtocolFile);
protocol = textscan(fileID, '%s%s%s%s');
fclose(fileID);

fs = 16000;

% get file and label lists
folderlist = protocol{1};
filelist = protocol{2};
spooftype = protocol{3};
labels = protocol{4};

% get indices of genuine and spoof files
genuineIdx = find(strcmp(spooftype,'human'));
spoofIdx = find(strcmp(spooftype,'S1'));

% process each development trial: feature extraction and scoring
scores = zeros(size(filelist));
disp('Computing scores for evaluation trials...');
eval_data_extracted = cell(size(filelist));

k = 1;

for i=1:length(spoofIdx)
%     if (i ~= genuineIdx(k) && i ~= spoofIdx(k))
%         continue;
%     else
%         k = k + 1;
%     end
    disp(i);
    x = eval_data{spoofIdx(i)};
    % featrue extraction
    if (length(x) < data_size)
        x = padarray(x, [data_size-length(x), 0], 'post', 'circular');
    end
    x_cqcc = cqcc(x(1:data_size), fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    eval_data_extracted{i} = x_cqcc;
    
    %score computation
    llk_genuine = mean(compute_llk(x_cqcc,genuineGMM.m,genuineGMM.s,genuineGMM.w));
    llk_spoof = mean(compute_llk(x_cqcc,spoofGMM.m,spoofGMM.s,spoofGMM.w));
    %compute log-likelihood ratio
    scores(i) = llk_genuine - llk_spoof;
end
disp('Done!');

% compute performance
[Pmiss,Pfa] = rocch(scores(strcmp(labels,'human')),scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100; 
fprintf('Evaluation EER is %.2f\n', EER);
