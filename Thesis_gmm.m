%% GMM training

% %pathToDatabase = "..\..\";
% pathToDatabase = "../../../g/data1a/wa66/Prasanth";
% 
% % add required libraries to the path
% addpath(genpath('../../../g/data1a/wa66/Prasanth/GMM'));
% addpath(genpath('../../../g/data1a/wa66/Prasanth/tDCF_v1/bosaris_toolkit.1.06/bosaris_toolkit'));
% addpath(genpath('../../../g/data1a/wa66/Prasanth/tDCF_v1'));
% 
% load(fullfile(pathToDatabase,'test.mat'));
% load(fullfile(pathToDatabase,'train.mat'));
% 
% fileID = fopen(fullfile(pathToDatabase,'rnn_truth_test.txt'));
% rnn_truth_test = textscan(fileID, '%s');
% fclose(fileID);

% rnn_truth_test = rnn_truth_test{1};

% train GMM for GENUINE data
disp('Training GMM for GENUINE...');
[genuineGMM.m, genuineGMM.s, genuineGMM.w] = vl_gmm(transpose(train_genuine), 512, 'verbose', 'MaxNumIterations',200);
disp('Done!');

% train GMM for SPOOF data
disp('Training GMM for SPOOF...');
[spoofGMM.m, spoofGMM.s, spoofGMM.w] = vl_gmm(transpose(train_spoof), 512, 'verbose', 'MaxNumIterations',200);
disp('Done!');

% process each development trial: feature extraction and scoring
scores = zeros(size(test_data,1),1);
disp('Computing scores for development trials...');
for i=1:length(scores)
    %score computation
    llk_genuine = mean(compute_llk(transpose(squeeze(test_data(i,:,:))),genuineGMM.m,genuineGMM.s,genuineGMM.w));
    llk_spoof = mean(compute_llk(transpose(squeeze(test_data(i,:,:))),spoofGMM.m,spoofGMM.s,spoofGMM.w));
    % compute log-likelihood ratio
    scores(i) = llk_genuine - llk_spoof;
end
disp('Done!');

% compute performance
[Pmiss,Pfa] = rocch(scores(strcmp(ground_truth_test,'1')),scores(strcmp(ground_truth_test,'0')));
EER = rocch2eer(Pmiss,Pfa) * 100; 
fprintf('EER is %.2f\n', EER);
