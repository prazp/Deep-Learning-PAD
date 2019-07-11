addpath(genpath('GMM'));
addpath(genpath('bosaris_toolkit'));
addpath(genpath('tDCF_v1'));

[Pmiss,Pfa] = rocch(log10(genuine(strcmp(ground_truth_test,'1'))),log10(genuine(strcmp(ground_truth_test,'0'))));
EER = rocch2eer(Pmiss,Pfa) * 100; 
fprintf('EER is %.2f\n', EER);
