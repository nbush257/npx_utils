function mono_res = run_CCG_analysis(path_list,run_inhibitory)
%% function mono_res = run_CCG_analysis(path_list,run_inhibitory)
% given a cell array of paths to KS2 folders, combine into one spikes
% struct and run CCG analysis to get monosynaptic connections
all_times = [];
all_cluID = [];
all_shanks = [];

% %% for getting waveforms
% basepath = '/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/data/processed/m2020-18/catgt_m2020-18_g0/m2020-18_g0_imec0';
% session = sessionTemplate(basepath,'showGUI',false);
% session.extracellular.fileName = 'm2020-18_g0_tcat.imec0.ap.bin'
% session.spikeSorting = {struct()}
% session.spikeSorting{1}.method='KiloSort'
% session.spikeSorting{1}.format='Phy'
% session.spikeSorting{1}.relativePath='imec0_ks2';
% spikes = loadSpikes('session',session,'showWaveforms',false)
%%


for ii = 1:length(path_list)
    spikes = loadSpikes('basepath',path_list{ii},'getWaveformsFromDat',false);
    grp = tdfread([path_list{ii} '/cluster_group.tsv']);
    kslabel = tdfread([path_list{ii} '/cluster_KSLabel.tsv']);
    is_good_ks = all(kslabel.KSLabel=='good',2);
    is_good_grp = all(grp.group=='good ',2);
    good_clusters = find(is_good_ks.*is_good_grp);
    
    idx = ismember(spikes.cluID,good_clusters);
    times = spikes.times(idx);
    clu_ids = spikes.cluID(idx);
    all_times = [all_times times];
    all_cluID = [all_cluID clu_ids];
    
    shankID = ones(1,length(clu_ids))*(ii-1);
    all_shanks = [all_shanks shankID];

end

allspikes = struct();
allspikes.times = all_times;
allspikes.cluID = all_cluID;
allspikes.shankID = all_shanks;
mono_res = ce_MonoSynConvClick(allspikes,'includeInhibitoryConnections',run_inhibitory);

end

