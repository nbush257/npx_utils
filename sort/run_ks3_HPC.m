%% This is the run function for the not motion corrected dynaresp data as of 2021-08-17

function run_ks3_mc(probe_dir)
addpath(genpath('/active/ramirez_j/ramirezlab/nbush/helpers/kilosort3'))
addpath(genpath('/active/ramirez_j/ramirezlab/nbush/helpers/npy-matlab'))
addpath(genpath('/active/ramirez_j/ramirezlab/nbush/projects/npx_utils'))
pathToYourConfigFile = '/active/ramirez_j/ramirezlab/nbush/projects/dynaresp/dynaresp'; 


temp_dir = probe_dir;
ops.trange    = [0 Inf]; % time range to sort
% total number of channels in your recording
ops.NchanTOT = 385;

run(fullfile(pathToYourConfigFile, 'dynaresp_config_ks3_HPC.m'))
load([probe_dir,'/mc_meta.mat'],'sr')

chanmap_dir = dir([probe_dir,'/*kilosortChanMap.mat']);
ops.chanMap=fullfile(probe_dir,chanmap_dir.name);
ops.fs=sr;

%% this block runs all the steps of the algorithm

% find the binary file
bin_fns          = dir(fullfile(probe_dir, '*tcat.imec*ap.bin'));
orig_bin = fullfile(probe_dir, bin_fns(1).name);
ops.fbinary = fullfile(temp_dir, bin_fns(1).name);
ops.fproc = fullfile(temp_dir,'temp_wh.dat');
%%
fprintf('Running KS3 on: \n%s \n', ops.fbinary)

%%
rez                = preprocessDataSub(ops);
rez                = datashift2(rez, 1);

[rez, st3, tF]     = extract_spikes(rez);

rez                = template_learning(rez, tF, st3);

[rez, st3, tF]     = trackAndSort(rez);

rez                = final_clustering(rez, tF, st3);

rez                = find_merges(rez, 1);
%%
ks_dir = fullfile(temp_dir, 'ks3_v2');
mkdir(ks_dir)

rezToPhy2(rez, ks_dir);

%% 
whitened_name = strrep(bin_fns(1).name,'tcat.','tcat.whitened.');
whitened_dest = fullfile(ks_dir,whitened_name);
whitened_dest_windows = strrep(whitened_dest,'/active/ramirez_j/ramirezlab/nbush','Y:');
mv_cmd = sprintf('mv %s %s',ops.fproc,whitened_dest);
system(mv_cmd);
%% Modify paramfile
param_fn = fullfile(ks_dir,'params.py');
modify_param(param_fn,whitened_dest_windows)
%%
plot_sanity(rez,ks_dir);
