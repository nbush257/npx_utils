function run_ks2_NEB(cfg,bin_root)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
addpath(genpath('/active/ramirez_j/ramirezlab/nbush/helpers/kilosort2'));
addpath(genpath('/active/ramirez_j/ramirezlab/nbush/helpers/npy-matlab'));
addpath(genpath('/active/ramirez_j/ramirezlab/nbush/helpers/'));
disp(cfg)
disp(bin_root)
% chanMapFile = '/active/ramirez_j/ramirezlab/nbush/helpers/kilosort2/configFiles/neuropixPhase3B2_kilosortChanMap.mat';
sr=SGLXMetaToCoords_NEB(bin_root,0);
SGLXMetaToCoords_NEB(bin_root,1);
SGLXMetaToCoords_NEB(bin_root,2);
root_out = [bin_root '/ks2'];
mkdir(root_out);
ops.trange    = [0 Inf]; % time range to sort
ops.NchanTOT  = 385; % total number of channels in your recording

run(fullfile(cfg))
ops.fproc   = fullfile(root_out, 'temp_wh.dat'); % proc file on a fast SSD


fprintf('Looking for data inside %s \n', bin_root)

% main parameter changes from Kilosort2 to v2.5
ops.sig        = 20;  % spatial smoothness constant for registration
ops.fshigh     = 300; % high-pass more aggresively
ops.nblocks    = 5; % blocks for registration. 0 turns it off, 1 does rigid registration. Replaces "datashift" option. 
ops.useStableMode = false;

% Get the channel map
fs = dir(fullfile(bin_root, '/*ChanMap.mat'));
ops.chanMap = fullfile(bin_root, fs(1).name);

% Get thi binary data
fs = dir(fullfile(bin_root, '/*.bin'));
ops.fbinary = fullfile(bin_root, fs(1).name);
ops.fs=sr;
% preprocess data to create temp_wh.dat
rez = preprocessDataSub(ops);
%
% NEW STEP TO DO DATA REGISTRATION
rez = datashift2(rez, 1); % last input is for shifting data
figure;
set(gcf, 'Color', 'w')
% raster plot of all spikes at their original depths
st_shift = rez.st0(:,2); %+ imin(batch_id)' * dd;
for j = 8:100
    % for each amplitude bin, plot all the spikes of that size in the
    % same shade of gray
    ix = rez.st0(:, 3)==j; % the amplitudes are rounded to integers
    plot(rez.st0(ix, 1)/ops.fs, st_shift(ix), '.', 'color', [1 1 1] * max(0, 1-j/40)) % the marker color here has been carefully tuned
    hold on
end
axis tight

xlabel('time (sec)')
ylabel('spike position (um)')
title('Drift map')
saveas(gcf,[root_out '/drift_map.png'])

close all
figure
plot(rez.dshift)
xlabel('batch')
ylabel('Drift (\mu m)')
saveas(gcf,[root_out '/shift_trace.png'])
close all
% ORDER OF BATCHES IS NOW RANDOM, controlled by random number generator
iseed = 1;
                 
% main tracking and template matching algorithm
rez = learnAndSolve8b(rez, iseed);

% OPTIONAL: remove double-counted spikes - solves issue in which individual spikes are assigned to multiple templates.
% See issue 29: https://github.com/MouseLand/Kilosort/issues/29
rez = remove_ks2_duplicate_spikes(rez);

% final merges
rez = find_merges(rez, 1);

% final splits by SVD
rez = splitAllClusters(rez, 1);

% decide on cutoff
rez = set_cutoff(rez);
% eliminate widely spread waveforms (likely noise)
rez.good = get_good_units(rez);

fprintf('found %d good units \n', sum(rez.good>0))

% write to Phy
fprintf('Saving results to Phy  \n')

rezToPhy_neb(rez, root_out);

end

