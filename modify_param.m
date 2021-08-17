function [] = modify_param(param_fn, ReplaceString)
%%change data [e.g. initial conditions] in model file
% InputFile - string
% OutputFile - string
% SearchString - string
% ReplaceString - string
% read whole model file data into cell array
fid = fopen(param_fn);
data = textscan(fid, '%s', 'Delimiter', '\n', 'CollectOutput', true);
fclose(fid);
% modify the cell array
% find the position where changes need to be applied and insert new data
new_line = sprintf("dat_path = r'%s'",ReplaceString);
data{1}{1} = char(new_line);
% write the modified cell array into the text file
fid = fopen(param_fn, 'w');
for I = 1:length(data{1})
    fprintf(fid, '%s\n', char(data{1}{I}));
end
fclose(fid);