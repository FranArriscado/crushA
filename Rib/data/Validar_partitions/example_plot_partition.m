thisDir = fileparts(mfilename('fullpath'));
sampleFile = fullfile(thisDir, 'partition_data_layer31_z157.6.mat');
plot_partition_mesh(sampleFile, 'PlotMode', '2D', 'ShowNodeLabels', true);
