function fig = open_partition_plot(matFile)
%OPEN_PARTITION_PLOT Open a partition MAT file and display its 2D plot.
%   fig = OPEN_PARTITION_PLOT(matFile) loads the given MAT file and opens
%   the partition plot using plot_partition_mesh without saving or closing
%   the figure.
%   fig = OPEN_PARTITION_PLOT() opens a file picker first.

    if nargin < 1 || strlength(string(matFile)) == 0
        thisDir = fileparts(mfilename('fullpath'));
        [fileName, filePath] = uigetfile(fullfile(thisDir, '*.mat'), ...
            'Select a partition MAT file');
        if isequal(fileName, 0)
            fprintf('No file selected.\n');
            fig = [];
            return;
        end
        matFile = fullfile(filePath, fileName);
    end

    matFile = char(string(matFile));
    fprintf('Opening partition plot for:\n  %s\n', matFile);

    fig = plot_partition_mesh(matFile, ...
        'PlotMode', '2D', ...
        'ShowNodeLabels', true, ...
        'SaveJpg', false);

    figure(fig);
end
