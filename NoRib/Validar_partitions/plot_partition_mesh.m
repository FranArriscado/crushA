function fig = plot_partition_mesh(matFile, varargin)
%PLOT_PARTITION_MESH Visualize nodesCoords and connectivity from a MAT file.
%   fig = plot_partition_mesh(matFile) loads the MAT file and renders the
%   nodes together with the edges defined in connectivity.
%
%   fig = plot_partition_mesh(matFile, 'PlotMode', '2D') draws the XY view.
%   fig = plot_partition_mesh(matFile, 'PlotMode', '3D') draws the 3D view.
%   fig = plot_partition_mesh(matFile, 'ShowNodeLabels', true) also draws
%   each node index next to the corresponding point.
%   fig = plot_partition_mesh(..., 'SaveJpg', true) also exports a JPG
%   next to the MAT file using the same base name.
%   fig = plot_partition_mesh(..., 'OutputImage', 'C:\path\plot.jpg')
%   overrides the default JPG destination.
%   The closing edge N -> 1 is highlighted so the loop wrap point is easy
%   to see, and nodes 1 / N are emphasised.

    parser = inputParser;
    parser.addRequired('matFile', @(x) ischar(x) || isstring(x));
    parser.addParameter('PlotMode', '3D', @(x) any(validatestring(x, {'2D', '3D'})));
    parser.addParameter('ShowNodeLabels', true, @(x) islogical(x) || isnumeric(x));
    parser.addParameter('NodeColor', [0.12, 0.47, 0.71], @(x) isnumeric(x) && numel(x) == 3);
    parser.addParameter('EdgeColor', [0.15, 0.15, 0.15], @(x) isnumeric(x) && numel(x) == 3);
    parser.addParameter('ClosingEdgeColor', [0.85, 0.15, 0.15], @(x) isnumeric(x) && numel(x) == 3);
    parser.addParameter('SaveJpg', false, @(x) islogical(x) || isnumeric(x));
    parser.addParameter('OutputImage', "", @(x) ischar(x) || isstring(x));
    parser.parse(matFile, varargin{:});

    matFile = char(parser.Results.matFile);
    plotMode = validatestring(parser.Results.PlotMode, {'2D', '3D'});
    showNodeLabels = logical(parser.Results.ShowNodeLabels);
    nodeColor = parser.Results.NodeColor;
    edgeColor = parser.Results.EdgeColor;
    closingEdgeColor = parser.Results.ClosingEdgeColor;
    saveJpg = logical(parser.Results.SaveJpg);
    outputImage = char(parser.Results.OutputImage);

    matFile = char(string(matFile));

    if ~isfile(matFile)
        error('File not found: %s', matFile);
    end

    data = load(matFile);

    if ~isfield(data, 'nodesCoords')
        error('The MAT file does not contain a variable named "nodesCoords".');
    end

    if ~isfield(data, 'connectivity')
        error('The MAT file does not contain a variable named "connectivity".');
    end

    nodes = data.nodesCoords;
    connectivity = data.connectivity;

    validateattributes(nodes, {'numeric'}, {'2d', 'ncols', 3, 'real', 'finite'}, ...
        mfilename, 'nodesCoords');
    validateattributes(connectivity, {'numeric'}, {'2d', 'ncols', 2, 'integer', 'positive'}, ...
        mfilename, 'connectivity');

    nodeCount = size(nodes, 1);
    if any(connectivity(:) > nodeCount)
        error('Connectivity references node indices outside the range 1..%d.', nodeCount);
    end

    fig = figure('Color', 'w', 'Name', sprintf('Mesh View (%s): %s', plotMode, matFile));
    ax = axes(fig);
    hold(ax, 'on');

    labelOffsets = computeLabelOffsets(nodes);
    closingEdgeIdx = find(connectivity(:, 1) == nodeCount & connectivity(:, 2) == 1, 1);

    if strcmp(plotMode, '2D')
        for edgeIdx = 1:size(connectivity, 1)
            pair = connectivity(edgeIdx, :);
            edgePoints = nodes(pair, 1:2);
            thisEdgeColor = edgeColor;
            thisLineWidth = 1.2;
            if ~isempty(closingEdgeIdx) && edgeIdx == closingEdgeIdx
                thisEdgeColor = closingEdgeColor;
                thisLineWidth = 2.0;
            end
            plot(ax, edgePoints(:, 1), edgePoints(:, 2), ...
                '-', 'Color', thisEdgeColor, 'LineWidth', thisLineWidth);
        end

        scatter(ax, nodes(:, 1), nodes(:, 2), ...
            40, nodeColor, 'filled', 'MarkerEdgeColor', 'k');
        scatter(ax, nodes(1, 1), nodes(1, 2), ...
            85, [0.10, 0.60, 0.10], 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.9);
        scatter(ax, nodes(end, 1), nodes(end, 2), ...
            85, closingEdgeColor, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.9);

        if showNodeLabels
            for nodeIdx = 1:nodeCount
                labelText = sprintf('%d', nodeIdx);
                labelColor = [0.1, 0.1, 0.1];
                fontWeight = 'normal';
                if nodeIdx == 1
                    labelText = sprintf('1 (start)');
                    labelColor = [0.10, 0.45, 0.10];
                    fontWeight = 'bold';
                elseif nodeIdx == nodeCount
                    labelText = sprintf('%d (end)', nodeIdx);
                    labelColor = closingEdgeColor;
                    fontWeight = 'bold';
                end
                text(ax, nodes(nodeIdx, 1) + labelOffsets(nodeIdx, 1), ...
                    nodes(nodeIdx, 2) + labelOffsets(nodeIdx, 2), ...
                    labelText, 'FontSize', 8, 'Color', labelColor, ...
                    'FontWeight', fontWeight, 'Clipping', 'on');
            end
        end

        xlabel(ax, 'X');
        ylabel(ax, 'Y');
        title(ax, sprintf('2D View   Nodes: %d   Edges: %d   Closing edge: %d -> 1', ...
            nodeCount, size(connectivity, 1), nodeCount));
    else
        for edgeIdx = 1:size(connectivity, 1)
            pair = connectivity(edgeIdx, :);
            edgePoints = nodes(pair, :);
            thisEdgeColor = edgeColor;
            thisLineWidth = 1.2;
            if ~isempty(closingEdgeIdx) && edgeIdx == closingEdgeIdx
                thisEdgeColor = closingEdgeColor;
                thisLineWidth = 2.0;
            end
            plot3(ax, edgePoints(:, 1), edgePoints(:, 2), edgePoints(:, 3), ...
                '-', 'Color', thisEdgeColor, 'LineWidth', thisLineWidth);
        end

        scatter3(ax, nodes(:, 1), nodes(:, 2), nodes(:, 3), ...
            40, nodeColor, 'filled', 'MarkerEdgeColor', 'k');
        scatter3(ax, nodes(1, 1), nodes(1, 2), nodes(1, 3), ...
            85, [0.10, 0.60, 0.10], 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.9);
        scatter3(ax, nodes(end, 1), nodes(end, 2), nodes(end, 3), ...
            85, closingEdgeColor, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.9);

        if showNodeLabels
            for nodeIdx = 1:nodeCount
                labelText = sprintf('%d', nodeIdx);
                labelColor = [0.1, 0.1, 0.1];
                fontWeight = 'normal';
                if nodeIdx == 1
                    labelText = sprintf('1 (start)');
                    labelColor = [0.10, 0.45, 0.10];
                    fontWeight = 'bold';
                elseif nodeIdx == nodeCount
                    labelText = sprintf('%d (end)', nodeIdx);
                    labelColor = closingEdgeColor;
                    fontWeight = 'bold';
                end
                text(ax, nodes(nodeIdx, 1) + labelOffsets(nodeIdx, 1), ...
                    nodes(nodeIdx, 2) + labelOffsets(nodeIdx, 2), ...
                    nodes(nodeIdx, 3) + labelOffsets(nodeIdx, 3), ...
                    labelText, 'FontSize', 8, 'Color', labelColor, ...
                    'FontWeight', fontWeight, 'Clipping', 'on');
            end
        end

        xlabel(ax, 'X');
        ylabel(ax, 'Y');
        zlabel(ax, 'Z');
        view(ax, 3);
        rotate3d(fig, 'on');
        title(ax, sprintf('3D View   Nodes: %d   Edges: %d   Closing edge: %d -> 1', ...
            nodeCount, size(connectivity, 1), nodeCount));
    end

    axis(ax, 'equal');
    grid(ax, 'on');
    hold(ax, 'off');

    if saveJpg
        if strlength(string(outputImage)) == 0
            [matDir, matName] = fileparts(matFile);
            outputImage = fullfile(matDir, [matName, '.jpg']);
        end

        exportgraphics(fig, outputImage, 'Resolution', 300);
        fprintf('Saved JPG to: %s\n', outputImage);
    end
end

function labelOffsets = computeLabelOffsets(nodes)
    span = max(nodes, [], 1) - min(nodes, [], 1);
    fallbackSpan = max(max(abs(nodes), [], 1), 1);
    span(span == 0) = fallbackSpan(span == 0);
    baseOffset = 0.018 * span;
    centroid = mean(nodes, 1);

    dirVec = nodes - centroid;
    dirNorm = sqrt(sum(dirVec.^2, 2));
    dirNorm(dirNorm == 0) = 1;
    dirUnit = dirVec ./ dirNorm;

    labelOffsets = dirUnit .* baseOffset;
    zeroRows = all(abs(labelOffsets) < 1e-12, 2);
    labelOffsets(zeroRows, :) = repmat(baseOffset, sum(zeroRows), 1);
end
