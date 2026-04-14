function plot_partition_layers_sections(matFile)
%PLOT_PARTITION_LAYERS_SECTIONS Show partition layers in 3D and labeled 2D.
%   plot_partition_layers_sections(matFile) loads a MAT file containing a
%   layer struct array, plots every detected layer in 3D, then prompts the
%   user to choose which layers to inspect as labeled 2D sections.
%
%   The loader first looks for struct fields named:
%     - NodesCoordinates / ConnectivityList
%     - nodesCoords / connectivity
%   If those names are not found, it falls back to the first struct field
%   pair that looks like [N x 3] node coordinates and [M x 2] connectivity.

    if nargin < 1 || strlength(string(matFile)) == 0
        [fileName, filePath] = uigetfile('*.mat', 'Select a partition layers MAT file');
        if isequal(fileName, 0)
            fprintf('No file selected.\n');
            return;
        end
        matFile = fullfile(filePath, fileName);
    end

    matFile = char(string(matFile));
    if ~isfile(matFile)
        error('File not found: %s', matFile);
    end

    data = load(matFile);
    [layers, variableName, nodesField, connectivityField] = findLayerStruct(data);

    numLayers = numel(layers);
    nodesPerLayer = cell(numLayers, 1);
    connectivityPerLayer = cell(numLayers, 1);
    validLayerMask = false(numLayers, 1);
    layerLabels = strings(numLayers, 1);

    for layerIdx = 1:numLayers
        [nodes, connectivity] = normalizeLayer(layers(layerIdx), nodesField, connectivityField);
        nodesPerLayer{layerIdx} = nodes;
        connectivityPerLayer{layerIdx} = connectivity;

        if ~isempty(nodes)
            validLayerMask(layerIdx) = true;
        end

        layerLabels(layerIdx) = sprintf('Layer %d   Nodes: %d   Edges: %d', ...
            layerIdx, size(nodes, 1), size(connectivity, 1));
    end

    validLayerIdx = find(validLayerMask);
    if isempty(validLayerIdx)
        error('No layers with node coordinates were found in %s.', matFile);
    end

    plotLayers3D(matFile, variableName, validLayerIdx, nodesPerLayer, connectivityPerLayer);

    selectedLayerIdx = selectLayersToPlot(layerLabels(validLayerIdx), validLayerIdx);
    if isempty(selectedLayerIdx)
        fprintf('No 2D layer selection made. 3D plot remains open.\n');
        return;
    end

    plotLayers2D(matFile, selectedLayerIdx, nodesPerLayer, connectivityPerLayer);
end

function [layers, variableName, nodesField, connectivityField] = findLayerStruct(data)
    variableNames = fieldnames(data);
    preferredPairs = {
        'NodesCoordinates', 'ConnectivityList';
        'nodesCoords',      'connectivity';
        'NodesCoords',      'ConnectivityList';
        'NodesCoordinates', 'connectivity'
    };

    for varIdx = 1:numel(variableNames)
        variableName = variableNames{varIdx};
        value = data.(variableName);
        if ~isstruct(value) || isempty(value)
            continue;
        end

        fieldNames = fieldnames(value);
        for pairIdx = 1:size(preferredPairs, 1)
            nodesField = preferredPairs{pairIdx, 1};
            connectivityField = preferredPairs{pairIdx, 2};
            if ismember(nodesField, fieldNames) && ismember(connectivityField, fieldNames)
                layers = value;
                return;
            end
        end

        [nodesField, connectivityField] = inferLayerFields(value);
        if ~isempty(nodesField)
            layers = value;
            return;
        end
    end

    error(['Could not find a layer struct array in the MAT file. ', ...
        'Expected fields like NodesCoordinates/ConnectivityList or nodesCoords/connectivity.']);
end

function [nodesField, connectivityField] = inferLayerFields(layers)
    nodesField = '';
    connectivityField = '';
    fieldNames = fieldnames(layers);

    for i = 1:numel(fieldNames)
        for j = 1:numel(fieldNames)
            if i == j
                continue;
            end

            candidateNodesField = fieldNames{i};
            candidateConnectivityField = fieldNames{j};

            for layerIdx = 1:numel(layers)
                nodes = layers(layerIdx).(candidateNodesField);
                connectivity = layers(layerIdx).(candidateConnectivityField);

                if ismatrix(nodes) && isnumeric(nodes) && size(nodes, 2) == 3 && ...
                        ismatrix(connectivity) && isnumeric(connectivity) && size(connectivity, 2) == 2
                    nodesField = candidateNodesField;
                    connectivityField = candidateConnectivityField;
                    return;
                end
            end
        end
    end
end

function [nodes, connectivity] = normalizeLayer(layerData, nodesField, connectivityField)
    nodes = layerData.(nodesField);
    connectivity = layerData.(connectivityField);

    if isempty(nodes)
        nodes = zeros(0, 3);
    else
        validateattributes(nodes, {'numeric'}, {'2d', 'ncols', 3, 'real', 'finite'});
        nodes = double(nodes);
    end

    if isempty(connectivity)
        connectivity = zeros(0, 2);
    else
        validateattributes(connectivity, {'numeric'}, {'2d', 'ncols', 2, 'real', 'finite'});
        connectivity = round(double(connectivity));
        validRows = all(connectivity >= 1, 2) & all(connectivity <= size(nodes, 1), 2) & ...
            connectivity(:, 1) ~= connectivity(:, 2);
        connectivity = connectivity(validRows, :);
    end
end

function plotLayers3D(matFile, variableName, validLayerIdx, nodesPerLayer, connectivityPerLayer)
    fig = figure('Color', 'w', 'Name', sprintf('Partition Layers 3D: %s', matFile));
    ax = axes(fig);
    hold(ax, 'on');
    colors = lines(max(numel(validLayerIdx), 7));

    for localIdx = 1:numel(validLayerIdx)
        layerIdx = validLayerIdx(localIdx);
        nodes = nodesPerLayer{layerIdx};
        connectivity = connectivityPerLayer{layerIdx};
        color = colors(1 + mod(localIdx - 1, size(colors, 1)), :);

        for edgeIdx = 1:size(connectivity, 1)
            edgeNodes = connectivity(edgeIdx, :);
            xyz = nodes(edgeNodes, :);
            plot3(ax, xyz(:, 1), xyz(:, 2), xyz(:, 3), '-', ...
                'Color', color, 'LineWidth', 1.1);
        end

        scatter3(ax, nodes(:, 1), nodes(:, 2), nodes(:, 3), ...
            16, color, 'filled', 'MarkerEdgeColor', 'k');

        centroid = mean(nodes, 1);
        text(ax, centroid(1), centroid(2), centroid(3), sprintf('L%d', layerIdx), ...
            'FontSize', 9, 'FontWeight', 'bold', 'Color', color, ...
            'BackgroundColor', 'w', 'Margin', 0.15);
    end

    axis(ax, 'equal');
    grid(ax, 'on');
    view(ax, 3);
    rotate3d(fig, 'on');
    xlabel(ax, 'X');
    ylabel(ax, 'Y');
    zlabel(ax, 'Z');
    title(ax, sprintf('3D Partition Layers   Variable: %s   Layers: %d', ...
        variableName, numel(validLayerIdx)));
    hold(ax, 'off');
end

function selectedLayerIdx = selectLayersToPlot(validLayerLabels, validLayerIdx)
    prompt = sprintf(['Select one or more layers for the 2D section plot.\n', ...
        'Close the dialog or press Cancel to skip the 2D view.']);

    try
        [selection, ok] = listdlg( ...
            'PromptString', prompt, ...
            'ListString', cellstr(validLayerLabels), ...
            'SelectionMode', 'multiple', ...
            'ListSize', [380, 260], ...
            'Name', 'Select Layers');

        if ok
            selectedLayerIdx = validLayerIdx(selection);
        else
            selectedLayerIdx = [];
        end
    catch
        fprintf('%s\n', prompt);
        for labelIdx = 1:numel(validLayerLabels)
            fprintf('  %s\n', validLayerLabels(labelIdx));
        end
        selectedLayerIdx = input('Enter layer indices as a vector, for example [3 5 8]: ');
        if isempty(selectedLayerIdx)
            selectedLayerIdx = [];
        else
            selectedLayerIdx = intersect(validLayerIdx, unique(selectedLayerIdx(:)'));
        end
    end
end

function plotLayers2D(matFile, selectedLayerIdx, nodesPerLayer, connectivityPerLayer)
    nPlots = numel(selectedLayerIdx);
    nCols = ceil(sqrt(nPlots));
    nRows = ceil(nPlots / nCols);

    fig = figure('Color', 'w', 'Name', sprintf('Partition Layers 2D: %s', matFile));
    tl = tiledlayout(fig, nRows, nCols, 'TileSpacing', 'compact', 'Padding', 'compact');

    for plotIdx = 1:nPlots
        layerIdx = selectedLayerIdx(plotIdx);
        nodes = nodesPerLayer{layerIdx};
        connectivity = connectivityPerLayer{layerIdx};
        xy = projectNodesToPlane(nodes);
        labelOffset = computeLabelOffset2D(xy);

        ax = nexttile(tl);
        hold(ax, 'on');

        for edgeIdx = 1:size(connectivity, 1)
            edgeNodes = connectivity(edgeIdx, :);
            edgeXY = xy(edgeNodes, :);
            plot(ax, edgeXY(:, 1), edgeXY(:, 2), '-', 'Color', [0.15, 0.15, 0.15], 'LineWidth', 1.1);
        end

        scatter(ax, xy(:, 1), xy(:, 2), 34, [0.10, 0.45, 0.80], 'filled', 'MarkerEdgeColor', 'k');

        for nodeIdx = 1:size(xy, 1)
            text(ax, xy(nodeIdx, 1) + labelOffset(1), xy(nodeIdx, 2) + labelOffset(2), ...
                sprintf('%d', nodeIdx), 'FontSize', 8, 'Color', [0.1, 0.1, 0.1], ...
                'BackgroundColor', 'w', 'Margin', 0.1);
        end

        axis(ax, 'equal');
        grid(ax, 'on');
        xlabel(ax, 'Local X');
        ylabel(ax, 'Local Y');
        title(ax, sprintf('Layer %d   Nodes: %d   Edges: %d', ...
            layerIdx, size(nodes, 1), size(connectivity, 1)));
        hold(ax, 'off');
    end

    title(tl, '2D Sections with Labeled Nodes');
end

function xy = projectNodesToPlane(nodes)
    if size(nodes, 1) <= 1
        xy = [nodes(:, 1), zeros(size(nodes, 1), 1)];
        return;
    end

    centered = nodes - mean(nodes, 1);
    [~, ~, basis] = svd(centered, 'econ');

    if size(basis, 2) < 2
        xy = [nodes(:, 1), nodes(:, 2)];
        return;
    end

    xy = centered * basis(:, 1:2);
end

function offset = computeLabelOffset2D(xy)
    if isempty(xy)
        offset = [0, 0];
        return;
    end

    span = max(xy, [], 1) - min(xy, [], 1);
    span(span == 0) = 1;
    offset = 0.015 * span;
end
