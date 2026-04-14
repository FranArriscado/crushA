%
%                extract_rib_layer.m
%
%  Headless extraction script: reads an Abaqus .inp file, runs
%  partitioning, and saves one cross-section layer as a .mat file
%  with nodesCoords and connectivity — ready for ribAnalysis.py.
%
%  This script bypasses the CrushAnalytica GUI entirely by manually
%  constructing the minimum required data structures.
%
%  HOW TO USE:
%  1. Set the USER INPUTS below (file path, crush direction, etc.)
%  2. Make sure all CrushAnalytica .m files are on your MATLAB path
%  3. Run: >> extract_rib_layer
%  4. Inspect the plotted layers to find one with a rib
%  5. When prompted, enter the layer index to save
%
%  Francisco Arriscado — FEUP, 2026
%==========================================================================
clear; clc; close all;

%% ======================== USER INPUTS ==================================
% Path to your Abaqus .inp file
inpFileName = 'SIS.inp';  % <-- CHANGE THIS to your actual file

% Partition offset (distance between layers) [mm]
partitionOffset = 10;

% Crush direction: axial crush = along Z (default for SIS)
% For axial: theta=0, phi=0 → normal vector = [0, 0, 1]
% For oblique: set theta to the oblique angle
theta = 0;   % [rad] polar angle (0 = axial)
phi   = 0;   % [rad] azimuth angle

% Crush direction definition
% false = use crush plane normal vector (most common)
% true  = use custom direction vector
crushDirectionDef = false;
directionVec = [0, 0, 1];  % only used if crushDirectionDef = true

% Which layer(s) to save.
% Supported values:
%   0       -> interactive prompt after viewing
%   scalar  -> save one layer, e.g. 12
%   vector  -> save multiple layers, e.g. [3 7 9]
%   'all'   -> save every layer
%   'ribs'  -> save only layers with ribs
targetLayerSelection = 0;

% Output filename base.
% If one layer is saved, this exact filename is used.
% If multiple layers are saved, filenames become:
%   <name>_layer_###.mat
outputFileName = 'partition_data_testing.mat';

%% ======================== STEP 1: READ .INP FILE =======================
fprintf('=== Step 1: Reading .inp file ===\n');
fprintf('File: %s\n', inpFileName);

% abaqusInpRead expects a log function — provide a dummy
log = @(msg) fprintf('  [INP] %s\n', msg);
FEModelData = abaqusInpRead(inpFileName, log);

fprintf('  Nodes: %d\n', size(FEModelData.Nodes(1).Coordinates, 1));
fprintf('  Element sets: %d\n', FEModelData.Elements(1).Count);

%% ======================== STEP 2: IDENTIFY COMPOSITES ==================
fprintf('\n=== Step 2: Identifying composites ===\n');
% compositeIdentifier.m normally needs the full app struct + GUI table.
% We only need the part that assigns CompositeIDs to Elements, which
% meshTreatment requires. This replicates lines 120-131 of
% compositeIdentifier.m without any GUI dependencies.

% Assign composite IDs to each section
for i = 1:length(FEModelData.Sections)
    % Build a composite name from the section's material/layup info
    if ~isempty(FEModelData.Sections(i).LayupName)
        compName = FEModelData.Sections(i).LayupName;
    elseif isfield(FEModelData.Sections(i), 'MaterialLayup') && ...
            ~isempty(FEModelData.Sections(i).MaterialLayup)
        compName = FEModelData.Sections(i).MaterialLayup;
    else
        compName = sprintf('composite_%d', i);
    end
    FEModelData.Sections(i).CompositeIDs = compName;
end

% Map composite IDs to each element via its section's element set
% IMPORTANT: meshTreatment indexes CompositeIDs by element ID value,
% not by sequential index. So the cell array must be large enough
% to accommodate the maximum element ID.
maxElementID = max(FEModelData.Elements(1).ID);
elements_composite = cell(maxElementID, 1);
for i = 1:maxElementID
    elements_composite{i} = 'NotDefined';
end
for i = 1:length(FEModelData.Sections)
    if isfield(FEModelData.Sections(i), 'ElementID') && ...
            ~isempty(FEModelData.Sections(i).ElementID)
        for j = 1:length(FEModelData.Sections(i).ElementID)
            eid = FEModelData.Sections(i).ElementID(j);
            if eid >= 1 && eid <= maxElementID
                elements_composite{eid} = FEModelData.Sections(i).CompositeIDs;
            end
        end
    end
end
FEModelData.Elements(1).CompositeIDs = elements_composite;

% meshTreatment also indexes Thickness, PlyCount, and Density by element ID.
% Ensure these fields exist and are sized to maxElementID.
if ~isfield(FEModelData.Elements(1), 'Density') || isempty(FEModelData.Elements(1).Density)
    FEModelData.Elements(1).Density = zeros(maxElementID, 1);
end
if length(FEModelData.Elements(1).Density) < maxElementID
    FEModelData.Elements(1).Density(end+1:maxElementID) = 0;
end
if ~isfield(FEModelData.Elements(1), 'Thickness') || isempty(FEModelData.Elements(1).Thickness)
    FEModelData.Elements(1).Thickness = zeros(maxElementID, 1);
end
if length(FEModelData.Elements(1).Thickness) < maxElementID
    FEModelData.Elements(1).Thickness(end+1:maxElementID) = 0;
end
if ~isfield(FEModelData.Elements(1), 'PlyCount') || isempty(FEModelData.Elements(1).PlyCount)
    FEModelData.Elements(1).PlyCount = zeros(maxElementID, 1);
end
if length(FEModelData.Elements(1).PlyCount) < maxElementID
    FEModelData.Elements(1).PlyCount(end+1:maxElementID) = 0;
end

nAssigned = sum(~strcmp(elements_composite, 'NotDefined'));
fprintf('  Assigned composite IDs to %d / %d elements (max ID = %d)\n', ...
    nAssigned, length(FEModelData.Elements(1).ID), maxElementID);

%% ======================== STEP 3: PROCESS MESH =========================
fprintf('\n=== Step 3: Processing mesh ===\n');
log2 = @(msg) fprintf('  [MESH] %s\n', msg);
[elements, FEModelData] = meshTreatment(FEModelData);

fprintf('  Elements after treatment: %d\n', size(elements.ConnectivityList, 1));

%% ======================== STEP 4: DEFINE IMPACTOR ======================
fprintf('\n=== Step 4: Defining impactor ===\n');

% Compute crush plane normal vector from angles
nx = sin(theta) * cos(phi);
ny = sin(theta) * sin(phi);
nz = cos(theta);
crushPlaneNormalVec = [nx, ny, nz];
fprintf('  Crush normal: [%.3f, %.3f, %.3f]\n', crushPlaneNormalVec);

% Get bounding box of model
nodesCoords = FEModelData.Nodes.Coordinates;
maxX = max(nodesCoords(:,1)); minX = min(nodesCoords(:,1));
maxY = max(nodesCoords(:,2)); minY = min(nodesCoords(:,2));
maxZ = max(nodesCoords(:,3)); minZ = min(nodesCoords(:,3));

fprintf('  Model bounds: X=[%.1f, %.1f], Y=[%.1f, %.1f], Z=[%.1f, %.1f]\n', ...
    minX, maxX, minY, maxY, minZ, maxZ);

% Build bounding box vertices (same as impactorDefinition.m)
tolBB = 5;
bbVerts = [
    minX-tolBB, minY-tolBB, minZ-tolBB;
    minX-tolBB, minY-tolBB, maxZ+tolBB;
    minX-tolBB, maxY+tolBB, minZ-tolBB;
    minX-tolBB, maxY+tolBB, maxZ+tolBB;
    maxX+tolBB, minY-tolBB, minZ-tolBB;
    maxX+tolBB, minY-tolBB, maxZ+tolBB;
    maxX+tolBB, maxY+tolBB, minZ-tolBB;
    maxX+tolBB, maxY+tolBB, maxZ+tolBB];

% Reference point P0: first bounding box vertex (default behaviour)
% For axial crush along Z, this starts from the min-Z end
P0 = bbVerts(1,:);
fprintf('  P0 (start point): [%.1f, %.1f, %.1f]\n', P0);

% Build impactor quadrilateral (same logic as impactorDefinition.m)
% The impactor is a large plane that slices through the model.
% For axial crush (normal=[0,0,1]), it's a horizontal plane.
%
% We need to compute the 4 corners of a plane that:
%   - Has normal = crushPlaneNormalVec
%   - Passes through P0
%   - Is large enough to cover the entire model bounding box

% Build orthonormal basis in the plane
n = crushPlaneNormalVec / norm(crushPlaneNormalVec);

% Find two vectors orthogonal to n
if abs(n(1)) < 0.9
    v_temp = [1, 0, 0];
else
    v_temp = [0, 1, 0];
end
v = cross(n, v_temp);
v = v / norm(v);
w = cross(n, v);
w = w / norm(w);

% Project bounding box vertices onto the plane's local axes
projections_v = (bbVerts - P0) * v';
projections_w = (bbVerts - P0) * w';

% Impactor size: span all projections with margin
margin = 50;
v_min = min(projections_v) - margin;
v_max = max(projections_v) + margin;
w_min = min(projections_w) - margin;
w_max = max(projections_w) + margin;

% 4 corner nodes of impactor plane
impactorNodes = [
    P0 + v_min * v + w_min * w;
    P0 + v_max * v + w_min * w;
    P0 + v_max * v + w_max * w;
    P0 + v_min * v + w_max * w];

% Impactor struct
impactor.NodesCoordinates = impactorNodes;
impactor.ConnectivityList = [1 2 4; 2 3 4];  % two triangles
impactor.NormalVec = crushPlaneNormalVec;

fprintf('  Impactor defined (%.0f × %.0f mm plane)\n', ...
    v_max - v_min, w_max - w_min);

%% ======================== STEP 5: RUN PARTITIONING =====================
fprintf('\n=== Step 5: Partitioning ===\n');

% Build input struct (minimum fields needed by partitioning.m)
partInput.PartitionsOffset = partitionOffset;
partInput.crushDirectionDef = crushDirectionDef;
partInput.directionVec = directionVec;

fprintf('  Partition offset: %.1f mm\n', partitionOffset);
fprintf('  Running partitioning...\n');

layers = partitioning(FEModelData, elements, impactor, partInput);

numLayers = length(layers);
fprintf('  Found %d layers\n', numLayers);

%% ======================== STEP 6: INSPECT LAYERS =======================
fprintf('\n=== Step 6: Layer inspection ===\n');

% Print summary of each layer
fprintf('\n  Layer | Nodes | Edges | Max Degree | Has Rib?\n');
fprintf('  ------|-------|-------|------------|--------\n');

ribLayers = [];
for i = 1:numLayers
    nc = size(layers(i).NodesCoordinates, 1);
    ne = size(layers(i).ConnectivityList, 1);
    
    if nc == 0 || ne == 0
        fprintf('  %5d | %5d | %5d |     -      |    -\n', i, nc, ne);
        continue;
    end
    
    % Check for branching (rib detection)
    G = graph(layers(i).ConnectivityList(:,1), layers(i).ConnectivityList(:,2));
    deg = degree(G);
    maxDeg = max(deg);
    hasRib = maxDeg > 2;
    
    if hasRib
        ribLayers = [ribLayers, i];
    end
    
    fprintf('  %5d | %5d | %5d |     %d      |    %s\n', ...
        i, nc, ne, maxDeg, string(hasRib));
end

fprintf('\n  Layers with ribs: %s\n', mat2str(ribLayers));

%% ======================== STEP 7: PLOT CANDIDATE LAYERS ================
if ~isempty(ribLayers)
    nPlot = min(length(ribLayers), 6);
    figure('Name', 'Ribbed Layers', 'Position', [100 100 1200 600]);
    
    for p = 1:nPlot
        idx = ribLayers(p);
        subplot(2, 3, p);
        
        coords = layers(idx).NodesCoordinates;
        conn = layers(idx).ConnectivityList;
        
        % Plot edges
        for e = 1:size(conn, 1)
            n1 = conn(e, 1); n2 = conn(e, 2);
            plot3([coords(n1,1), coords(n2,1)], ...
                  [coords(n1,2), coords(n2,2)], ...
                  [coords(n1,3), coords(n2,3)], 'b-', 'LineWidth', 0.5);
            hold on;
        end
        
        % Highlight junction nodes (degree > 2)
        G = graph(conn(:,1), conn(:,2));
        deg = degree(G);
        junctions = find(deg > 2);
        if ~isempty(junctions)
            scatter3(coords(junctions,1), coords(junctions,2), ...
                     coords(junctions,3), 50, 'r', 'filled');
        end
        
        axis equal; grid on; view(2);
        title(sprintf('Layer %d (%d nodes, %d junctions)', ...
            idx, size(coords,1), length(junctions)));
    end
    sgtitle('Ribbed cross-sections (red = junction nodes)');
end

%% ======================== STEP 8: SELECT AND SAVE ======================
selectedLayers = resolve_layer_selection(targetLayerSelection, numLayers, ribLayers);

fprintf('\n  Selected layers to save: %s\n', mat2str(selectedLayers));
if isempty(selectedLayers)
    error('No layers selected for saving.');
end

if false
% Check topology
G = graph(connectivity(:,1), connectivity(:,2));
deg = degree(G);
fprintf('  Max degree: %d\n', max(deg));
if max(deg) > 2
    junctions = find(deg > 2);
    fprintf('  Junction nodes: %d (at indices %s)\n', ...
        length(junctions), mat2str(junctions'));
    fprintf('  ✓ This layer has ribs!\n');
else
    fprintf('  This layer has NO ribs (simple loop or chain)\n');
end

save(outputFileName, 'nodesCoords', 'connectivity');
fprintf('\n  Saved to: %s\n', outputFileName);

% Also save the MATLAB curvature/force data for validation
fprintf('\n=== Computing MATLAB reference (curvatureSpline) ===\n');
% Use a constant flat_transition for the simplified case
flat_transition = 1221.947188466746;  % from DataCurvatureCarbon.xlsx
flatTransition = flat_transition * ones(size(nodesCoords,1), 1);

% We can call curvatureSpline directly if layersProp is on the path.
% Since curvatureSpline is a local function inside layersProp.m,
% we'll replicate the call approach or note the force for manual comparison.
fprintf('  NOTE: To get the MATLAB reference force, run:\n');
fprintf('    RoC = curvatureSpline(nodesCoords, connectivity, flatTransition);\n');
fprintf('  But curvatureSpline is a local function in layersProp.m.\n');
fprintf('  You may need to copy it out or run layersProp on this layer.\n');
fprintf('\n  Alternatively, just run ribAnalysis.py on %s\n', outputFileName);
fprintf('  and compare the topology detection / RoC values.\n');

end

for k = 1:length(selectedLayers)
    layerIdx = selectedLayers(k);
    nodesCoords = layers(layerIdx).NodesCoordinates;
    connectivity = layers(layerIdx).ConnectivityList;
    outputPath = build_output_filename(outputFileName, layerIdx, length(selectedLayers));

    fprintf('\n=== Saving layer %d (%d of %d) ===\n', layerIdx, k, length(selectedLayers));
    fprintf('  Nodes: %d\n', size(nodesCoords, 1));
    fprintf('  Edges: %d\n', size(connectivity, 1));

    G = graph(connectivity(:,1), connectivity(:,2));
    deg = degree(G);
    fprintf('  Max degree: %d\n', max(deg));
    if max(deg) > 2
        junctions = find(deg > 2);
        fprintf('  Junction nodes: %d (at indices %s)\n', ...
            length(junctions), mat2str(junctions'));
        fprintf('  This layer has ribs.\n');
    else
        fprintf('  This layer has NO ribs (simple loop or chain)\n');
    end

    save(outputPath, 'nodesCoords', 'connectivity');
    fprintf('\n  Saved to: %s\n', outputPath);

    fprintf('\n=== Computing MATLAB reference (curvatureSpline) ===\n');
    flat_transition = 1221.947188466746;  % from DataCurvatureCarbon.xlsx
    flatTransition = flat_transition * ones(size(nodesCoords,1), 1); %#ok<NASGU>

    fprintf('  NOTE: To get the MATLAB reference force, run:\n');
    fprintf('    RoC = curvatureSpline(nodesCoords, connectivity, flatTransition);\n');
    fprintf('  But curvatureSpline is a local function in layersProp.m.\n');
    fprintf('  You may need to copy it out or run layersProp on this layer.\n');
    fprintf('\n  Alternatively, just run ribAnalysis.py on %s\n', outputPath);
    fprintf('  and compare the topology detection / RoC values.\n');
end

fprintf('\n=== Done ===\n');


function selectedLayers = resolve_layer_selection(selection, numLayers, ribLayers)
if isnumeric(selection)
    if isscalar(selection) && selection == 0
        fprintf(['\n  Enter layer selection to save ', ...
                 '(examples: 12, [3 7 9], all, ribs): ']);
        raw = strtrim(input('', 's'));
        if isempty(raw)
            error('No layer selection provided.');
        end

        if strcmpi(raw, 'all')
            selectedLayers = 1:numLayers;
        elseif strcmpi(raw, 'ribs')
            selectedLayers = ribLayers;
        else
            parsed = str2num(raw); %#ok<ST2NM>
            if isempty(parsed)
                error('Could not parse layer selection: %s', raw);
            end
            selectedLayers = parsed;
        end
    else
        selectedLayers = selection;
    end
elseif ischar(selection) || (isstring(selection) && isscalar(selection))
    selection = char(string(selection));
    if strcmpi(selection, 'all')
        selectedLayers = 1:numLayers;
    elseif strcmpi(selection, 'ribs')
        selectedLayers = ribLayers;
    else
        error('Unsupported targetLayerSelection value: %s', selection);
    end
else
    error('Unsupported targetLayerSelection type.');
end

selectedLayers = unique(selectedLayers(:)', 'stable');
if isempty(selectedLayers)
    error('Layer selection is empty.');
end
if any(selectedLayers < 1 | selectedLayers > numLayers | mod(selectedLayers, 1) ~= 0)
    error('Invalid layer selection: %s', mat2str(selectedLayers));
end
end


function outputPath = build_output_filename(baseName, layerIdx, numSelected)
[folder, stem, ext] = fileparts(baseName);
if isempty(ext)
    ext = '.mat';
end

if numSelected == 1
    outputPath = fullfile(folder, [stem ext]);
else
    outputPath = fullfile(folder, sprintf('%s_layer_%03d%s', stem, layerIdx, ext));
end
end
