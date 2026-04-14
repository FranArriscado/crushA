function result = validate_rib_optimizer_frozen_app(sourcePartitionMat, optimizerMat, newRunDir, thicknessOverride)
% =========================================================================
%           validate_rib_optimizer_frozen_app.m
%
% Validate an optimizer output against a "frozen metadata" replay of the
% last CrushAnalytica app run.
%
% Idea:
%   - Use the saved NewRun layer from the real app as the source of nodal
%     metadata (thickness, flat transition, composite IDs, crush-stress
%     baseline).
%   - Keep those non-geometry assignments frozen.
%   - Rebuild the optimized 2D section from the optimizer .mat.
%   - Recompute only the geometry-dependent pieces with app-like logic:
%       * node lengths from the optimized branched connectivity
%       * curvature / RoC using the app-style curvatureSpline logic
%       * curvature-dependent part of crush stress from materialProp
%
% it does NOT try to mirror the Python forward pass. It aims to answer:
% "If this optimized 2D section were replayed through the CrushAnalytica
% logic while keeping the original app-side nodal assignment frozen, what
% force would MATLAB predict?"
%
% Supported optimizer outputs:
%   - Ribbed output:    wall_nodesCoords + rib_nodesCoords + junctions
%   - No-rib output:    nodesCoords + connectivity
%
% Francisco Arriscado / FEUP / 2026
% =========================================================================

clearvars -except sourcePartitionMat optimizerMat newRunDir thicknessOverride;
clc;

thisDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(thisDir);
addpath(rootDir);
minRocClamp = 15.0;

if nargin >= 1 && (ischar(sourcePartitionMat) || (isstring(sourcePartitionMat) && isscalar(sourcePartitionMat))) && ...
        (strcmpi(char(sourcePartitionMat), 'select') || strcmpi(char(sourcePartitionMat), 'ui'))
    [sourcePartitionMat, optimizerMat, newRunDir, thicknessOverride] = ...
        interactive_pick_validation_inputs(rootDir, thisDir);
end

if ~exist('newRunDir', 'var') || isempty(newRunDir)
    newRunDir = fullfile(thisDir, 'NewRun');
end

if ~exist('thicknessOverride', 'var')
    thicknessOverride = [];
end

if ~exist('sourcePartitionMat', 'var') || isempty(sourcePartitionMat)
    sourcePartitionMat = fullfile(thisDir, 'data', 'partition_data_layer2_z-132.4.mat');
end

if ~exist('optimizerMat', 'var') || isempty(optimizerMat)
    optimizerMat = auto_pick_optimizer_mat(rootDir, sourcePartitionMat);
end

assert(isfolder(newRunDir), 'NewRun directory not found: %s', newRunDir);
assert(isfile(sourcePartitionMat), 'Source partition MAT not found: %s', sourcePartitionMat);
assert(isfile(optimizerMat), 'Optimizer MAT not found: %s', optimizerMat);

runFiles.layers       = latest_matching_file(newRunDir, 'PartitionLayers_*.mat');
runFiles.materialProp = latest_matching_file(newRunDir, 'MaterialProp_*.mat');
runFiles.elements     = latest_matching_file(newRunDir, 'Elements_*.mat');
runFiles.femodel      = latest_matching_file(newRunDir, 'FEModelData_*.mat');
runFiles.impactor     = latest_matching_file(newRunDir, 'Impactor_*.mat');

layers = load_named_or_first(runFiles.layers, 'layer');
materialProp = load_named_or_first(runFiles.materialProp, 'materialprop');

sourceData = load(sourcePartitionMat);
assert(isfield(sourceData, 'nodesCoords') && isfield(sourceData, 'connectivity'), ...
    'Expected nodesCoords/connectivity in %s', sourcePartitionMat);
sourceNodes = double(sourceData.nodesCoords);
sourceConn = double(sourceData.connectivity);
if isfield(sourceData, 'thickness')
    sourceThickness = double(sourceData.thickness(:));
else
    sourceThickness = [];
end

opt = load(optimizerMat);
optGeometryMode = detect_optimizer_geometry(opt);
if strcmp(optGeometryMode, 'rib')
    requiredOpt = {'wall_nodesCoords', 'wall_connectivity', 'rib_nodesCoords', ...
        'junction_wall_indices', 'forceFinal'};
elseif strcmp(optGeometryMode, 'no_rib')
    requiredOpt = {'nodesCoords', 'connectivity', 'forceFinal'};
else
    error('Unsupported optimizer geometry mode: %s', optGeometryMode);
end
for k = 1:numel(requiredOpt)
    assert(isfield(opt, requiredOpt{k}), 'Missing %s in %s', requiredOpt{k}, optimizerMat);
end

layerIdx = find_matching_source_layer(sourceNodes, sourceConn, sourceThickness, layers, sourcePartitionMat);
srcLayer = layers(layerIdx);

srcTopo = extract_section_topology(sourceNodes, sourceConn);

[srcCurvOnlyStress, srcCurvSourceCount] = curvature_only_stress_vector( ...
    double(srcLayer.Curvature(:)), srcLayer.NodesCompositeIDs, materialProp);

srcCrushStress = double(srcLayer.CrushStress(:));
srcFrozenMultiplier = ones(size(srcCrushStress));
maskCurv = abs(srcCurvOnlyStress) > 1e-12;
srcFrozenMultiplier(maskCurv) = srcCrushStress(maskCurv) ./ srcCurvOnlyStress(maskCurv);
srcFrozenMultiplier(~maskCurv & abs(srcCrushStress) <= 1e-12) = 1;
srcFrozenMultiplier(~maskCurv & abs(srcCrushStress) > 1e-12) = NaN;
if any(~isfinite(srcFrozenMultiplier))
    badCount = nnz(~isfinite(srcFrozenMultiplier));
    warning('validate_rib_optimizer_frozen_app:frozenMultiplierFallback', ...
        ['%d source nodes had undefined frozen multipliers. ' ...
         'Falling back to 1.0 for those nodes.'], badCount);
    srcFrozenMultiplier(~isfinite(srcFrozenMultiplier)) = 1;
end

mapped = map_frozen_source_metadata(srcLayer, srcTopo, srcFrozenMultiplier, opt, optGeometryMode);
[optFullNodes, optFullConn, optMasks] = build_full_optimized_topology(opt, optGeometryMode);
[mapped, thicknessMode] = apply_optimizer_role_thicknesses_if_available(mapped, opt, optMasks, sourceThickness, thicknessOverride);

optNodeLength = node_lengths_from_connectivity(optFullNodes, optFullConn);
optRoC = curvature_spline_like_app(optFullNodes, optFullConn, mapped.flatTransition);
[optCurvOnlyStress, optCurvSourceCount] = curvature_only_stress_vector( ...
    optRoC, mapped.nodesCompositeIDs, materialProp);
optCrushStress = mapped.frozenMultiplier .* optCurvOnlyStress;
optArea = optNodeLength .* mapped.thickness;
optForceNode = optCrushStress .* optArea;
optForceTotal = sum(optForceNode);
optWallForce = sum(optForceNode(optMasks.wallMask));
optRibForce = sum(optForceNode(optMasks.ribMask));

srcReplayNodeLength = node_lengths_from_connectivity(double(srcLayer.NodesCoordinates), double(srcLayer.ConnectivityList));
srcReplayArea = srcReplayNodeLength .* double(srcLayer.Thickness(:));
srcReplayCrushStress = srcFrozenMultiplier .* srcCurvOnlyStress;
srcReplayForceNode = srcReplayCrushStress .* srcReplayArea;
srcReplayForceTotal = sum(srcReplayForceNode);

sourceSavedForce = double(srcLayer.ForceTotal);
pythonForce = double(opt.forceFinal);
pythonWallForce = get_scalar_field(opt, 'wallForceFinal', NaN);
pythonRibForce = get_scalar_field(opt, 'ribForceFinal', NaN);
if strcmp(optGeometryMode, 'no_rib')
    pythonWallForce = pythonForce;
    pythonRibForce = 0;
end

srcReplayDiff = srcReplayForceTotal - sourceSavedForce;
optDiff = optForceTotal - pythonForce;
optDiffPct = 100 * optDiff / max(abs(pythonForce), eps);
if ~isnan(pythonWallForce)
    optWallDiffPct = 100 * (optWallForce - pythonWallForce) / max(abs(pythonWallForce), eps);
else
    optWallDiffPct = NaN;
end
if ~isnan(pythonRibForce) && abs(pythonRibForce) > 1e-12
    optRibDiffPct = 100 * (optRibForce - pythonRibForce) / pythonRibForce;
else
    optRibDiffPct = NaN;
end

fprintf('\n===============================================================\n');
fprintf('  FROZEN APP REPLAY VALIDATION\n');
fprintf('===============================================================\n');
fprintf('  Source partition : %s\n', sourcePartitionMat);
fprintf('  Optimizer result : %s\n', optimizerMat);
fprintf('  NewRun folder    : %s\n', newRunDir);
fprintf('  Matched app layer: %d\n', layerIdx);
fprintf('  NewRun layers    : %s\n', runFiles.layers);
fprintf('  MaterialProp     : %s\n', runFiles.materialProp);
fprintf('  Geometry mode    : %s\n', optGeometryMode);
fprintf('===============================================================\n');
fprintf('  Source nodes/edges     : %d / %d\n', size(sourceNodes,1), size(sourceConn,1));
if strcmp(optGeometryMode, 'rib')
    fprintf('  Optimized wall nodes   : %d\n', size(opt.wall_nodesCoords,1));
    fprintf('  Optimized rib total    : %d\n', size(opt.rib_nodesCoords,1));
else
    fprintf('  Optimized loop nodes   : %d\n', size(opt.nodesCoords,1));
end
fprintf('  Optimized full nodes   : %d\n', size(optFullNodes,1));
fprintf('  Curvature source hits  : source=%d/%d  optimized=%d/%d\n', ...
    srcCurvSourceCount, numel(srcCrushStress), optCurvSourceCount, numel(optCrushStress));
fprintf('  Thickness mode         : %s\n', thicknessMode);
fprintf('  App curvature clamp min: %.1f mm\n', minRocClamp);
fprintf('===============================================================\n');

fprintf('\n  ORIGINAL GEOMETRY SELF-CHECK\n');
fprintf('  -------------------------------------------------------------\n');
fprintf('  Saved app force        : %12.4f N\n', sourceSavedForce);
fprintf('  Frozen replay force    : %12.4f N\n', srcReplayForceTotal);
fprintf('  Difference             : %+12.4f N\n', srcReplayDiff);

fprintf('\n  OPTIMIZED GEOMETRY COMPARISON\n');
fprintf('  -------------------------------------------------------------\n');
fprintf('  %-26s %12s %12s %10s\n', 'Component', 'Frozen App', 'Python', 'Diff %');
fprintf('  %s\n', repmat('-', 1, 66));
if strcmp(optGeometryMode, 'rib')
    fprintf('  %-26s %12.4f %12.4f %+9.4f%%\n', ...
        'Wall force [N]', optWallForce, pythonWallForce, optWallDiffPct);
    fprintf('  %-26s %12.4f %12.4f %+9.4f%%\n', ...
        'Rib force [N]', optRibForce, pythonRibForce, optRibDiffPct);
    fprintf('  %s\n', repmat('-', 1, 66));
    fprintf('  %-26s %12.4f %12.4f %+9.4f%%\n', ...
        'Total force [N]', optForceTotal, pythonForce, optDiffPct);
else
    fprintf('  %-26s %12.4f %12.4f %+9.4f%%\n', ...
        'Loop / total force [N]', optForceTotal, pythonForce, optDiffPct);
end
fprintf('===============================================================\n');

csvPath = fullfile(newRunDir, 'validation_rib_frozen_log.csv');
append_validation_row(csvPath, sourcePartitionMat, optimizerMat, layerIdx, ...
    sourceSavedForce, srcReplayForceTotal, pythonForce, optForceTotal, ...
    optWallForce, pythonWallForce, optRibForce, pythonRibForce);
fprintf('\nResult appended to: %s\n', csvPath);

[~, optBaseName] = fileparts(optimizerMat);
diagDir = fullfile(newRunDir, 'Diag');
if ~isfolder(diagDir)
    mkdir(diagDir);
end
diagPath = fullfile(diagDir, ['diag_frozen_' optBaseName '.mat']);
save(diagPath, ...
    'layerIdx', 'sourcePartitionMat', 'optimizerMat', 'runFiles', ...
    'srcTopo', 'sourceSavedForce', 'srcReplayForceTotal', 'srcReplayForceNode', ...
    'srcFrozenMultiplier', 'srcCurvOnlyStress', ...
    'mapped', 'optFullNodes', 'optFullConn', 'optMasks', ...
    'optNodeLength', 'optRoC', 'optCurvOnlyStress', 'optCrushStress', ...
    'optArea', 'optForceNode', 'optForceTotal', 'optWallForce', 'optRibForce', ...
    'pythonForce', 'pythonWallForce', 'pythonRibForce');
fprintf('Diagnostics saved to: %s\n', diagPath);

result = struct();
result.layerIdx = layerIdx;
result.sourcePartitionMat = sourcePartitionMat;
result.optimizerMat = optimizerMat;
result.newRunDir = newRunDir;
result.geometryMode = optGeometryMode;
result.sourceSavedForce = sourceSavedForce;
result.sourceReplayForce = srcReplayForceTotal;
result.pythonForce = pythonForce;
result.frozenAppForce = optForceTotal;
result.frozenAppWallForce = optWallForce;
result.frozenAppRibForce = optRibForce;
result.totalDiffN = optDiff;
result.totalDiffPct = optDiffPct;
result.thicknessMode = thicknessMode;
result.csvPath = csvPath;
result.diagPath = diagPath;

end

function [sourcePartitionMat, optimizerMat, newRunDir, thicknessOverride] = ...
    interactive_pick_validation_inputs(rootDir, thisDir)

defaultSourceDir = first_existing_dir({ ...
    fullfile(rootDir, 'data'), ...
    fullfile(thisDir, 'data'), ...
    rootDir});

[srcName, srcDir] = uigetfile( ...
    {'*.mat','MAT-files (*.mat)'}, ...
    'Select source partition MAT', ...
    fullfile(defaultSourceDir, '*.mat'));
if isequal(srcName, 0)
    error('Validation cancelled while selecting the source partition MAT.');
end
sourcePartitionMat = fullfile(srcDir, srcName);

defaultOptDir = first_existing_dir({ ...
    fullfile(rootDir, 'results', 'Rib'), ...
    fullfile(rootDir, 'results', 'NoRib'), ...
    fullfile(rootDir, 'results'), ...
    rootDir});
[optName, optDir] = uigetfile( ...
    {'*.mat','MAT-files (*.mat)'}, ...
    'Select optimizer result MAT', ...
    fullfile(defaultOptDir, '*.mat'));
if isequal(optName, 0)
    error('Validation cancelled while selecting the optimizer result MAT.');
end
optimizerMat = fullfile(optDir, optName);

defaultRunDir = first_existing_dir({ ...
    fullfile(thisDir, 'NewRun'), ...
    fullfile(rootDir, 'Rib', 'NewRun'), ...
    rootDir});

if isfolder(defaultRunDir)
    pickRunDir = questdlg( ...
        sprintf('Use default NewRun folder?\n\n%s', defaultRunDir), ...
        'Validation run folder', ...
        'Use default', 'Choose folder', 'Use default');
    if isempty(pickRunDir)
        error('Validation cancelled while selecting the run folder.');
    elseif strcmp(pickRunDir, 'Choose folder')
        newRunDir = uigetdir(defaultRunDir, 'Select folder containing PartitionLayers/MaterialProp');
        if isequal(newRunDir, 0)
            error('Validation cancelled while selecting the run folder.');
        end
    else
        newRunDir = defaultRunDir;
    end
else
    newRunDir = uigetdir(rootDir, 'Select folder containing PartitionLayers/MaterialProp');
    if isequal(newRunDir, 0)
        error('Validation cancelled while selecting the run folder.');
    end
end

validate_run_folder_snapshot(newRunDir);

thicknessOverride = [];
optPreview = load(optimizerMat);
isNoRib = isfield(optPreview, 'nodesCoords') && isfield(optPreview, 'connectivity') && ...
    ~(isfield(optPreview, 'wall_nodesCoords') && isfield(optPreview, 'rib_nodesCoords'));
hasScalarThickness = (isfield(optPreview, 'thickness') && ~isempty(optPreview.thickness)) || ...
    (isfield(optPreview, 'wallThickness') && ~isempty(optPreview.wallThickness));

if isNoRib && ~hasScalarThickness
    useOverride = questdlg( ...
        ['The selected no-rib result does not store scalar thickness in the .mat.' newline ...
         'Do you want to enter a thickness override now?'], ...
        'No-rib thickness', ...
        'Enter thickness', 'Skip', 'Enter thickness');
    if isempty(useOverride)
        error('Validation cancelled while choosing the no-rib thickness handling.');
    elseif strcmp(useOverride, 'Enter thickness')
        answer = inputdlg( ...
            {'Scalar thickness override for this no-rib result [mm]:'}, ...
            'Thickness override', ...
            [1 60], ...
            {''});
        if isempty(answer)
            error('Validation cancelled while entering the thickness override.');
        end
        thicknessOverride = str2double(strtrim(answer{1}));
        assert(isfinite(thicknessOverride) && thicknessOverride > 0, ...
            'Thickness override must be a positive number.');
    end
end

fprintf('\nPicker selection summary\n');
fprintf('  Source partition : %s\n', sourcePartitionMat);
fprintf('  Optimizer result : %s\n', optimizerMat);
fprintf('  Run folder       : %s\n', newRunDir);
if isempty(thicknessOverride)
    fprintf('  Thickness override: <none>\n');
else
    fprintf('  Thickness override: %.6f mm\n', thicknessOverride);
end
end

function folder = first_existing_dir(candidates)
folder = '';
for i = 1:numel(candidates)
    if isfolder(candidates{i})
        folder = candidates{i};
        return;
    end
end
folder = pwd;
end

function validate_run_folder_snapshot(folder)
requiredPatterns = {'PartitionLayers_*.mat', 'MaterialProp_*.mat'};
missing = cell(0,1);
for i = 1:numel(requiredPatterns)
    if isempty(dir(fullfile(folder, requiredPatterns{i})))
        missing{end+1,1} = requiredPatterns{i}; %#ok<AGROW>
    end
end
assert(isempty(missing), ...
    'Selected run folder is missing required files: %s', strjoin(missing, ', '));
end

function geometryMode = detect_optimizer_geometry(opt)
if isfield(opt, 'wall_nodesCoords') && isfield(opt, 'rib_nodesCoords') && ...
        isfield(opt, 'junction_wall_indices')
    geometryMode = 'rib';
elseif isfield(opt, 'nodesCoords') && isfield(opt, 'connectivity')
    geometryMode = 'no_rib';
else
    error(['Could not determine optimizer geometry mode. Expected either ' ...
        'rib fields (wall_nodesCoords/rib_nodesCoords/junction_wall_indices) ' ...
        'or no-rib fields (nodesCoords/connectivity).']);
end
end

function [mapped, thicknessMode] = apply_optimizer_role_thicknesses_if_available(mapped, opt, optMasks, sourceThickness, thicknessOverride)
thicknessMode = 'source_mapped';

if ~any(optMasks.ribMask)
    if ~isempty(thicknessOverride)
        mapped.thickness(:) = double(thicknessOverride);
        thicknessMode = 'user_override';
    elseif isfield(opt, 'thickness') && ~isempty(opt.thickness)
        mapped.thickness(:) = double(opt.thickness);
        thicknessMode = 'optimizer_scalar';
    elseif isfield(opt, 'wallThickness') && ~isempty(opt.wallThickness)
        mapped.thickness(:) = double(opt.wallThickness);
        thicknessMode = 'optimizer_scalar';
    elseif ~isempty(sourceThickness)
        uniqueThickness = unique(round(double(sourceThickness(:)), 8));
        if numel(uniqueThickness) == 1
            mapped.thickness(:) = uniqueThickness(1);
            thicknessMode = 'source_partition_scalar';
        end
    end
    if strcmp(thicknessMode, 'source_mapped')
        warning('validate_rib_optimizer_frozen_app:noRibThicknessFallback', ...
            ['No scalar thickness was found in the no-rib optimizer result. ' ...
             'Using frozen source-mapped app thickness instead.']);
    end
    return;
end

hasRoleThickness = isfield(opt, 'wallThickness') && isfield(opt, 'ribThickness') && ...
    isfield(opt, 'junctionThicknesses') && numel(opt.junctionThicknesses) >= 2;
if ~hasRoleThickness
    return;
end

wallThickness = double(opt.wallThickness);
ribThickness = double(opt.ribThickness);
junctionThicknesses = double(opt.junctionThicknesses(:));

thicknessVec = mapped.thickness(:);
thicknessVec(optMasks.wallMask) = wallThickness;
thicknessVec(optMasks.ribMask) = ribThickness;

junctionIdx = find(optMasks.junctionMask);
if numel(junctionIdx) == 2
    thicknessVec(junctionIdx(1)) = junctionThicknesses(1);
    thicknessVec(junctionIdx(2)) = junctionThicknesses(2);
elseif ~isempty(junctionIdx)
    thicknessVec(junctionIdx) = junctionThicknesses(1);
end

mapped.thickness = thicknessVec;
mapped.optimizerRoleThickness = struct( ...
    'wallThickness', wallThickness, ...
    'ribThickness', ribThickness, ...
    'junctionThicknesses', junctionThicknesses(1:min(2, numel(junctionThicknesses))));
thicknessMode = 'optimizer_roles';
end

function optimizerMat = auto_pick_optimizer_mat(rootDir, sourcePartitionMat)
[~, srcName] = fileparts(sourcePartitionMat);
token = regexprep(srcName, '^partition_data_', '');
searchDirs = { ...
    fullfile(rootDir, 'results', 'Rib'), ...
    fullfile(rootDir, 'results', 'Rib'), ...
    fullfile(rootDir, 'results', 'Partions Processing examples')};

candidates = struct('folder', {}, 'name', {}, 'datenum', {});
for i = 1:numel(searchDirs)
    if ~isfolder(searchDirs{i})
        continue;
    end
    d = dir(fullfile(searchDirs{i}, ['*' token '*.mat']));
    for k = 1:numel(d)
        if startsWith(d(k).name, 'partition_data_', 'IgnoreCase', true)
            continue;
        end
        candidates(end+1).folder = searchDirs{i}; %#ok<AGROW>
        candidates(end).name = d(k).name;
        candidates(end).datenum = d(k).datenum;
    end
end

assert(~isempty(candidates), 'No optimizer MAT found automatically for token "%s".', token);
[~, idx] = max([candidates.datenum]);
optimizerMat = fullfile(candidates(idx).folder, candidates(idx).name);
end

function pathOut = latest_matching_file(folder, pattern)
d = dir(fullfile(folder, pattern));
assert(~isempty(d), 'No files matching %s in %s', pattern, folder);
[~, idx] = max([d.datenum]);
pathOut = fullfile(folder, d(idx).name);
end

function value = load_named_or_first(matPath, preferredName)
d = load(matPath);
if nargin >= 2 && isfield(d, preferredName)
    value = d.(preferredName);
    return;
end
f = fieldnames(d);
assert(~isempty(f), 'No variables found in %s', matPath);
value = d.(f{1});
end

function scalar = get_scalar_field(s, fieldName, defaultValue)
if isfield(s, fieldName)
    scalar = double(s.(fieldName));
else
    scalar = defaultValue;
end
end

function layerIdx = find_matching_source_layer(sourceNodes, sourceConn, sourceThickness, layers, sourcePartitionMat)
tol = 1e-8;
bestIdx = NaN;
bestScore = inf;

for i = 1:numel(layers)
    if isempty(layers(i).NodesCoordinates) || isempty(layers(i).ConnectivityList)
        continue;
    end
    layerNodes = double(layers(i).NodesCoordinates);
    layerConn = double(layers(i).ConnectivityList);
    if size(layerNodes, 1) ~= size(sourceNodes, 1) || size(layerConn, 1) ~= size(sourceConn, 1)
        continue;
    end

    if max(abs(layerNodes(:) - sourceNodes(:))) <= tol && isequal(layerConn, sourceConn)
        if isempty(sourceThickness)
            layerIdx = i;
            return;
        end
        if isfield(layers(i), 'Thickness') && numel(layers(i).Thickness) == numel(sourceThickness)
            if max(abs(double(layers(i).Thickness(:)) - sourceThickness(:))) <= 1e-6
                layerIdx = i;
                return;
            end
        else
            layerIdx = i;
            return;
        end
    end

    if isequal(layerConn, sourceConn)
        score = max(vecnorm(layerNodes - sourceNodes, 2, 2));
        if score < bestScore
            bestScore = score;
            bestIdx = i;
        end
    end
end

if ~isnan(bestIdx) && bestScore <= 1e-3
    warning('validate_rib_optimizer_frozen_app:approxLayerMatch', ...
        'Using approximate source-layer match at index %d (max node delta %.4g mm).', bestIdx, bestScore);
    layerIdx = bestIdx;
    return;
end

tok = regexp(sourcePartitionMat, 'partition_data_layer(\d+)_', 'tokens', 'once');
if ~isempty(tok)
    idx = str2double(tok{1});
    if idx >= 1 && idx <= numel(layers)
        warning('validate_rib_optimizer_frozen_app:fallbackLayerIndex', ...
            'Falling back to layer index parsed from filename: %d', idx);
        layerIdx = idx;
        return;
    end
end

error('Could not match %s to any saved layer in the NewRun snapshot.', sourcePartitionMat);
end

function mapped = map_frozen_source_metadata(srcLayer, srcTopo, srcFrozenMultiplier, opt, optGeometryMode)
srcCoords = double(srcLayer.NodesCoordinates);
if size(srcCoords, 2) == 2
    srcCoords(:,3) = 0;
end

if strcmp(optGeometryMode, 'rib')
    assert(strcmp(srcTopo.kind, 'rib'), ...
        'Rib optimizer output requires a ribbed source partition topology.');

    srcWallLoop = srcTopo.wallLoop(:);
    srcRibBranch = srcTopo.ribBranch(:);
    srcJunctions = srcTopo.junctionNodes(:);

    optWallNodes = double(opt.wall_nodesCoords);
    if size(optWallNodes, 2) == 2
        optWallNodes(:,3) = 0;
    end
    optRibNodes = double(opt.rib_nodesCoords);
    if size(optRibNodes, 2) == 2
        optRibNodes(:,3) = 0;
    end
    optJWallIdx = double(opt.junction_wall_indices(:));

    assert(numel(srcJunctions) == 2, 'Expected exactly 2 source junction nodes.');
    assert(numel(optJWallIdx) == 2, 'Expected exactly 2 optimizer junction indices.');

    best.score = inf;
    best.wallLoop = [];
    best.ribBranch = [];
    best.swapJunctions = false;
    best.reverseWall = false;

    for swapJunctions = [false true]
        if ~swapJunctions
            j0Node = srcJunctions(1);
            j1Node = srcJunctions(2);
            ribCandidate = srcRibBranch;
        else
            j0Node = srcJunctions(2);
            j1Node = srcJunctions(1);
            ribCandidate = flipud(srcRibBranch);
        end

        if ribCandidate(1) ~= j0Node || ribCandidate(end) ~= j1Node
            continue;
        end

        for reverseWall = [false true]
            if reverseWall
                wallCandidate = flipud(srcWallLoop);
            else
                wallCandidate = srcWallLoop;
            end

            j0Pos = find(wallCandidate == j0Node, 1);
            j1Pos = find(wallCandidate == j1Node, 1);
            if isempty(j0Pos) || isempty(j1Pos)
                continue;
            end

            wallAligned = rotate_to_index(wallCandidate, j0Pos, optJWallIdx(1));
            j1AlignedPos = find(wallAligned == j1Node, 1);

            if numel(wallAligned) == size(optWallNodes,1)
                score = abs(j1AlignedPos - optJWallIdx(2));
            else
                score = abs((j1AlignedPos - 1) / max(numel(wallAligned), 1) - ...
                            (optJWallIdx(2) - 1) / max(size(optWallNodes,1), 1));
            end

            if score < best.score
                best.score = score;
                best.wallLoop = wallAligned;
                best.ribBranch = ribCandidate;
                best.swapJunctions = swapJunctions;
                best.reverseWall = reverseWall;
            end
        end
    end

    assert(~isempty(best.wallLoop), 'Could not align source topology to optimizer ordering.');
    if best.score > 0
        warning('validate_rib_optimizer_frozen_app:topologyAlignmentApprox', ...
            'Source topology alignment is approximate (score=%.4g).', best.score);
    end

    srcWallCoords = srcCoords(best.wallLoop, :);
    srcRibCoords = srcCoords(best.ribBranch, :);

    srcWallU = closed_branch_u(srcWallCoords);
    optWallU = closed_branch_u(optWallNodes);
    srcWallNearest = nearest_closed_u_indices(srcWallU, optWallU);
    wallSourceNodes = best.wallLoop(srcWallNearest);

    srcRibU = open_branch_u(srcRibCoords);
    optRibU = open_branch_u(optRibNodes);
    srcRibNearest = nearest_open_u_indices(srcRibU, optRibU);
    ribSourceNodesFull = best.ribBranch(srcRibNearest);

    if size(optRibNodes, 1) >= 2
        ribSourceNodesInterior = ribSourceNodesFull(2:end-1);
    else
        ribSourceNodesInterior = zeros(0,1);
    end

    mapped = struct();
    mapped.thickness = [double(srcLayer.Thickness(wallSourceNodes)); ...
                        double(srcLayer.Thickness(ribSourceNodesInterior))];
    mapped.flatTransition = [double(srcLayer.FlatTransition(wallSourceNodes)); ...
                             double(srcLayer.FlatTransition(ribSourceNodesInterior))];
    mapped.frozenMultiplier = [double(srcFrozenMultiplier(wallSourceNodes)); ...
                               double(srcFrozenMultiplier(ribSourceNodesInterior))];
    mapped.nodesCompositeIDs = [srcLayer.NodesCompositeIDs(wallSourceNodes,:); ...
                                srcLayer.NodesCompositeIDs(ribSourceNodesInterior,:)];

    mapped.alignedSourceWallLoop = best.wallLoop;
    mapped.alignedSourceRibBranch = best.ribBranch;
    mapped.wallSourceNodes = wallSourceNodes;
    mapped.ribSourceNodesFull = ribSourceNodesFull;
    mapped.ribSourceNodesInterior = ribSourceNodesInterior;
    mapped.wallSourceU = srcWallU;
    mapped.wallQueryU = optWallU;
    mapped.ribSourceU = srcRibU;
    mapped.ribQueryU = optRibU;
    mapped.alignmentScore = best.score;
    mapped.swapJunctions = best.swapJunctions;
    mapped.reverseWall = best.reverseWall;
    mapped.sourceTopologyKind = srcTopo.kind;
    return;
end

assert(strcmp(optGeometryMode, 'no_rib'), 'Unsupported optimizer geometry mode: %s', optGeometryMode);

if strcmp(srcTopo.kind, 'rib')
    sourceLoop = srcTopo.wallLoop(:);
else
    sourceLoop = srcTopo.loop(:);
end

optLoopNodes = double(opt.nodesCoords);
if size(optLoopNodes, 2) == 2
    optLoopNodes(:,3) = 0;
end

best = align_closed_loops(srcCoords, sourceLoop, optLoopNodes);
if best.score > 1e-3
    warning('validate_rib_optimizer_frozen_app:closedLoopAlignmentApprox', ...
        'Closed-loop alignment is approximate (mean node distance %.4g mm).', best.score);
end

loopSourceNodes = best.sourceNodes;

mapped = struct();
mapped.thickness = double(srcLayer.Thickness(loopSourceNodes));
mapped.flatTransition = double(srcLayer.FlatTransition(loopSourceNodes));
mapped.frozenMultiplier = double(srcFrozenMultiplier(loopSourceNodes));
mapped.nodesCompositeIDs = srcLayer.NodesCompositeIDs(loopSourceNodes,:);
mapped.alignedSourceLoop = best.alignedLoop;
mapped.loopSourceNodes = loopSourceNodes;
mapped.loopSourceU = best.sourceU;
mapped.loopQueryU = best.queryU;
mapped.alignmentScore = best.score;
mapped.reverseLoop = best.reverseLoop;
mapped.sourceTopologyKind = srcTopo.kind;
end

function [fullNodes, fullConn, masks] = build_full_optimized_topology(opt, optGeometryMode)
if strcmp(optGeometryMode, 'no_rib')
    fullNodes = double(opt.nodesCoords);
    if size(fullNodes, 2) == 2
        fullNodes(:,3) = 0;
    end
    nLoop = size(fullNodes, 1);
    assert(nLoop >= 3, 'No-rib optimizer loop must contain at least 3 nodes.');

    if isfield(opt, 'connectivity') && ~isempty(opt.connectivity)
        fullConn = double(opt.connectivity);
    else
        fullConn = [(1:nLoop)' [(2:nLoop)'; 1]];
    end
    if size(fullConn, 2) ~= 2 || any(fullConn(:) < 1) || any(fullConn(:) > nLoop)
        fullConn = [(1:nLoop)' [(2:nLoop)'; 1]];
    end

    masks = struct();
    masks.wallMask = true(nLoop, 1);
    masks.ribMask = false(nLoop, 1);
    masks.junctionMask = false(nLoop, 1);
    masks.nWall = nLoop;
    masks.nRibInterior = 0;
    masks.ribPath = zeros(0,1);
    return;
end

wallNodes = double(opt.wall_nodesCoords);
if size(wallNodes, 2) == 2
    wallNodes(:,3) = 0;
end

nWall = size(wallNodes, 1);
assert(nWall >= 3, 'Optimizer wall must contain at least 3 nodes.');

if isfield(opt, 'wall_connectivity') && ~isempty(opt.wall_connectivity)
    wallConn = double(opt.wall_connectivity);
else
    wallConn = [(1:nWall)' [(2:nWall)'; 1]];
end
if size(wallConn, 2) ~= 2 || any(wallConn(:) < 1) || any(wallConn(:) > nWall)
    wallConn = [(1:nWall)' [(2:nWall)'; 1]];
end

ribNodesFull = double(opt.rib_nodesCoords);
if size(ribNodesFull, 2) == 2
    ribNodesFull(:,3) = 0;
end
assert(size(ribNodesFull, 1) >= 2, 'Optimizer rib must contain at least the two junction nodes.');

optJWallIdx = double(opt.junction_wall_indices(:));
assert(numel(optJWallIdx) == 2, 'Expected 2 junction wall indices.');

j0Mismatch = norm(ribNodesFull(1,:) - wallNodes(optJWallIdx(1),:));
j1Mismatch = norm(ribNodesFull(end,:) - wallNodes(optJWallIdx(2),:));
if max(j0Mismatch, j1Mismatch) > 1e-4
    warning('validate_rib_optimizer_frozen_app:ribEndpointMismatch', ...
        'Rib endpoint coordinates do not exactly match wall junction coordinates (max %.4g mm).', ...
        max(j0Mismatch, j1Mismatch));
end

ribInteriorNodes = ribNodesFull(2:end-1, :);
fullNodes = [wallNodes; ribInteriorNodes];

nRibInterior = size(ribInteriorNodes, 1);
if nRibInterior > 0
    ribPath = [optJWallIdx(1); (nWall+1:nWall+nRibInterior)'; optJWallIdx(2)];
else
    ribPath = optJWallIdx(:);
end

if numel(ribPath) >= 2
    ribConn = [ribPath(1:end-1) ribPath(2:end)];
else
    ribConn = zeros(0,2);
end

fullConn = [wallConn; ribConn];

masks = struct();
masks.wallMask = false(size(fullNodes,1), 1);
masks.wallMask(1:nWall) = true;
masks.ribMask = false(size(fullNodes,1), 1);
if nRibInterior > 0
    masks.ribMask(nWall+1:end) = true;
end
masks.junctionMask = false(size(fullNodes,1), 1);
masks.junctionMask(optJWallIdx) = true;
masks.nWall = nWall;
masks.nRibInterior = nRibInterior;
masks.ribPath = ribPath;
end

function u = closed_branch_u(nodes)
nodes = double(nodes);
if size(nodes, 2) == 2
    nodes(:,3) = 0;
end

n = size(nodes, 1);
if n == 0
    u = zeros(0,1);
    return;
elseif n == 1
    u = 0;
    return;
end

ds = vecnorm(diff(nodes, 1, 1), 2, 2);
closingLength = norm(nodes(1,:) - nodes(end,:));
totalLength = sum(ds) + closingLength;
if totalLength <= eps
    u = (0:n-1)' / n;
    return;
end

s = [0; cumsum(ds)];
u = s / totalLength;
end

function u = open_branch_u(nodes)
nodes = double(nodes);
if size(nodes, 2) == 2
    nodes(:,3) = 0;
end

n = size(nodes, 1);
if n == 0
    u = zeros(0,1);
    return;
elseif n == 1
    u = 0;
    return;
end

ds = vecnorm(diff(nodes, 1, 1), 2, 2);
totalLength = sum(ds);
if totalLength <= eps
    u = linspace(0, 1, n)';
    return;
end

s = [0; cumsum(ds)];
u = s / totalLength;
end

function idx = nearest_closed_u_indices(sourceU, queryU)
sourceU = sourceU(:)';
queryU = queryU(:);
assert(~isempty(sourceU), 'Source closed-branch parameterization is empty.');

dist = abs(queryU - sourceU);
dist = min(dist, 1 - dist);
[~, idx] = min(dist, [], 2);
idx = idx(:);
end

function idx = nearest_open_u_indices(sourceU, queryU)
sourceU = sourceU(:)';
queryU = queryU(:);
assert(~isempty(sourceU), 'Source open-branch parameterization is empty.');

dist = abs(queryU - sourceU);
[~, idx] = min(dist, [], 2);
idx = idx(:);
end

function rotated = rotate_to_index(values, currentPos, desiredPos)
values = values(:);
n = numel(values);
assert(currentPos >= 1 && currentPos <= n, 'Current position out of range.');
assert(desiredPos >= 1 && desiredPos <= n, 'Desired position out of range.');

shift = desiredPos - currentPos;
idx = mod((0:n-1)' - shift, n) + 1;
rotated = values(idx);
end

function nodeLength = node_lengths_from_connectivity(nodes, connectivity)
nodes = double(nodes);
connectivity = double(connectivity);
if size(nodes, 2) == 2
    nodes(:,3) = 0;
end
if isempty(connectivity)
    nodeLength = zeros(size(nodes,1), 1);
    return;
end

edgeVec = nodes(connectivity(:,2), :) - nodes(connectivity(:,1), :);
edgeLength = sqrt(sum(edgeVec.^2, 2));
nNodes = size(nodes, 1);
nodeLength = accumarray(connectivity(:,1), edgeLength/2, [nNodes 1], @sum, 0) + ...
             accumarray(connectivity(:,2), edgeLength/2, [nNodes 1], @sum, 0);
end

function [stressVec, hitCount] = curvature_only_stress_vector(rocVec, nodesCompositeIDs, materialProp)
rocVec = double(rocVec(:));
nNodes = numel(rocVec);
stressVec = zeros(nNodes, 1);
hitCount = 0;

for j = 1:nNodes
    if iscell(nodesCompositeIDs)
        nodeMaterials = nodesCompositeIDs(j,:);
    else
        nodeMaterials = nodesCompositeIDs(j);
    end
    [stressVec(j), nodeHit] = curvature_only_stress_single(rocVec(j), nodeMaterials, materialProp);
    hitCount = hitCount + nodeHit;
end
end

function [stress, nodeHit] = curvature_only_stress_single(roc, nodeMaterials, materialProp)
names = valid_composite_names(nodeMaterials);
numDefined = numel(names);
if numDefined == 0
    stress = 0;
    nodeHit = 0;
    return;
end

stressVals = zeros(numDefined, 1);
nodeHit = 0;

for k = 1:numDefined
    materialName = names{k};
    if isfield(materialProp, materialName)
        materialEntry = materialProp.(materialName);
    else
        validName = matlab.lang.makeValidName(materialName);
        if isfield(materialProp, validName)
            materialEntry = materialProp.(validName);
        else
            materialEntry = struct();
        end
    end

    constantCrushStress = get_numeric_field(materialEntry, 'constantCrushStress', 0);
    curvatureScale = curvature_factor(roc, materialEntry);
    stressVals(k) = constantCrushStress * curvatureScale;
    if isfinite(stressVals(k))
        nodeHit = 1;
    else
        stressVals(k) = 0;
    end
end

stress = sum(stressVals) / numDefined;
end

function factor = curvature_factor(roc, materialEntry)
factor = 1;
if ~isstruct(materialEntry) || ~isfield(materialEntry, 'MatrixCurvatureEq')
    return;
end

eqData = materialEntry.MatrixCurvatureEq;
if isempty(eqData)
    return;
end
if iscell(eqData)
    eqHandle = eqData{1};
else
    eqHandle = eqData;
end

try
    factor = feval(eqHandle, double(roc));
catch
    factor = 1;
end

factor = double(factor);
if isempty(factor)
    factor = 1;
elseif numel(factor) > 1
    factor = factor(1);
end

if ~isfinite(factor)
    factor = 0;
end
end

function names = valid_composite_names(rawNames)
if ischar(rawNames) || isstring(rawNames)
    rawNames = cellstr(rawNames);
elseif ~iscell(rawNames)
    rawNames = {rawNames};
end

names = cell(0,1);
for k = 1:numel(rawNames)
    name = rawNames{k};
    if isstring(name)
        name = char(name);
    end
    if ~ischar(name)
        continue;
    end
    name = strtrim(name);
    if isempty(name) || strcmpi(name, 'NotDefined')
        continue;
    end
    names{end+1,1} = name; %#ok<AGROW>
end
end

function value = get_numeric_field(s, fieldName, defaultValue)
if isstruct(s) && isfield(s, fieldName) && ~isempty(s.(fieldName))
    value = double(s.(fieldName));
else
    value = defaultValue;
end
if isempty(value)
    value = defaultValue;
elseif numel(value) > 1
    value = value(1);
end
end

function topo = extract_section_topology(nodes, edges)
nodes = double(nodes);
edges = double(edges);
if size(nodes, 2) == 2
    nodes(:,3) = 0;
end

nNodes = size(nodes, 1);
G = graph(edges(:,1), edges(:,2), [], nNodes);
componentIdx = conncomp(G);
assert(max(componentIdx) == 1, 'Expected a single connected component in source topology.');

[~, degree] = build_adjacency(edges, nNodes);
if all(degree == 2)
    topo = extract_closed_loop_topology(nodes, edges);
else
    topo = extract_wall_and_ribs_like_optimizer(nodes, edges);
end
end

function topo = extract_closed_loop_topology(nodes, edges)
nodes = double(nodes);
edges = double(edges);
if size(nodes, 2) == 2
    nodes(:,3) = 0;
end

nNodes = size(nodes, 1);
[adj, degree] = build_adjacency(edges, nNodes);
assert(all(degree == 2), 'Expected a single closed loop (all nodes degree 2).');

startNode = edges(1,1);
orderedIDs = zeros(nNodes, 1);
orderedIDs(1) = startNode;
prevNode = NaN;
currentNode = startNode;

for i = 2:nNodes
    neighborsHere = adj{currentNode};
    if i == 2
        nextNode = neighborsHere(1);
    else
        nextNode = neighborsHere(neighborsHere ~= prevNode);
        assert(~isempty(nextNode), 'Failed to continue closed-loop ordering.');
        nextNode = nextNode(1);
    end
    orderedIDs(i) = nextNode;
    prevNode = currentNode;
    currentNode = nextNode;
end

topo = struct();
topo.kind = 'no_rib';
topo.loop = orderedIDs;
topo.wallLoop = orderedIDs;
topo.ribBranch = zeros(0,1);
topo.ribInterior = zeros(0,1);
topo.junctionNodes = zeros(0,1);
topo.j0WallIdx = [];
topo.j1WallIdx = [];
topo.nRibInterior = 0;
end

function topo = extract_wall_and_ribs_like_optimizer(nodes, edges)
nodes = double(nodes);
edges = double(edges);
if size(nodes, 2) == 2
    nodes(:,3) = 0;
end

nNodes = size(nodes, 1);
G = graph(edges(:,1), edges(:,2), [], nNodes);
componentIdx = conncomp(G);
assert(max(componentIdx) == 1, 'Expected a single connected component in source topology.');

[adj, degree] = build_adjacency(edges, nNodes);
[branches, junctions, endpoints] = decompose_branches_like_optimizer(edges, adj, degree, 1:nNodes);

assert(numel(junctions) == 2, 'Expected 2 junction nodes, got %d.', numel(junctions));
assert(numel(branches) == 3, 'Expected 3 branches, got %d.', numel(branches));

branchLengths = zeros(numel(branches), 1);
for b = 1:numel(branches)
    xyz = nodes(branches{b}, :);
    branchLengths(b) = sum(vecnorm(diff(xyz, 1, 1), 2, 2));
end

[~, ribIdx] = min(branchLengths);
wallIdx = find((1:numel(branches)) ~= ribIdx);

wallBr0 = branches{wallIdx(1)}(:);
wallBr1 = branches{wallIdx(2)}(:);
ribBranch = branches{ribIdx}(:);

if wallBr0(end) == wallBr1(1)
    wallLoop = [wallBr0; wallBr1(2:end)];
elseif wallBr0(end) == wallBr1(end)
    wallLoop = [wallBr0; flipud(wallBr1(1:end-1))];
elseif wallBr0(1) == wallBr1(1)
    wallLoop = [flipud(wallBr0); wallBr1(2:end)];
elseif wallBr0(1) == wallBr1(end)
    wallLoop = [wallBr1; wallBr0(2:end)];
else
    error('Wall branches do not share the two junction nodes.');
end

if wallLoop(end) == wallLoop(1)
    wallLoop(end) = [];
end

j0 = junctions(1);
j1 = junctions(2);
j0WallIdx = find(wallLoop == j0, 1);
j1WallIdx = find(wallLoop == j1, 1);

if ribBranch(1) == j0 && ribBranch(end) == j1
    % already oriented
elseif ribBranch(1) == j1 && ribBranch(end) == j0
    ribBranch = flipud(ribBranch);
else
    error('Rib branch endpoints do not match the detected junction nodes.');
end

topo = struct();
topo.kind = 'rib';
topo.wallLoop = wallLoop;
topo.ribBranch = ribBranch;
topo.ribInterior = ribBranch(2:end-1);
topo.junctionNodes = [j0; j1];
topo.j0WallIdx = j0WallIdx;
topo.j1WallIdx = j1WallIdx;
topo.nRibInterior = numel(topo.ribInterior);
topo.endpoints = endpoints(:);
topo.branches = branches;
topo.branchLengths = branchLengths;
end

function best = align_closed_loops(srcCoords, sourceLoop, optLoopNodes)
srcCoords = double(srcCoords);
sourceLoop = sourceLoop(:);
optLoopNodes = double(optLoopNodes);

best.score = inf;
best.alignedLoop = [];
best.sourceNodes = [];
best.sourceU = [];
best.queryU = [];
best.reverseLoop = false;
best.startIndex = NaN;

for reverseLoop = [false true]
    if reverseLoop
        loopCandidate = flipud(sourceLoop);
    else
        loopCandidate = sourceLoop;
    end

    for startIndex = 1:numel(loopCandidate)
        alignedLoop = rotate_to_index(loopCandidate, startIndex, 1);
        alignedCoords = srcCoords(alignedLoop, :);
        sourceU = closed_branch_u(alignedCoords);
        queryU = closed_branch_u(optLoopNodes);
        nearest = nearest_closed_u_indices(sourceU, queryU);
        sourceNodes = alignedLoop(nearest);
        score = mean(vecnorm(srcCoords(sourceNodes,:) - optLoopNodes, 2, 2));

        if score < best.score
            best.score = score;
            best.alignedLoop = alignedLoop;
            best.sourceNodes = sourceNodes;
            best.sourceU = sourceU;
            best.queryU = queryU;
            best.reverseLoop = reverseLoop;
            best.startIndex = startIndex;
        end
    end
end

assert(~isempty(best.alignedLoop), 'Could not align closed loops.');
end

function [adj, degree] = build_adjacency(edges, nNodes)
edges = double(edges);
if nargin < 2 || isempty(nNodes)
    if isempty(edges)
        nNodes = 0;
    else
        nNodes = max(edges(:));
    end
end

adj = cell(nNodes, 1);
for i = 1:size(edges, 1)
    a = edges(i,1);
    b = edges(i,2);
    adj{a}(end+1) = b; %#ok<AGROW>
    adj{b}(end+1) = a; %#ok<AGROW>
end

degree = zeros(nNodes, 1);
for i = 1:nNodes
    degree(i) = numel(adj{i});
end
end

function [branches, junctionNodes, endNodes] = decompose_branches_like_optimizer(edgesComp, adj, degree, compNodes)
edgesComp = double(edgesComp);
compNodes = compNodes(:);
if isempty(compNodes)
    branches = {};
    junctionNodes = zeros(0,1);
    endNodes = zeros(0,1);
    return;
end

compMask = false(max(max(edgesComp(:)), max(compNodes)), 1);
compMask(compNodes) = true;

junctionNodes = compNodes(degree(compNodes) > 2);
endNodes = compNodes(degree(compNodes) == 1);

branches = cell(0,1);
visitedEdges = false(size(edgesComp,1), 1);
markEdge = @(a,b) find((edgesComp(:,1) == a & edgesComp(:,2) == b) | ...
                       (edgesComp(:,1) == b & edgesComp(:,2) == a), 1);

startNodes = [endNodes; junctionNodes];
for s = 1:numel(startNodes)
    startNode = startNodes(s);
    neighborsStart = adj{startNode};

    for n = 1:numel(neighborsStart)
        nb = neighborsStart(n);
        if nb > numel(compMask) || ~compMask(nb)
            continue;
        end

        eID = markEdge(startNode, nb);
        if isempty(eID) || visitedEdges(eID)
            continue;
        end

        path = [startNode; nb];
        visitedEdges(eID) = true;
        currentNode = nb;
        prevNode = startNode;

        while true
            if degree(currentNode) ~= 2
                break;
            end

            nextNodes = adj{currentNode};
            nextNodes = nextNodes(nextNodes ~= prevNode);
            nextNodes = nextNodes(compMask(nextNodes));
            if isempty(nextNodes)
                break;
            end

            nextNode = nextNodes(1);
            eID = markEdge(currentNode, nextNode);
            if isempty(eID) || visitedEdges(eID)
                break;
            end

            visitedEdges(eID) = true;
            path(end+1,1) = nextNode; %#ok<AGROW>
            prevNode = currentNode;
            currentNode = nextNode;
        end

        branches{end+1,1} = path; %#ok<AGROW>
    end
end

for i = 1:size(edgesComp, 1)
    if visitedEdges(i)
        continue;
    end

    loopPath = edgesComp(i,:)';
    visitedEdges(i) = true;
    prevNode = loopPath(1);
    currentNode = loopPath(2);

    while true
        nextNodes = adj{currentNode};
        nextNodes = nextNodes(nextNodes ~= prevNode);
        nextNodes = nextNodes(compMask(nextNodes));
        if isempty(nextNodes)
            break;
        end

        nextNode = nextNodes(1);
        eID = markEdge(currentNode, nextNode);
        if isempty(eID) || visitedEdges(eID)
            break;
        end

        visitedEdges(eID) = true;
        loopPath(end+1,1) = nextNode; %#ok<AGROW>
        prevNode = currentNode;
        currentNode = nextNode;
    end

    branches{end+1,1} = loopPath; %#ok<AGROW>
end
end

function key = edge_key_str(a, b)
a = min(a, b);
b = max(a, b);
key = sprintf('%d_%d', a, b);
end

function RoCMapped = curvature_spline_like_app(nodesCoords, connectivity, flatTransition)
nodesCoords = double(nodesCoords);
connectivity = double(connectivity);
flatTransition = double(flatTransition(:));
minRocClamp = 15.0;

if size(nodesCoords, 2) == 2
    nodesCoords(:,3) = 0;
end
if isscalar(flatTransition)
    flatTransition = repmat(flatTransition, size(nodesCoords,1), 1);
end

RoCMapped = zeros(size(nodesCoords,1), 1);
tol = 1e-2;
overlapCount = 10;

G = graph(connectivity(:,1), connectivity(:,2), [], size(nodesCoords,1));
componentIndices = conncomp(G);
numSections = max(componentIndices);

sectionEdges = cell(numSections, 1);
sectionNodes = cell(numSections, 1);
for e = 1:size(connectivity, 1)
    n1 = connectivity(e,1);
    n2 = connectivity(e,2);
    c1 = componentIndices(n1);
    c2 = componentIndices(n2);
    if c1 == c2
        sectionEdges{c1}(end+1,:) = [n1 n2]; %#ok<AGROW>
    else
        warning('validate_rib_optimizer_frozen_app:crossComponentEdge', ...
            'Edge connects nodes in different components: %d-%d', n1, n2);
    end
end
for s = 1:numSections
    sectionNodes{s} = find(componentIndices == s);
end

for p = 1:numSections
    sectionConnectivity = sectionEdges{p};
    localNodeIds = sectionNodes{p};
    localNodes = nodesCoords(localNodeIds, :);

    allNodeIDs = sectionConnectivity(:);
    [~, ~, ic] = unique(allNodeIDs);
    counts = accumarray(ic, 1);

    if any(counts > 2)
        sectionMask = false(size(nodesCoords,1), 1);
        sectionMask(localNodeIds) = true;
        [adj, degree] = build_adjacency(sectionConnectivity, size(nodesCoords,1));
        junctionNodes = find((degree > 2) & sectionMask);
        endNodes = find((degree == 1) & sectionMask);

        branches = decompose_branches_like_optimizer(sectionConnectivity, adj, degree, localNodeIds);
        if iscell(branches)
            branches = branches(:);
        end

        radiusSum = zeros(numel(localNodeIds), 1);
        radiusCount = zeros(numel(localNodeIds), 1);

        for b = 1:numel(branches)
            ids = branches{b};
            [isIn, locNodes] = ismember(ids, localNodeIds);
            assert(all(isIn), 'Failed to localize branch nodes inside the section node set.');
            xyz = localNodes(locNodes, :);
            radius = branch_radius_from_points(xyz, tol);
            radiusSum(locNodes) = radiusSum(locNodes) + radius;
            radiusCount(locNodes) = radiusCount(locNodes) + 1;
        end

        rocLocal = inf(numel(localNodeIds), 1);
        hasRadius = radiusCount > 0;
        rocLocal(hasRadius) = radiusSum(hasRadius) ./ radiusCount(hasRadius);
        if isempty(junctionNodes) && isempty(endNodes)
            warning('validate_rib_optimizer_frozen_app:branchingNoEndpoints', ...
                'Detected branching without endpoints or junctions while replaying curvature.');
        end

    elseif numel(find(counts == 1)) == 0 || numel(find(counts == 1)) == 2
        endNodes = unique(allNodeIDs(counts(ic) == 1));
        if isempty(endNodes)
            isClosed = true;
            startNode = sectionConnectivity(1,1);
        else
            isClosed = false;
            startNode = endNodes(1);
        end

        orderedIDs = zeros(numel(localNodeIds), 1);
        orderedIDs(1) = startNode;

        mask1 = sectionConnectivity(:,1) == startNode;
        mask2 = sectionConnectivity(:,2) == startNode;
        idx = find(mask1 | mask2, 1);
        if isempty(idx)
            error('Could not start ordered traversal for section %d.', p);
        end

        if mask1(idx)
            nextNode = sectionConnectivity(idx,2);
        else
            nextNode = sectionConnectivity(idx,1);
        end
        orderedIDs(2) = nextNode;

        edgesLeft = sectionConnectivity;
        edgesLeft(idx,:) = [];
        i = 2;
        while i < numel(localNodeIds)
            currNode = orderedIDs(i);
            mask1 = edgesLeft(:,1) == currNode;
            mask2 = edgesLeft(:,2) == currNode;
            idx = find(mask1 | mask2, 1);
            if isempty(idx)
                break;
            end

            if mask1(idx)
                nextNode = edgesLeft(idx,2);
            else
                nextNode = edgesLeft(idx,1);
            end

            orderedIDs(i+1) = nextNode;
            edgesLeft(idx,:) = [];
            i = i + 1;
        end

        orderedIDs = orderedIDs(orderedIDs > 0);
        [isIn, locNodes] = ismember(orderedIDs, localNodeIds);
        assert(all(isIn), 'Failed to map ordered nodes to local section node IDs.');
        orderedNodes = localNodes(locNodes, :);

        orderedRadius = chain_radius_from_points(orderedNodes, isClosed, overlapCount, tol);
        rocLocal = inf(numel(localNodeIds), 1);
        rocLocal(locNodes) = orderedRadius;

    else
        error('Mesh is branched or disconnected in a way the replay curvature cannot resolve.');
    end

    RoCMapped(localNodeIds) = rocLocal;
end

RoCMapped = min(max(RoCMapped, minRocClamp), flatTransition);
end

function append_validation_row(csvPath, sourcePartitionMat, optimizerMat, layerIdx, ...
    sourceSavedForce, sourceReplayForce, pythonForce, frozenAppForce, ...
    frozenAppWallForce, pythonWallForce, frozenAppRibForce, pythonRibForce)

needHeader = ~isfile(csvPath);
[fid, msg] = fopen(csvPath, 'a');
assert(fid >= 0, 'Could not open %s for append: %s', csvPath, msg);
cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>

if needHeader
    fprintf(fid, ['timestamp,layerIdx,sourcePartitionMat,optimizerMat,' ...
        'totalDiffN,totalDiffPct\n']);
end

totalDiff = frozenAppForce - pythonForce;
totalDiffPct = 100 * totalDiff / max(abs(pythonForce), eps);

fprintf(fid, ['%s,%d,%s,%s,%.12g,%.12g\n'], ...
    datestr(now, 'yyyy-mm-dd HH:MM:SS'), ...
    layerIdx, ...
    csv_escape(sourcePartitionMat), ...
    csv_escape(optimizerMat), ...
    totalDiff, totalDiffPct);
end

function radius = branch_radius_from_points(xyz, tol)
xyz = double(xyz);
if size(xyz, 2) == 2
    xyz(:,3) = 0;
end

n = size(xyz, 1);
if n < 3
    radius = inf(n, 1);
    return;
end

ds = vecnorm(diff(xyz, 1, 1), 2, 2);
s = [0; cumsum(ds)];
if s(end) <= eps
    radius = inf(n, 1);
    return;
end
s = s / s(end);

try
    [sx, ~] = spaps(s, xyz(:,1), tol);
    [sy, ~] = spaps(s, xyz(:,2), tol);
    [sz, ~] = spaps(s, xyz(:,3), tol);
catch
    radius = inf(n, 1);
    return;
end

sQuery = s;
dx  = fnval(fnder(sx,1), sQuery);  dx  = dx(:);
dy  = fnval(fnder(sy,1), sQuery);  dy  = dy(:);
dz  = fnval(fnder(sz,1), sQuery);  dz  = dz(:);
d2x = fnval(fnder(sx,2), sQuery);  d2x = d2x(:);
d2y = fnval(fnder(sy,2), sQuery);  d2y = d2y(:);
d2z = fnval(fnder(sz,2), sQuery);  d2z = d2z(:);

T = [dx dy dz];
N = [d2x d2y d2z];
crossTN = cross(T, N, 2);
num = vecnorm(crossTN, 2, 2);
den = vecnorm(T, 2, 2).^3;

curvature = zeros(n, 1);
mask = den > eps;
curvature(mask) = num(mask) ./ den(mask);

radius = inf(n, 1);
mask = curvature > eps;
radius(mask) = 1 ./ curvature(mask);
end

function radius = chain_radius_from_points(orderedNodes, isClosed, overlapCount, tol)
orderedNodes = double(orderedNodes);
if size(orderedNodes, 2) == 2
    orderedNodes(:,3) = 0;
end

nOriginal = size(orderedNodes, 1);
if nOriginal < 3
    radius = inf(nOriginal, 1);
    return;
end

X = orderedNodes(:,1);
Y = orderedNodes(:,2);
Z = orderedNodes(:,3);

if isClosed
    actualOverlap = min(overlapCount, nOriginal - 1);
    if actualOverlap < 2
        isClosed = false;
        actualOverlap = 0;
        Xext = X;
        Yext = Y;
        Zext = Z;
    else
        Xext = [X(end-actualOverlap+1:end); X; X(1:actualOverlap)];
        Yext = [Y(end-actualOverlap+1:end); Y; Y(1:actualOverlap)];
        Zext = [Z(end-actualOverlap+1:end); Z; Z(1:actualOverlap)];
    end
else
    actualOverlap = 0;
    Xext = X;
    Yext = Y;
    Zext = Z;
end

dX = diff(Xext);
dY = diff(Yext);
dZ = diff(Zext);
ds = sqrt(dX.^2 + dY.^2 + dZ.^2);
s = [0; cumsum(ds)];
if s(end) <= eps
    radius = inf(nOriginal, 1);
    return;
end
s = s / s(end);

try
    [sx, ~] = spaps(s, Xext, tol);
    [sy, ~] = spaps(s, Yext, tol);
    [sz, ~] = spaps(s, Zext, tol);
catch
    radius = inf(nOriginal, 1);
    return;
end

if isClosed && actualOverlap >= 2
    sQuery = linspace(s(actualOverlap+1), s(end-actualOverlap), nOriginal)';
else
    sQuery = linspace(s(1), s(end), nOriginal)';
end

dx  = fnval(fnder(sx,1), sQuery);  dx  = dx(:);
dy  = fnval(fnder(sy,1), sQuery);  dy  = dy(:);
dz  = fnval(fnder(sz,1), sQuery);  dz  = dz(:);
d2x = fnval(fnder(sx,2), sQuery);  d2x = d2x(:);
d2y = fnval(fnder(sy,2), sQuery);  d2y = d2y(:);
d2z = fnval(fnder(sz,2), sQuery);  d2z = d2z(:);

T = [dx dy dz];
N = [d2x d2y d2z];
crossTN = cross(T, N, 2);
num = vecnorm(crossTN, 2, 2);
den = vecnorm(T, 2, 2).^3;

curvature = zeros(nOriginal, 1);
mask = den > eps;
curvature(mask) = num(mask) ./ den(mask);

radius = inf(nOriginal, 1);
mask = curvature > eps;
radius(mask) = 1 ./ curvature(mask);
end

function out = csv_escape(value)
if isstring(value)
    value = char(value);
end
if ~ischar(value)
    value = char(string(value));
end
value = strrep(value, '"', '""');
out = ['"' value '"'];
end
