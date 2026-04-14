% ValidateOriginal.m
% =========================================================================
% Validates Python optimizer output using the MATLAB analytical tool's
% ground-truth formulation, adapted for a single 2D cross-section.
%
% This script replicates the EXACT same physics as layersProp.m:
%   - curvatureSpline: spaps (smoothing spline), linspace sQuery, overlap=10
%   - RoC clamped to [min_roc, flat_transition]
%   - nodeLength = sum(adjacent edge lengths) / 2
%   - area = nodeLength x thickness
%   - crushStress = constantCrushStress x curvatureEq(RoC)
%   - forceNode = crushStress x area
%   - forceTotal = sum(forceNode)
%
% DEPENDENCIES: Curve Fitting Toolbox (spaps, fnval, fnder)
%
% Francisco Arriscado / FEUP / 2026
% =========================================================================

clear; clc; close all;

%% ======================== PROJECT PATHS ================================

addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..'));
cfg = config();

%% ======================== USER INPUTS ==================================

% Shape to validate (original geometry from data/)
% Options: 'square', 'ellipse', 'rrect', 'dshape', 'cross', 'teardrop', etc.
shape = 'square';

% Python initial force (from the optimizer log, step 0)
pythonPredictedForce = 113548.9609;

% Resolve full path automatically
dataFile = fullfile(cfg.data, ['partition_data_' shape '.mat']);

% Physical parameters (must match what the optimizer used)
thickness_val       = 2.0;    % [mm]
constantCrushStress = 90;     % [MPa]
min_roc             = 15.0;   % [mm] must match optimizer min_roc

% Material curve coefficients (from DataCurvatureCarbon.xlsx)
curvEq      = @(x) 7.951858693686812  .* x.^(-0.9409590693634183) + 0.9900990009786546;
curvEqLower = @(x) 4.78983973619878   .* x.^(-0.5883999449654764) + 0.6523314616586323;
curvEqUpper = @(x) 19.560792535656788 .* x.^(-1.3900057011370337) + 1.151550789276735;

flat_transition = 1221.947188466746;  % [mm]


%% ======================== LOAD DATA ====================================

data         = load(dataFile);
nodeCoords   = double(data.nodesCoords);
connectivity = double(data.connectivity);
numNodes     = size(nodeCoords, 1);
numEdges     = size(connectivity, 1);
fprintf('Loaded %d nodes, %d edges from %s\n', numNodes, numEdges, dataFile);


%% ======================== ORDER NODES ==================================

G   = graph(connectivity(:,1), connectivity(:,2));
deg = degree(G);
assert(all(deg == 2), 'Expected a single closed loop (all nodes degree 2).');

startNode     = connectivity(1,1);
orderedIDs    = zeros(numNodes, 1);
orderedIDs(1) = startNode;

mask1 = connectivity(:,1) == startNode;
mask2 = connectivity(:,2) == startNode;
idx   = find(mask1 | mask2, 1);

if mask1(idx)
    nextNode = connectivity(idx,2);
else
    nextNode = connectivity(idx,1);
end
orderedIDs(2) = nextNode;

edgesLeft        = connectivity;
edgesLeft(idx,:) = [];

i = 2;
while i < numNodes
    currNode = orderedIDs(i);
    mask1    = edgesLeft(:,1) == currNode;
    mask2    = edgesLeft(:,2) == currNode;
    idx      = find(mask1 | mask2, 1);
    if isempty(idx), break; end
    if mask1(idx)
        nextNode = edgesLeft(idx,2);
    else
        nextNode = edgesLeft(idx,1);
    end
    orderedIDs(i+1)  = nextNode;
    edgesLeft(idx,:) = [];
    i = i + 1;
end

orderedIDs   = orderedIDs(orderedIDs > 0);
orderedNodes = nodeCoords(orderedIDs, :);
nOrd         = size(orderedNodes, 1);
fprintf('Ordered %d nodes in closed loop.\n', nOrd);


%% ======================== EDGE LENGTHS =================================

orderedConnectivity = [(1:nOrd)', [(2:nOrd)'; 1]];

vecDiff    = orderedNodes(orderedConnectivity(:,2), :) ...
           - orderedNodes(orderedConnectivity(:,1), :);
edgeLength = sqrt(sum(vecDiff.^2, 2));
perimeter  = sum(edgeLength);
fprintf('Perimeter: %.4f mm\n', perimeter);


%% ======================== NODE LENGTH ==================================

nodeLength = zeros(nOrd, 1);
for j = 1:nOrd
    idxNodeRows   = find(any(orderedConnectivity == j, 2));
    nodeLength(j) = sum(edgeLength(idxNodeRows)) / 2;
end
fprintf('Sum(nodeLength): %.4f mm (should equal perimeter: %.4f)\n', ...
    sum(nodeLength), perimeter);


%% ======================== AREA =========================================

area = nodeLength * thickness_val;


%% ======================== CURVATURE (RoC) ==============================

overlapCount = 10;
splineTol    = 1e-2;

X = orderedNodes(:,1);
Y = orderedNodes(:,2);
Z = orderedNodes(:,3);

X_ext = [X(end-overlapCount+1:end); X; X(1:overlapCount)];
Y_ext = [Y(end-overlapCount+1:end); Y; Y(1:overlapCount)];
Z_ext = [Z(end-overlapCount+1:end); Z; Z(1:overlapCount)];

dX = diff(X_ext); dY = diff(Y_ext); dZ = diff(Z_ext);
ds = sqrt(dX.^2 + dY.^2 + dZ.^2);
s  = [0; cumsum(ds)];
s  = s / s(end);

[sx, ~] = spaps(s', X_ext', splineTol);
[sy, ~] = spaps(s', Y_ext', splineTol);
[sz, ~] = spaps(s', Z_ext', splineTol);

sQuery = linspace(s(overlapCount+1), s(end-overlapCount), nOrd);

dx  = fnval(fnder(sx,1), sQuery)';
dy  = fnval(fnder(sy,1), sQuery)';
dz  = fnval(fnder(sz,1), sQuery)';
d2x = fnval(fnder(sx,2), sQuery)';
d2y = fnval(fnder(sy,2), sQuery)';
d2z = fnval(fnder(sz,2), sQuery)';

T        = [dx dy dz];
N        = [d2x d2y d2z];
cross_TN = cross(T, N, 2);
num      = vecnorm(cross_TN, 2, 2);
den      = vecnorm(T, 2, 2).^3;

curvature = num ./ den;
RoC       = 1 ./ curvature;
RoC       = min(max(RoC, min_roc), flat_transition);

fprintf('RoC: min=%.2f, max=%.2f, median=%.2f mm\n', ...
    min(RoC), max(RoC), median(RoC));


%% ======================== CRUSH STRESS & FORCE =========================

crushStress      = constantCrushStress .* curvEq(RoC);
crushStressLower = constantCrushStress .* curvEqLower(RoC);
crushStressUpper = constantCrushStress .* curvEqUpper(RoC);

forceNode      = crushStress      .* area;
forceNodeLower = crushStressLower .* area;
forceNodeUpper = crushStressUpper .* area;

forceTotal      = sum(forceNode);
forceTotalLower = sum(forceNodeLower);
forceTotalUpper = sum(forceNodeUpper);


%% ======================== COMPARISON ===================================

diff_N   = forceTotal - pythonPredictedForce;
diff_pct = 100 * diff_N / pythonPredictedForce;

fprintf('\n=========================================================\n');
fprintf('  VALIDATION RESULTS\n');
fprintf('  (MATLAB ground-truth formulation vs Python optimizer)\n');
fprintf('=========================================================\n');
fprintf('  MATLAB force (mean):     %12.2f N\n', forceTotal);
fprintf('  MATLAB force (lower):    %12.2f N\n', forceTotalLower);
fprintf('  MATLAB force (upper):    %12.2f N\n', forceTotalUpper);
fprintf('  Python force (from log): %12.2f N\n', pythonPredictedForce);
fprintf('  ---------------------------------------------------------\n');
fprintf('  Difference:              %+12.2f N  (%+.2f%%)\n', diff_N, diff_pct);
fprintf('=========================================================\n');

fprintf('\n--- Diagnostics ---\n');
fprintf('Perimeter:           %.4f mm\n', perimeter);
fprintf('Thickness:           %.2f mm\n', thickness_val);
fprintf('min_roc:             %.2f mm\n', min_roc);
fprintf('flat_transition:     %.2f mm\n', flat_transition);
fprintf('Mean crushStress:    %.2f MPa\n', mean(crushStress));
fprintf('Nodes at min_roc (clamped): %d / %d\n', sum(RoC <= min_roc + 0.01), nOrd);
fprintf('Nodes at flat_tr  (clamped): %d / %d\n', sum(RoC >= flat_transition - 0.01), nOrd);


%% ======================== LOG TO CSV ===================================

csvFile = fullfile(cfg.validation, 'validation_log.csv');

if ~isfile(csvFile)
    fid = fopen(csvFile, 'w');
    fprintf(fid, 'Timestamp,DataFile,Thickness mm,min roc mm,MATLAB Force N,Python Force N,Difference N,Difference pct\n');
    fclose(fid);
end

fid = fopen(csvFile, 'a');
assert(fid ~= -1, 'Cannot open %s for writing. Is it open in Excel?', csvFile);
ts  = datestr(now, 31);
fprintf(fid, '%s,%s,%.2f,%.1f,%.2f,%.2f,%.2f,%.4f\n', ...
    ts, dataFile, thickness_val, min_roc, ...
    forceTotal, pythonPredictedForce, diff_N, diff_pct);
fclose(fid);

fprintf('Result appended to %s\n', csvFile);

%% ======================== EXPORT DIAGNOSTICS ===========================

% Dense spline curve — real spaps evaluated at 1000 points for Python plotting
nDense      = 1000;
sDense      = linspace(s(overlapCount+1), s(end-overlapCount), nDense);
splineDense = [fnval(sx, sDense)', fnval(sy, sDense)', fnval(sz, sDense)'];  % [1000 x 3]

save(fullfile(cfg.validation, 'validation_diagnostics.mat'), ...
    'orderedNodes', ...    % [nOrd x 3] ordered node coordinates
    'RoC', ...             % [nOrd x 1] radius of curvature per node (spaps, clamped)
    'crushStress', ...     % [nOrd x 1] crush stress per node
    'nodeLength', ...      % [nOrd x 1] contributing length per node
    'forceNode', ...       % [nOrd x 1] force per node
    'forceTotal', ...      % scalar — total force
    'sQuery', ...          % [1 x nOrd] query points used for per-node values
    'splineDense');        % [1000 x 3] real spaps curve at 1000 dense points

fprintf('Diagnostics saved to: validation_diagnostics.mat\n');
