% generate_stadium.m
% =========================================================================
% Generates a "stadium"/capsule cross-section and saves it as a .mat file
% ready to be loaded by the Python optimizer.
%
% Geometry:
%   - Horizontal top and bottom straight segments of length L_mid
%   - Left and right curved endcaps given by half-ellipses with semi-axes:
%         a_cap along x,  b_cap along y
%   - The full contour is centred at the origin
%   - Node ordering is CCW, starting at the bottom-left tangent point
%
% Special case:
%   - If a_cap = b_cap, the endcaps are semicircles and the profile is the
%     classic stadium / obround shape.
%
% Output .mat fields  (must match what loadData() expects):
%   nodesCoords   [N x 3]  double  - (x, y, 0) in mm, row = one node
%   connectivity  [N x 2]  double  - 1-based edge pairs, closed loop
%                                   edge k connects node k -> node k+1
%                                   last edge connects node N -> node 1
%
% Node distribution:
%   Nodes are split proportionally across the 4 segments:
%     1. bottom straight
%     2. right half-ellipse
%     3. top straight
%     4. left half-ellipse
%   Curved caps use an arc-length lookup table so their spacing is nearly
%   uniform even when a_cap ~= b_cap.
%
% Reference:
%   The default values below reproduce the family of shapes represented by
%   partition_data_ellipse_cleaned.mat: straight flats at y = +/-25 mm with
%   symmetric curved endcaps and total width 105 mm.
%
% Dependencies: none (core MATLAB only)
%
% Francisco Arriscado / FEUP / 2026
% =========================================================================

clear; clc; close all;

%% ======================== USER PARAMETERS ================================

L_mid   = 100;     % [mm] Length of the straight middle section (top/bottom)
a_cap   = 50;     % [mm] Half-ellipse semi-axis along x for each endcap
b_cap   = 50;     % [mm] Half-ellipse semi-axis along y for each endcap
n_total = 100;     % Total node count on the closed loop

outputFile = 'partition_data_stadium.mat';

% Optional overrides (leave as 0 to use arc-length proportional split)
n_mid_override = 0;   % nodes on each straight segment
n_cap_override = 0;   % nodes on each half-ellipse endcap

% Dense lookup size for cap arc-length inversion
n_dense_cap = 4000;

% Optional comparison against the supplied cleaned reference shape
compareWithReference = true;


%% ======================== INPUT VALIDATION ===============================

assert(L_mid > 0,                 'L_mid must be positive.');
assert(a_cap > 0 && b_cap > 0,    'a_cap and b_cap must be positive.');
assert(n_total >= 12,             'n_total should be at least 12.');
assert(n_dense_cap >= 500,        'n_dense_cap should be at least 500.');


%% ======================== ARC-LENGTH SPLIT ===============================
% Segment lengths:
%   bottom straight = L_mid
%   right cap       = length of half-ellipse
%   top straight    = L_mid
%   left cap        = length of half-ellipse

t_dense = linspace(-pi/2, pi/2, n_dense_cap + 1)';
cap_speed = sqrt((a_cap * sin(t_dense)).^2 + (b_cap * cos(t_dense)).^2);
s_cap_dense = cumtrapz(t_dense, cap_speed);
L_cap = s_cap_dense(end);
L_total = 2 * L_mid + 2 * L_cap;

if n_mid_override > 0 && n_cap_override > 0
    n_mid = n_mid_override;
    n_cap = n_cap_override;
    n_total = 2 * n_mid + 2 * n_cap;
    fprintf('Using manual split: n_mid=%d, n_cap=%d  (total=%d)\n', ...
        n_mid, n_cap, n_total);
else
    n_pair_total = n_total / 2;
    if mod(n_total, 2) ~= 0
        warning('n_total must be even for paired segment counts. Rounding up to %d.', n_total + 1);
        n_total = n_total + 1;
        n_pair_total = n_total / 2;
    end

    n_mid = max(2, round(n_pair_total * L_mid / (L_mid + L_cap)));
    n_cap = max(2, n_pair_total - n_mid);

    if n_mid < 2
        n_mid = 2;
        n_cap = n_pair_total - n_mid;
    elseif n_cap < 2
        n_cap = 2;
        n_mid = n_pair_total - n_cap;
    end

    if n_mid < 2 || n_cap < 2
        error('Automatic split produced fewer than 2 nodes on a segment. Increase n_total.');
    end

    n_total = 2 * n_mid + 2 * n_cap;
    fprintf('Arc-length split: n_mid=%d, n_cap=%d  (total=%d)\n', ...
        n_mid, n_cap, n_total);
end

seg_labels  = {'bottom straight','right cap','top straight','left cap'};
seg_lengths = [L_mid, L_cap, L_mid, L_cap];
seg_counts  = [n_mid, n_cap, n_mid, n_cap];


%% ======================== SEGMENT GENERATORS =============================
% Every segment includes its START point and excludes its END point.

seg_straight = @(ps, pe, n) ps + (linspace(0, 1, n + 1)' * (pe - ps));
trim = @(pts) pts(1:end-1, :);

% Build one half-ellipse with nearly uniform arc-length spacing.
% Side = +1 for the right cap, -1 for the left cap.
half_ellipse = @(side, n) makeHalfEllipse(side, n, L_mid, a_cap, b_cap, t_dense, s_cap_dense);


%% ======================== NODE GENERATION ================================
% Start at the bottom-left tangent point:
%   (-L_mid/2, -b_cap)
% Then walk CCW: bottom -> right cap -> top -> left cap

p1s = [-L_mid/2, -b_cap];
p1e = [ L_mid/2, -b_cap];
seg1 = trim(seg_straight(p1s, p1e, n_mid));

seg2 = half_ellipse(+1, n_cap);

p3s = [ L_mid/2,  b_cap];
p3e = [-L_mid/2,  b_cap];
seg3 = trim(seg_straight(p3s, p3e, n_mid));

seg4 = half_ellipse(-1, n_cap);

xy = [seg1; seg2; seg3; seg4];
nodesCoords = [xy, zeros(size(xy,1), 1)];
N = size(nodesCoords, 1);

assert(N == n_total, ...
    'Node count mismatch: expected %d, assembled %d.', n_total, N);


%% ======================== CONNECTIVITY ===================================

connectivity = [(1:N)', [2:N, 1]'];


%% ======================== SEGMENT BOUNDARY INDICES =======================

seg_start_idx = cumsum([1, n_mid, n_cap, n_mid]);


%% ======================== QUALITY CHECKS =================================

dx    = diff(nodesCoords([1:end, 1], 1));
dy    = diff(nodesCoords([1:end, 1], 2));
edgeL = sqrt(dx.^2 + dy.^2);

perimeter   = sum(edgeL);
mean_edge   = mean(edgeL);
std_edge    = std(edgeL);
cv_edge     = std_edge / mean_edge;

fprintf('\n--- Quality Report ---\n');
fprintf('Parameters    : L_mid=%.2f mm, a_cap=%.2f mm, b_cap=%.2f mm\n', ...
    L_mid, a_cap, b_cap);
fprintf('Total size    : width=%.2f mm, height=%.2f mm\n', L_mid + 2*a_cap, 2*b_cap);
fprintf('Nodes         : %d\n', N);
fprintf('Perimeter     : %.4f mm  (expected %.4f mm, error %.2e mm)\n', ...
    perimeter, L_total, abs(perimeter - L_total));
fprintf('Mean edge len : %.4f mm\n', mean_edge);
fprintf('Std  edge len : %.4f mm\n', std_edge);
fprintf('CV (std/mean) : %.4f     (< 0.3 recommended)\n', cv_edge);

fprintf('\nPer-segment spacing:\n');
fprintf('  %-16s  nodes  arc_len [mm]  mean_edge [mm]\n', 'Segment');
fprintf('  %s\n', repmat('-', 1, 60));
cumN = 0;
for k = 1:4
    nk  = seg_counts(k);
    idx = cumN + (1:nk);
    ek  = edgeL(idx);
    fprintf('  %-16s  %5d  %12.3f  %13.4f\n', ...
        seg_labels{k}, nk, seg_lengths(k), mean(ek));
    cumN = cumN + nk;
end

junctions = [
    -L_mid/2, -b_cap;
     L_mid/2, -b_cap;
     L_mid/2,  b_cap;
    -L_mid/2,  b_cap
];

fprintf('\nJunction continuity:\n');
for k = 1:4
    actual_xy = nodesCoords(seg_start_idx(k), 1:2);
    gap = norm(actual_xy - junctions(k, :));
    fprintf('  Junction %d gap = %.2e mm\n', k, gap);
end

centroid = mean(nodesCoords(:,1:2), 1);
fprintf('Centroid      : (%.4f, %.4f) mm  (expected approximately (0, 0))\n', ...
    centroid(1), centroid(2));


%% ======================== OPTIONAL REFERENCE CHECK =======================

if compareWithReference
    scriptDir = fileparts(mfilename('fullpath'));
    referenceFile = fullfile(scriptDir, '..', 'Validar_partitions', ...
        'New_Mat&Plot', 'partition_data_ellipse_cleaned.mat');

    if exist(referenceFile, 'file') == 2
        ref = load(referenceFile, 'nodesCoords');
        ref_xy = ref.nodesCoords(:, 1:2);

        if size(ref_xy, 1) == N
            [best_rms, best_shift] = bestCyclicRMSError(nodesCoords(:,1:2), ref_xy);
            fprintf('\nReference check:\n');
            fprintf('  File         : %s\n', referenceFile);
            fprintf('  Best shift   : %d nodes\n', best_shift);
            fprintf('  RMS mismatch : %.4e mm\n', best_rms);
        else
            fprintf('\nReference check skipped: node count mismatch (generated %d, reference %d).\n', ...
                N, size(ref_xy, 1));
        end
    else
        fprintf('\nReference check skipped: file not found.\n');
    end
end


%% ======================== SAVE ===========================================

save(outputFile, 'nodesCoords', 'connectivity');
fprintf('\nSaved -> %s\n', outputFile);
fprintf('Fields: nodesCoords [%d x 3], connectivity [%d x 2]\n', N, N);
fprintf('To use: set dataFileName = ''%s'' in the optimizer.\n', outputFile);


%% ======================== PLOTS ==========================================

figure('Name', 'Stadium geometry check', 'Position', [100 100 1200 400]);

% --- 1. Shape with segment starts ----------------------------------------
subplot(1,3,1);
hold on; axis equal; grid on;

closed = [nodesCoords(:,1:2); nodesCoords(1,1:2)];
plot(closed(:,1), closed(:,2), 'b-', 'LineWidth', 1.3);
scatter(nodesCoords(:,1), nodesCoords(:,2), 20, 'b', 'filled');

start_idx = seg_start_idx;
scatter(nodesCoords(start_idx,1), nodesCoords(start_idx,2), ...
    55, 'r', 'filled', 'DisplayName', 'Segment starts');

step = max(1, floor(N/20));
for k = 1:step:N
    text(nodesCoords(k,1)*1.04, nodesCoords(k,2)*1.04, num2str(k), ...
        'FontSize', 6, 'Color', [0.4 0.4 0.4]);
end

axis equal; grid on;
title(sprintf('Stadium  (N=%d)', N));
xlabel('x [mm]'); ylabel('y [mm]');
legend({'Contour','Nodes','Segment starts'}, 'Location', 'northwest');

text(0, -0.88*b_cap, 'bottom straight', 'HorizontalAlignment', 'center', ...
    'FontSize', 8, 'Color', [0.2 0.5 0.2]);
text(0,  0.88*b_cap, 'top straight', 'HorizontalAlignment', 'center', ...
    'FontSize', 8, 'Color', [0.2 0.5 0.2]);
text( L_mid/2 + 0.65*a_cap, 0, 'right cap', 'HorizontalAlignment', 'center', ...
    'FontSize', 8, 'Color', [0.2 0.2 0.7], 'Rotation', 90);
text(-L_mid/2 - 0.65*a_cap, 0, 'left cap', 'HorizontalAlignment', 'center', ...
    'FontSize', 8, 'Color', [0.2 0.2 0.7], 'Rotation', 90);

% --- 2. Edge length profile ----------------------------------------------
subplot(1,3,2);
plot(1:N, edgeL, '.-', 'MarkerSize', 5, 'LineWidth', 0.8);
hold on;

cumN = 0;
for k = 1:4
    xline(cumN + 0.5, ':', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.6);
    cumN = cumN + seg_counts(k);
end

yline(mean_edge, 'k--', sprintf('mean=%.2f mm', mean_edge));
xlabel('Edge index (k -> k+1)');
ylabel('Length [mm]');
title('Edge length profile');
grid on;
ylim([0, max(edgeL)*1.3]);
xlim([1, N]);

% --- 3. Approximate RoC preview (finite-difference) ----------------------
subplot(1,3,3);
wrap = @(v) [v(end,:); v; v(1,:)];
nw = wrap(nodesCoords(:,1:2));
d1 = nw(3:end,:)   - nw(1:end-2,:);
d2 = nw(3:end,:) - 2*nw(2:end-1,:) + nw(1:end-2,:);
cross_z = d1(:,1).*d2(:,2) - d1(:,2).*d2(:,1);
den     = (d1(:,1).^2 + d1(:,2).^2).^(3/2) + 1e-12;
kappa   = abs(cross_z) ./ den;
RoC_approx = min(max(1./kappa, 5), 1221.95);

semilogy(1:N, RoC_approx, '.-', 'MarkerSize', 5, 'LineWidth', 0.8);
hold on;

cumN = 0;
for k = 1:4
    xline(cumN + 0.5, ':', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.6);
    cumN = cumN + seg_counts(k);
end

yline(5,       'r--', 'min clamp (5 mm)', 'FontSize', 7, ...
    'LabelHorizontalAlignment', 'left');
yline(1221.95, 'r--', 'flat transition', 'FontSize', 7, ...
    'LabelHorizontalAlignment', 'left');
xlabel('Node index');
ylabel('RoC [mm]  (log scale)');
title('Approx. RoC  (FD, not spline)');
grid on;
ylim([3, 3e3]);
xlim([1, N]);

sgtitle(sprintf(['Stadium: L_{mid}=%.0f mm, a_{cap}=%.0f mm, b_{cap}=%.0f mm', ...
    '  |  N=%d  |  perimeter=%.1f mm'], L_mid, a_cap, b_cap, N, perimeter));

fprintf('\nPlot rendered. Checks to confirm:\n');
fprintf('  - Top and bottom are straight and horizontal\n');
fprintf('  - Left and right sides are half-elliptic caps\n');
fprintf('  - Node order is CCW starting at the bottom-left tangent point\n');


%% ======================== LOCAL FUNCTIONS ================================

function pts = makeHalfEllipse(side, n, L_mid, a_cap, b_cap, t_dense, s_dense)
    % side = +1 -> right cap,  side = -1 -> left cap
    s_target = linspace(0, s_dense(end), n + 1)';
    t_full = interp1(s_dense, t_dense, s_target, 'pchip');

    if side > 0
        % Bottom tangent -> top tangent along the right outer cap
        t_nodes = t_full(1:end-1);
        x = +L_mid/2 + a_cap * cos(t_nodes);
        y = b_cap * sin(t_nodes);
    else
        % Top tangent -> bottom tangent along the left outer cap
        t_nodes = flipud(t_full);
        t_nodes = t_nodes(1:end-1);
        x = -L_mid/2 - a_cap * cos(t_nodes);
        y = b_cap * sin(t_nodes);
    end

    pts = [x, y];
end

function [best_rms, best_shift] = bestCyclicRMSError(xy, ref_xy)
    n = size(xy, 1);
    best_rms = inf;
    best_shift = 0;

    for shift = 0:n-1
        ref_shift = ref_xy(mod((0:n-1) + shift, n) + 1, :);
        rms_err = sqrt(mean(sum((xy - ref_shift).^2, 2)));
        if rms_err < best_rms
            best_rms = rms_err;
            best_shift = shift;
        end
    end
end
