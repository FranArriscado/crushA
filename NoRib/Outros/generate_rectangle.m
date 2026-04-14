% generate_rectangle.m
% =========================================================================
% Generates a centered rectangle cross-section and saves it as a .mat file
% ready to be loaded by the Python optimizer.
%
% Geometry:
%   - Rectangle centered at the origin
%   - CCW traversal starting at the bottom-left corner
%   - Four straight segments, no corner rounding
%
% Output .mat fields  (must match what loadData() expects):
%   nodesCoords   [N x 3]  double  - (x, y, 0) in mm, row = one node
%   connectivity  [N x 2]  double  - 1-based edge pairs, closed loop
%                                  edge k connects node k -> node k+1
%                                  last edge connects node N -> node 1
%
% Node ordering:
%   Starts at bottom-left corner and walks CCW:
%     bottom edge  : (-W/2,-H/2) -> (+W/2,-H/2)
%     right edge   : (+W/2,-H/2) -> (+W/2,+H/2)
%     top edge     : (+W/2,+H/2) -> (-W/2,+H/2)
%     left edge    : (-W/2,+H/2) -> (-W/2,-H/2)
%
% Dependencies: none (core MATLAB only)
%
% Francisco Arriscado / FEUP / 2026
% =========================================================================

clear; clc; close all;

%% ======================== USER PARAMETERS ================================

W        = 2*94.496;     % [mm] Total width  (x-direction)
H        = 94.496;     % [mm] Total height (y-direction)
n_total  = 100;    % Total node count, split proportionally across the 4 sides

outputFile = 'partition_data_rectangle.mat';

% Optional overrides (leave as 0 to use the automatic split)
n_h_override = 0;  % nodes on each horizontal side
n_v_override = 0;  % nodes on each vertical side


%% ======================== INPUT VALIDATION ===============================

assert(W > 0 && H > 0, 'W and H must be positive.');
assert(n_total >= 8,   'n_total should be at least 8 (2 nodes per side).');

if mod(n_total, 2) ~= 0
    warning('n_total must be even for paired horizontal/vertical counts. Rounding up to %d.', n_total + 1);
    n_total = n_total + 1;
end


%% ======================== ARC-LENGTH PROPORTIONAL SPLIT ==================
% Each side has a length:
%   horizontal side : L_h = W
%   vertical side   : L_v = H
% There are two of each.

L_h     = W;
L_v     = H;
L_total = 2 * (L_h + L_v);

if n_h_override > 0 && n_v_override > 0
    n_h = n_h_override;
    n_v = n_v_override;
    n_total = 2 * n_h + 2 * n_v;
    fprintf('Using manual split: n_h=%d, n_v=%d  (total=%d)\n', n_h, n_v, n_total);
else
    n_pairs_total = n_total / 2;    % one pair for horizontals, one for verticals
    n_h = max(2, round(n_pairs_total * L_h / (L_h + L_v)));
    n_v = max(2, n_pairs_total - n_h);

    % Enforce a minimum of 2 nodes per side while preserving the pair total.
    if n_h < 2
        n_h = 2;
        n_v = n_pairs_total - n_h;
    elseif n_v < 2
        n_v = 2;
        n_h = n_pairs_total - n_v;
    end

    if n_h < 2 || n_v < 2
        error('Automatic split produced fewer than 2 nodes on one side. Increase n_total.');
    end

    n_total = 2 * n_h + 2 * n_v;
    fprintf('Arc-length split: n_h=%d, n_v=%d  (total=%d)\n', n_h, n_v, n_total);
end

seg_labels  = {'bottom','right','top','left'};
seg_lengths = [L_h, L_v, L_h, L_v];
seg_counts  = [n_h, n_v, n_h, n_v];


%% ======================== SEGMENT GENERATORS =============================
% Straight segment: p_start -> p_end, n nodes [n x 2]
% Includes the START point and excludes the END point so junctions are not
% duplicated when the four segments are concatenated.

seg_straight = @(ps, pe, n) ps + (linspace(0, 1, n + 1)' * (pe - ps));
trim = @(pts) pts(1:end-1, :);


%% ======================== NODE GENERATION ================================
% CCW traversal starting at the bottom-left corner.

p1s = [-W/2, -H/2];
p1e = [ W/2, -H/2];
seg1 = trim(seg_straight(p1s, p1e, n_h));

p2s = [ W/2, -H/2];
p2e = [ W/2,  H/2];
seg2 = trim(seg_straight(p2s, p2e, n_v));

p3s = [ W/2,  H/2];
p3e = [-W/2,  H/2];
seg3 = trim(seg_straight(p3s, p3e, n_h));

p4s = [-W/2,  H/2];
p4e = [-W/2, -H/2];
seg4 = trim(seg_straight(p4s, p4e, n_v));

xy = [seg1; seg2; seg3; seg4];

nodesCoords = [xy, zeros(size(xy, 1), 1)];
N = size(nodesCoords, 1);

assert(N == n_total, ...
    'Node count mismatch: expected %d, assembled %d.', n_total, N);


%% ======================== CONNECTIVITY ===================================

connectivity = [(1:N)', [2:N, 1]'];


%% ======================== SEGMENT BOUNDARY INDICES =======================

seg_start_idx = cumsum([1, n_h, n_v, n_h]);


%% ======================== QUALITY CHECKS =================================

dx    = diff(nodesCoords([1:end, 1], 1));
dy    = diff(nodesCoords([1:end, 1], 2));
edgeL = sqrt(dx.^2 + dy.^2);

perimeter   = sum(edgeL);
mean_edge   = mean(edgeL);
std_edge    = std(edgeL);
cv_edge     = std_edge / mean_edge;
expected_perim = 2 * (W + H);
perim_err      = abs(perimeter - expected_perim);

fprintf('\n--- Quality Report ---\n');
fprintf('Dimensions    : W=%.1f mm, H=%.1f mm\n', W, H);
fprintf('Nodes         : %d\n', N);
fprintf('Perimeter     : %.4f mm  (expected %.4f mm, error %.2e mm)\n', ...
    perimeter, expected_perim, perim_err);
fprintf('Mean edge len : %.4f mm\n', mean_edge);
fprintf('Std  edge len : %.4f mm\n', std_edge);
fprintf('CV (std/mean) : %.4f     (< 0.3 recommended)\n', cv_edge);

fprintf('\nPer-segment spacing:\n');
fprintf('  %-12s  nodes  arc_len [mm]  mean_edge [mm]\n', 'Segment');
fprintf('  %s\n', repmat('-', 1, 56));
cumN = 0;
for k = 1:4
    nk  = seg_counts(k);
    lk  = seg_lengths(k);
    idx = cumN + (1:nk);
    ek  = edgeL(idx);
    fprintf('  %-12s  %5d  %12.3f  %13.4f\n', seg_labels{k}, nk, lk, mean(ek));
    cumN = cumN + nk;
end

fprintf('\nJunction continuity:\n');
junctions = [
    -W/2, -H/2;
     W/2, -H/2;
     W/2,  H/2;
    -W/2,  H/2
];

for k = 1:4
    actual_xy = nodesCoords(seg_start_idx(k), 1:2);
    gap = norm(actual_xy - junctions(k, :));
    fprintf('  Corner %d gap = %.2e mm\n', k, gap);
end

centroid = mean(nodesCoords(:, 1:2));
fprintf('Centroid      : (%.4f, %.4f) mm  (expected approximately (0, 0))\n', ...
    centroid(1), centroid(2));


%% ======================== SAVE ===========================================

save(outputFile, 'nodesCoords', 'connectivity');
fprintf('\nSaved -> %s\n', outputFile);
fprintf('Fields: nodesCoords [%d x 3], connectivity [%d x 2]\n', N, N);


%% ======================== PLOTS ==========================================

figure('Name', 'Rectangle geometry check', 'Position', [100 100 1200 400]);

type_color = struct();
type_color.side = [0.12 0.47 0.71];
type_color.corner = [0.84 0.15 0.16];

% --- 1. Shape with node labels -------------------------------------------
subplot(1, 3, 1);
hold on; axis equal; grid on;

cumN = 0;
for k = 1:4
    nk = seg_counts(k);
    idx = cumN + (1:nk);
    next_start = mod(cumN + nk, N) + 1;
    idx_closed = [idx, next_start];
    plot(nodesCoords(idx_closed, 1), nodesCoords(idx_closed, 2), ...
        '-', 'Color', type_color.side, 'LineWidth', 1.6);
    scatter(nodesCoords(cumN + 1, 1), nodesCoords(cumN + 1, 2), ...
        36, type_color.side, 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 0.5);
    cumN = cumN + nk;
end

step = max(1, floor(N / 20));
for k = 1:step:N
    text(nodesCoords(k, 1) * 1.07, nodesCoords(k, 2) * 1.07, num2str(k), ...
        'FontSize', 6, 'Color', [0.45 0.45 0.45]);
end

title(sprintf('Rectangle  N=%d', N));
xlabel('x [mm]'); ylabel('y [mm]');
xlim([-W/2 * 1.3, W/2 * 1.3]);
ylim([-H/2 * 1.3, H/2 * 1.3]);


% --- 2. Edge length profile ----------------------------------------------
subplot(1, 3, 2);
plot(1:N, edgeL, '.-', 'MarkerSize', 4, 'LineWidth', 0.8, ...
    'Color', type_color.side);
hold on;

cumN = 0;
for k = 1:4
    nk = seg_counts(k);
    xline(cumN + 0.5, ':', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.6);
    cumN = cumN + nk;
end

yline(mean_edge, 'k--', sprintf('mean=%.2f mm', mean_edge), ...
    'LabelHorizontalAlignment', 'left', 'FontSize', 7);
xlabel('Edge index');
ylabel('Length [mm]');
title('Edge length profile');
grid on;
ylim([0, max(edgeL) * 1.3]);
xlim([1, N]);


% --- 3. Approximate RoC preview (finite-difference) ----------------------
subplot(1, 3, 3);

wrap2 = @(v) [v(end, :); v; v(1, :)];
nw = wrap2(nodesCoords(:, 1:2));
d1 = nw(3:end, :) - nw(1:end-2, :);
d2 = nw(3:end, :) - 2 * nw(2:end-1, :) + nw(1:end-2, :);
cross_z = d1(:, 1) .* d2(:, 2) - d1(:, 2) .* d2(:, 1);
kappa   = abs(cross_z) ./ ((d1(:, 1).^2 + d1(:, 2).^2).^(3/2) + 1e-12);
RoC_approx = min(max(1 ./ kappa, 5), 1221.95);

semilogy(1:N, RoC_approx, '.-', 'MarkerSize', 4, 'LineWidth', 0.8, ...
    'Color', type_color.side);
hold on;

cumN = 0;
for k = 1:4
    nk = seg_counts(k);
    xline(cumN + 0.5, ':', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.6);
    cumN = cumN + nk;
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

sgtitle(sprintf('Rectangle: W=%.0f  H=%.0f mm  |  N=%d  |  perimeter=%.1f mm', ...
    W, H, N, perimeter));

fprintf('\nPlot rendered. Checks to confirm:\n');
fprintf('  - Contour is centered at the origin\n');
fprintf('  - Node order is CCW starting at the bottom-left corner\n');
fprintf('  - Edge lengths are roughly uniform within each side type\n');
fprintf('\nTo use: set dataFileName = ''%s'' in the optimizer.\n', outputFile);
