% generate_square.m
% =========================================================================
% Generates a square cross-section and saves it as a .mat file ready to be
% loaded by the Python optimizer.
%
% Geometry:
%   - A centered square with side length S
%   - Node ordering is CCW, starting at the bottom-left corner
%
% Output .mat fields  (must match what loadData() expects):
%   nodesCoords   [N x 3]  double  - (x, y, 0) in mm, row = one node
%   connectivity  [N x 2]  double  - 1-based edge pairs, closed loop
%                                   edge k connects node k -> node k+1
%                                   last edge connects node N -> node 1
%
% Node distribution:
%   Each of the four sides gets a perimeter-proportional node count.
%   For a square this is an even split, with any remainder distributed to
%   the first sides in CCW order so the total matches n_total exactly.
%
% Dependencies: none (core MATLAB only)
%
% Francisco Arriscado / FEUP / 2026
% =========================================================================

clear; clc; close all;

%% ======================== USER PARAMETERS ================================

S         = 100;      % [mm]  Square side length
n_total   = 100;     % Total node count
outputFile = 'partition_data_square.mat';

% Optional override: nodes per side (bottom, right, top, left)
% Leave all at 0 to use the automatic perimeter-proportional split.
n_side_override = [0 0 0 0];


%% ======================== INPUT VALIDATION ================================

assert(S > 0, 'S must be positive.');
assert(n_total >= 8, 'n_total should be at least 8 (2 nodes per side).');
assert(numel(n_side_override) == 4, 'n_side_override must have 4 elements.');


%% ======================== NODE COUNT SPLIT ================================
% All four sides have the same length, so the perimeter-proportional split
% is simply an even distribution of the total node count.

if all(n_side_override > 0)
    n_side = n_side_override(:).';
    n_total = sum(n_side);
    fprintf('Using manual split: n_side = [%d %d %d %d]  (total=%d)\n', ...
        n_side(1), n_side(2), n_side(3), n_side(4), n_total);
else
    base_n = floor(n_total / 4);
    rem_n  = mod(n_total, 4);
    n_side = base_n * ones(1, 4);
    n_side(1:rem_n) = n_side(1:rem_n) + 1;

    if any(n_side < 2)
        error('n_total=%d is too small for a square with 2 nodes per side.', n_total);
    end

    n_total = sum(n_side);
    fprintf('Even split: n_side = [%d %d %d %d]  (total=%d)\n', ...
        n_side(1), n_side(2), n_side(3), n_side(4), n_total);
end

seg_labels = {'bottom','right','top','left'};
seg_lengths = (S * ones(1,4));


%% ======================== SEGMENT GENERATORS =============================
% Each side includes its START point and excludes its END point, so the
% four segments concatenate into a single closed loop without duplicates.

seg_line = @(ps, pe, n) ps + linspace(0, 1, n+1)' * (pe - ps);
trim = @(pts) pts(1:end-1, :);


%% ======================== NODE GENERATION ================================

% CCW order starting at the bottom-left corner
p1s = [-S/2, -S/2];   p1e = [ S/2, -S/2];   % bottom edge, left -> right
p2s = [ S/2, -S/2];   p2e = [ S/2,  S/2];   % right edge, bottom -> top
p3s = [ S/2,  S/2];   p3e = [-S/2,  S/2];   % top edge, right -> left
p4s = [-S/2,  S/2];   p4e = [-S/2, -S/2];   % left edge, top -> bottom

seg1 = trim(seg_line(p1s, p1e, n_side(1)));
seg2 = trim(seg_line(p2s, p2e, n_side(2)));
seg3 = trim(seg_line(p3s, p3e, n_side(3)));
seg4 = trim(seg_line(p4s, p4e, n_side(4)));

xy = [seg1; seg2; seg3; seg4];
nodesCoords = [xy, zeros(size(xy, 1), 1)];
N = size(nodesCoords, 1);

assert(N == n_total, ...
    'Node count mismatch: expected %d, assembled %d.', n_total, N);


%% ======================== CONNECTIVITY ===================================

connectivity = [(1:N)', [2:N, 1]'];


%% ======================== QUALITY CHECKS =================================

dx    = diff(nodesCoords([1:end, 1], 1));
dy    = diff(nodesCoords([1:end, 1], 2));
edgeL = sqrt(dx.^2 + dy.^2);

perimeter   = sum(edgeL);
mean_edge   = mean(edgeL);
std_edge    = std(edgeL);
cv_edge     = std_edge / mean_edge;

fprintf('\n--- Quality Report ---\n');
fprintf('Side length   : %.3f mm\n', S);
fprintf('Nodes         : %d\n', N);
fprintf('Perimeter     : %.3f mm   (expected %.3f mm)\n', perimeter, 4*S);
fprintf('Mean edge len : %.3f mm\n', mean_edge);
fprintf('Std  edge len : %.3f mm\n', std_edge);
fprintf('CV (std/mean) : %.4f     (< 0.3 recommended)\n', cv_edge);

fprintf('\nPer-side spacing:\n');
fprintf('  %-12s  nodes  arc_len [mm]  mean_edge [mm]\n', 'Side');
fprintf('  %s\n', repmat('-', 1, 52));
cumN = 0;
for k = 1:4
    nk  = n_side(k);
    idx = cumN + (1:nk);
    ek  = edgeL(idx);
    fprintf('  %-12s  %5d  %12.3f  %13.4f\n', ...
        seg_labels{k}, nk, seg_lengths(k), mean(ek));
    cumN = cumN + nk;
end

centroid = mean(nodesCoords(:,1:2));
fprintf('\nCentroid      : (%.3f, %.3f) mm\n', centroid(1), centroid(2));


%% ======================== SAVE ===========================================

save(outputFile, 'nodesCoords', 'connectivity');
fprintf('\nSaved -> %s\n', outputFile);
fprintf('Fields: nodesCoords [%d x 3], connectivity [%d x 2]\n', N, N);
fprintf('To use: set dataFileName = ''%s'' in the optimizer.\n', outputFile);


%% ======================== PLOTS ==========================================

figure('Name', 'Square geometry check', 'Position', [100 100 1200 400]);

% --- 1. Shape with node labels -------------------------------------------
subplot(1,3,1);
closed = [nodesCoords(:,1:2); nodesCoords(1,1:2)];
plot(closed(:,1), closed(:,2), 'b-', 'LineWidth', 1.2);
hold on;
scatter(nodesCoords(:,1), nodesCoords(:,2), 20, 'b', 'filled');
scatter(nodesCoords(1,1), nodesCoords(1,2), 60, 'r', 'filled');

step = max(1, floor(N/20));
for k = 1:step:N
    text(nodesCoords(k,1)*1.05, nodesCoords(k,2)*1.05, num2str(k), ...
        'FontSize', 6, 'Color', [0.4 0.4 0.4]);
end

axis equal; grid on;
title(sprintf('Square  (N=%d, S=%g mm)', N, S));
xlabel('x [mm]'); ylabel('y [mm]');
legend({'Contour','Nodes','Start'}, 'Location','northwest');

text(0, -S/2*0.95, 'bottom', 'FontSize', 8, 'HorizontalAlignment','center', ...
    'Color', [0.2 0.5 0.2]);
text(S/2*0.95, 0, 'right', 'FontSize', 8, 'HorizontalAlignment','center', ...
    'Color', [0.2 0.2 0.7], 'Rotation', 90);
text(0, S/2*0.95, 'top', 'FontSize', 8, 'HorizontalAlignment','center', ...
    'Color', [0.2 0.5 0.2]);
text(-S/2*0.95, 0, 'left', 'FontSize', 8, 'HorizontalAlignment','center', ...
    'Color', [0.2 0.2 0.7], 'Rotation', 90);

% --- 2. Edge length profile -----------------------------------------------
subplot(1,3,2);
plot(1:N, edgeL, '.-', 'MarkerSize', 5, 'LineWidth', 0.8);
hold on;

cumN = 0;
for k = 1:4
    nk = n_side(k);
    xline(cumN + 0.5, ':', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.6);
    cumN = cumN + nk;
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
    nk = n_side(k);
    xline(cumN + 0.5, ':', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.6);
    cumN = cumN + nk;
end
yline(5,       'r--', 'min clamp (5 mm)',    'FontSize', 7, ...
    'LabelHorizontalAlignment','left');
yline(1221.95, 'r--', 'flat transition',      'FontSize', 7, ...
    'LabelHorizontalAlignment','left');
xlabel('Node index');
ylabel('RoC [mm]  (log scale)');
title('Approx. RoC  (FD, not spline)');
grid on; ylim([3, 3e3]);
xlim([1, N]);

sgtitle(sprintf('Square: S=%.0f mm  |  N=%d  |  perimeter=%.1f mm', ...
    S, N, perimeter));

fprintf('\nPlot rendered. Checks to confirm:\n');
fprintf('  Shape  : four straight edges in CCW order\n');
fprintf('  Nodes  : uniform spacing per side up to remainder distribution\n');
fprintf('  RoC    : straight edges near flat_transition, corners near clamp\n');
