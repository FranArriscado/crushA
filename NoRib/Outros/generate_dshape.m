% generate_dshape.m
% =========================================================================
% Generates a D-shape cross-section and saves it as a .mat file ready to
% be loaded by the Python optimizer (optimizationModule_patched_v8.py).
%
% Geometry:
%   - Flat wall on the left  : x = 0,  y ∈ [-R, +R]  (vertical segment)
%   - Semicircle on the right: radius R, centred at origin, covering the
%     right half-plane (x ≥ 0), from (0,+R) clockwise to (0,-R)
%
%   The two segments join exactly at (0, +R) and (0, -R) so the contour
%   is a single closed loop with no duplicate nodes at the junctions.
%
% Output .mat fields  (must match what loadData() expects):
%   nodesCoords   [N×3]  double  — (x, y, 0) in mm, row = one node
%   connectivity  [N×2]  double  — 1-based edge pairs, closed loop
%                                   edge k connects node k → node k+1
%                                   last edge connects node N → node 1
%
% Node ordering:
%   Starts at bottom junction (0, -R), walks CCW:
%     flat wall  upward  → (0, +R)   [n_flat nodes, junction inclusive]
%     semicircle rightward → (0, -R) [n_arc  nodes, start exclusive, end inclusive]
%   This matches the CCW convention used in the existing square/ellipse
%   mockups and is consistent with the right-hand-rule orientation expected
%   by the curvature spline.
%
% Dependencies: none (core MATLAB only)
%
% Francisco Arriscado / FEUP / 2026
% =========================================================================

clear; clc; close all;

%% ======================== USER PARAMETERS ================================

R         = 30;      % [mm]  Semicircle radius = half the flat-wall height (2R tall)
n_total   = 100;     % Total node count — distributes proportionally by arc length
                     % Rule of thumb: 80–140 nodes ↔ well-resolved spline

outputFile = 'partition_data_dshape.mat';

% ── Optional: override individual counts (leave as 0 to use arc-length split)
n_flat_override = 0;   % nodes on flat wall   (0 = auto)
n_arc_override  = 0;   % nodes on semicircle  (0 = auto)


%% ======================== ARC-LENGTH PROPORTIONAL SPLIT ==================
% Perimeter of each segment:
%   flat wall  : L_flat = 2*R
%   semicircle : L_arc  = π*R
% Proportional node counts keep uniform spacing across both segments.

L_flat = 2 * R;
L_arc  = pi * R;
L_total = L_flat + L_arc;

if n_flat_override > 0 && n_arc_override > 0
    n_flat = n_flat_override;
    n_arc  = n_arc_override;
    n_total = n_flat + n_arc;
    fprintf('Using manual split: n_flat=%d, n_arc=%d\n', n_flat, n_arc);
else
    % Distribute proportionally, keep at least 3 nodes per segment
    n_flat = max(3, round(n_total * L_flat / L_total));
    n_arc  = n_total - n_flat;
    fprintf('Arc-length split: n_flat=%d, n_arc=%d  (total=%d)\n', ...
        n_flat, n_arc, n_total);
end


%% ======================== NODE GENERATION ================================
% ── Flat wall: (0, -R) → (0, +R), inclusive at both ends
%    n_flat nodes, evenly spaced along y.
%    These include the two junction nodes.

y_flat = linspace(-R, R, n_flat)';
x_flat = zeros(n_flat, 1);
z_flat = zeros(n_flat, 1);

flat_nodes = [x_flat, y_flat, z_flat];

% ── Semicircle: (0, +R) → (0, -R) going through (R, 0) (CCW in standard
%    orientation when viewed from +z).
%    θ goes from +90° to -90° (i.e., π/2 → -π/2).
%    We EXCLUDE the endpoints because they coincide with the flat-wall
%    junctions already stored as the last and first flat nodes.

theta_arc = linspace(pi/2, -pi/2, n_arc + 2)';   % n_arc+2 to include endpoints
theta_arc = theta_arc(2:end-1);                   % remove junction duplicates

x_arc = R * cos(theta_arc);
y_arc = R * sin(theta_arc);
z_arc = zeros(n_arc, 1);

arc_nodes = [x_arc, y_arc, z_arc];


%% ======================== ASSEMBLE =======================================

nodesCoords = [flat_nodes; arc_nodes];   % [N×3]
N = size(nodesCoords, 1);

% Verify node count
assert(N == n_flat + n_arc, ...
    'Node count mismatch: expected %d, got %d', n_flat + n_arc, N);

% Closed-loop connectivity (1-based, matching MATLAB convention)
%   Edge k : node k → node k+1
%   Edge N : node N → node 1
connectivity = [(1:N)', [2:N, 1]'];      % [N×2]


%% ======================== QUALITY CHECKS =================================

% ── Edge length uniformity
dx    = diff(nodesCoords([1:end,1], 1));
dy    = diff(nodesCoords([1:end,1], 2));
edgeL = sqrt(dx.^2 + dy.^2);

perimeter   = sum(edgeL);
mean_edge   = mean(edgeL);
std_edge    = std(edgeL);
cv_edge     = std_edge / mean_edge;       % coefficient of variation

fprintf('\n--- Quality Report ---\n');
fprintf('Nodes         : %d\n', N);
fprintf('Perimeter     : %.3f mm   (expected %.3f mm)\n', perimeter, L_total);
fprintf('Mean edge len : %.3f mm\n', mean_edge);
fprintf('Std  edge len : %.3f mm\n', std_edge);
fprintf('CV (std/mean) : %.4f     (< 0.3 recommended)\n', cv_edge);

% ── Junction continuity: check no gap at flat↔arc seams
junction_top_gap = norm(nodesCoords(n_flat, 1:2) - [0, R]);
junction_bot_gap = norm(nodesCoords(1,     1:2) - [0, -R]);
arc_start_gap    = norm(nodesCoords(n_flat+1, 1:2) - (R*[cos(theta_arc(1)), sin(theta_arc(1))]));

fprintf('Junction gap (bottom): %.2e mm  (should be ~0)\n', junction_bot_gap);
fprintf('Junction gap (top)   : %.2e mm  (should be ~0)\n', junction_top_gap);

% ── Centroid
centroid = mean(nodesCoords(:,1:2));
fprintf('Centroid      : (%.3f, %.3f) mm\n', centroid(1), centroid(2));


%% ======================== SAVE ===========================================

save(outputFile, 'nodesCoords', 'connectivity');
fprintf('\nSaved → %s\n', outputFile);
fprintf('Fields: nodesCoords [%d×3], connectivity [%d×2]\n', N, N);


%% ======================== PLOTS ==========================================

figure('Name', 'D-shape geometry check', 'Position', [100 100 1100 380]);

% --- 1. Shape with node labels (junction nodes highlighted) ---------------
subplot(1,3,1);
closed = [nodesCoords(:,1:2); nodesCoords(1,1:2)];
plot(closed(:,1), closed(:,2), 'b-', 'LineWidth', 1.2);
hold on;
scatter(nodesCoords(:,1), nodesCoords(:,2), 20, 'b', 'filled');

% Highlight junction nodes in red
junctions = [1, n_flat];
scatter(nodesCoords(junctions,1), nodesCoords(junctions,2), ...
    60, 'r', 'filled', 'DisplayName', 'Junctions');

% Label every ~10th node for visual check
step = max(1, floor(N/20));
for k = 1:step:N
    text(nodesCoords(k,1)*1.08, nodesCoords(k,2)*1.08, num2str(k), ...
        'FontSize', 6, 'Color', [0.4 0.4 0.4]);
end

axis equal; grid on;
title(sprintf('D-shape  (N=%d, R=%g mm)', N, R));
xlabel('x [mm]'); ylabel('y [mm]');
legend({'Contour','Nodes','Junctions'}, 'Location','northwest');

% Annotate segments
text(-R*0.25, 0, 'flat', 'FontSize', 8, 'HorizontalAlignment','center', ...
    'Color', [0.2 0.5 0.2]);
text(R*0.65, 0, 'arc', 'FontSize', 8, 'HorizontalAlignment','center', ...
    'Color', [0.2 0.2 0.7]);

% --- 2. Edge length profile -----------------------------------------------
subplot(1,3,2);
plot(1:N, edgeL, '.-', 'MarkerSize', 5, 'LineWidth', 0.8);
xline(n_flat + 0.5, 'r--', 'flat|arc', 'LabelVerticalAlignment','bottom');
yline(mean_edge, 'k--', sprintf('mean=%.2f mm', mean_edge));
xlabel('Edge index (k → k+1)');
ylabel('Length [mm]');
title('Edge length profile');
grid on;
ylim([0, max(edgeL)*1.3]);

% --- 3. Approximate RoC preview (finite-difference curvature) -------------
% This is a rough visual check, not the spline-based computation in MATLAB.
subplot(1,3,3);
wrap = @(v) [v(end-1:end,:); v; v(1:2,:)];
nw = wrap(nodesCoords(:,1:2));
d1 = nw(3:end,:)   - nw(1:end-2,:);
d2 = nw(3:end,:) - 2*nw(2:end-1,:) + nw(1:end-2,:);
cross_z = d1(:,1).*d2(:,2) - d1(:,2).*d2(:,1);
den     = (d1(:,1).^2 + d1(:,2).^2).^(3/2) + 1e-12;
kappa   = abs(cross_z) ./ den;
RoC_approx = min(max(1./kappa, 5), 1221.95);   % same clamps as optimizer

semilogy(1:N, RoC_approx, '.-', 'MarkerSize', 5, 'LineWidth', 0.8);
xline(n_flat + 0.5, 'r--', 'flat|arc');
yline(5,       'r:', 'min clamp (5)');
yline(1221.95, 'r:', 'flat transition');
xlabel('Node index');
ylabel('RoC [mm]  (log scale)');
title('Approximate RoC (FD, not spline)');
grid on;

sgtitle(sprintf('D-shape mockup  —  perimeter=%.1f mm, N=%d', perimeter, N));

fprintf('\nPlot rendered. Check:\n');
fprintf('  - Flat wall nodes are on the left (x≈0)\n');
fprintf('  - Semicircle curves to the right\n');
fprintf('  - RoC flat wall >> flat_transition (clamped at top)\n');
fprintf('  - RoC corner nodes near 5 mm (clamped at bottom)\n');
fprintf('\nTo use: set dataFileName = ''%s'' in the optimizer.\n', outputFile);
