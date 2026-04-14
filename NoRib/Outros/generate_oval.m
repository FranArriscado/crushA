% generate_ellipse.m
% =========================================================================
% Generates an ellipse cross-section and saves it as a .mat file ready to
% be loaded by the Python optimizer (optimizationModule_patched_v8.py).
%
% Geometry:
%   - Ellipse centred at the origin
%   - Semi-axes aligned with the coordinate axes
%   - Node order is CCW, starting at the bottom point (0, -b)
%
% Output .mat fields  (must match what loadData() expects):
%   nodesCoords   [N x 3]  double  - (x, y, 0) in mm, row = one node
%   connectivity  [N x 2]  double  - 1-based edge pairs, closed loop
%                                   edge k connects node k -> node k+1
%                                   last edge connects node N -> node 1
%
% Node distribution:
%   Nodes are placed at approximately equal arc-length spacing by sampling
%   the ellipse perimeter numerically and inverting the cumulative arc-length
%   map. This keeps the spacing much more uniform than uniform theta sampling.
%
% Dependencies: none (core MATLAB only)
%
% Francisco Arriscado / FEUP / 2026
% =========================================================================

clear; clc; close all;

%% ======================== USER PARAMETERS ================================

a         = 2*53.3137;     % [mm]  Semi-axis along x
b         = 53.3137;     % [mm]  Semi-axis along y
n_total   = 100;    % Total node count on the closed loop

outputFile = 'partition_data_ellipse.mat';

% Optional: increase this if you want a finer arc-length lookup table.
n_dense = 20000;

% Optional angular offset [deg]. Default keeps node 1 at the bottom point.
theta0_deg = -90;


%% ======================== INPUT VALIDATION ================================

assert(a > 0 && b > 0, 'Semi-axes a and b must be positive.');
assert(n_total >= 12,  'n_total should be at least 12 for a closed ellipse.');
assert(n_dense >= 1000, 'n_dense should be at least 1000 for stable inversion.');


%% ======================== ARC-LENGTH MAP ================================
% Parametric ellipse:
%   x(theta) = a cos(theta)
%   y(theta) = b sin(theta)
% Speed along the curve:
%   ds/dtheta = sqrt((a sin(theta))^2 + (b cos(theta))^2)
%
% Build a dense arc-length table over one full revolution and invert it to
% place nodes at equal arc-length fractions.

theta0 = deg2rad(theta0_deg);
theta_dense = linspace(theta0, theta0 + 2*pi, n_dense + 1)';   % include the endpoint
speed_dense = sqrt((a * sin(theta_dense)).^2 + (b * cos(theta_dense)).^2);
s_dense = cumtrapz(theta_dense, speed_dense);
perimeter = s_dense(end);
s_norm = s_dense / perimeter;

% Equal arc-length fractions, excluding the repeated endpoint at 1.0
u_nodes = (0:n_total-1)' / n_total;
theta_nodes = interp1(s_norm, theta_dense, u_nodes, 'pchip');


%% ======================== NODE GENERATION ================================

x = a * cos(theta_nodes);
y = b * sin(theta_nodes);
z = zeros(n_total, 1);

nodesCoords = [x, y, z];
N = size(nodesCoords, 1);

assert(N == n_total, ...
    'Node count mismatch: expected %d, got %d', n_total, N);


%% ======================== CONNECTIVITY ===================================

connectivity = [(1:N)', [2:N, 1]'];   % closed loop, 1-based


%% ======================== QUALITY CHECKS =================================

dx    = diff(nodesCoords([1:end,1], 1));
dy    = diff(nodesCoords([1:end,1], 2));
edgeL = sqrt(dx.^2 + dy.^2);

perimeter_num = sum(edgeL);
mean_edge     = mean(edgeL);
std_edge      = std(edgeL);
cv_edge       = std_edge / mean_edge;

fprintf('\n--- Quality Report ---\n');
fprintf('Axes          : a=%.1f mm, b=%.1f mm\n', a, b);
fprintf('Nodes         : %d\n', N);
fprintf('Perimeter     : %.4f mm  (dense-table estimate %.4f mm, error %.2e mm)\n', ...
    perimeter_num, perimeter, abs(perimeter_num - perimeter));
fprintf('Mean edge len : %.4f mm\n', mean_edge);
fprintf('Std  edge len : %.4f mm\n', std_edge);
fprintf('CV (std/mean) : %.4f     (< 0.3 recommended)\n', cv_edge);

centroid = mean(nodesCoords(:,1:2));
fprintf('Centroid      : (%.4f, %.4f) mm\n', centroid(1), centroid(2));


%% ======================== SAVE ===========================================

save(outputFile, 'nodesCoords', 'connectivity');
fprintf('\nSaved -> %s\n', outputFile);
fprintf('Fields: nodesCoords [%d x 3], connectivity [%d x 2]\n', N, N);
fprintf('To use: set dataFileName = ''%s'' in the optimizer.\n', outputFile);


%% ======================== PLOTS ==========================================

figure('Name', 'Ellipse geometry check', 'Position', [100 100 1200 400]);

% --- 1. Shape with node labels -------------------------------------------
subplot(1,3,1);
closed = [nodesCoords(:,1:2); nodesCoords(1,1:2)];
plot(closed(:,1), closed(:,2), 'b-', 'LineWidth', 1.2);
hold on;
scatter(nodesCoords(:,1), nodesCoords(:,2), 20, 'b', 'filled');
scatter(nodesCoords(1,1), nodesCoords(1,2), 60, 'r', 'filled', ...
    'DisplayName', 'Start node');

step = max(1, floor(N/20));
for k = 1:step:N
    text(nodesCoords(k,1)*1.05, nodesCoords(k,2)*1.05, num2str(k), ...
        'FontSize', 6, 'Color', [0.4 0.4 0.4]);
end

axis equal; grid on;
title(sprintf('Ellipse  (N=%d, a=%g mm, b=%g mm)', N, a, b));
xlabel('x [mm]'); ylabel('y [mm]');
legend({'Contour','Nodes','Start node'}, 'Location', 'northwest');

text(0, 0, 'center', 'FontSize', 8, 'HorizontalAlignment', 'center', ...
    'Color', [0.2 0.2 0.2]);

% --- 2. Edge length profile ----------------------------------------------
subplot(1,3,2);
plot(1:N, edgeL, '.-', 'MarkerSize', 5, 'LineWidth', 0.8);
yline(mean_edge, 'k--', sprintf('mean=%.2f mm', mean_edge));
xlabel('Edge index (k -> k+1)');
ylabel('Length [mm]');
title('Edge length profile');
grid on;
ylim([0, max(edgeL)*1.3]);

% --- 3. Approximate RoC preview (finite-difference curvature) ------------
subplot(1,3,3);
wrap = @(v) [v(end,:); v; v(1,:)];
nw = wrap(nodesCoords(:,1:2));
d1 = nw(3:end,:) - nw(1:end-2,:);
d2 = nw(3:end,:) - 2*nw(2:end-1,:) + nw(1:end-2,:);
cross_z = d1(:,1).*d2(:,2) - d1(:,2).*d2(:,1);
den     = (d1(:,1).^2 + d1(:,2).^2).^(3/2) + 1e-12;
kappa   = abs(cross_z) ./ den;
RoC_approx = min(max(1./kappa, 5), 1221.95);

semilogy(1:N, RoC_approx, '.-', 'MarkerSize', 5, 'LineWidth', 0.8);
hold on;
yline(5, 'r--', 'min clamp (5 mm)', 'FontSize', 7, ...
    'LabelHorizontalAlignment', 'left');
yline(1221.95, 'r--', 'flat transition', 'FontSize', 7, ...
    'LabelHorizontalAlignment', 'left');
xlabel('Node index');
ylabel('RoC [mm]  (log scale)');
title('Approx. RoC  (FD, not spline)');
grid on;
ylim([3, 3e3]);

sgtitle(sprintf('Ellipse: a=%.0f mm, b=%.0f mm  |  N=%d  |  perimeter=%.1f mm', ...
    a, b, N, perimeter_num));

fprintf('\nPlot rendered. Checks to confirm:\n');
fprintf('  - Nodes run CCW starting from the bottom point\n');
fprintf('  - Spacing is near-uniform because of arc-length inversion\n');
fprintf('  - Centroid is at the origin within numerical precision\n');
