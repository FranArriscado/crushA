% generate_circle.m
% =========================================================================
% Generates a circular cross-section and saves it as a .mat file ready to
% be loaded by the Python optimizer.
%
% Geometry:
%   - Circle of radius R centred at the origin
%   - Nodes are distributed uniformly in polar angle, which also gives
%     uniform arc-length spacing along the circle
%   - Ordering is CCW, starting at the bottom point (0, -R)
%
% Output .mat fields  (must match what loadData() expects):
%   nodesCoords   [N x 3]  double  - (x, y, 0) in mm, row = one node
%   connectivity  [N x 2]  double  - 1-based edge pairs, closed loop
%                                   edge k connects node k -> node k+1
%                                   last edge connects node N -> node 1
%
% Recommended node count: 80-160 for a smooth spline representation.
%
% Dependencies: none (core MATLAB only)
%
% Francisco Arriscado / FEUP / 2026
% =========================================================================

clear; clc; close all;

%% ======================== USER PARAMETERS ================================

R       = 75.3969;     % [mm] Circle radius
n_total = 100;    % Total node count around the circumference

outputFile = 'partition_data_circle.mat';

% Optional angular offset to rotate the starting point without changing
% shape or spacing. Default keeps node 1 at the bottom point.
theta0_deg = -90; % [deg]


%% ======================== INPUT VALIDATION ===============================

assert(R > 0,        'R must be positive.');
assert(n_total >= 8, 'n_total should be at least 8.');


%% ======================== NODE GENERATION ================================
% Build N equally spaced angular samples over one full revolution.
% We generate N+1 samples and drop the duplicate endpoint.

theta = linspace(deg2rad(theta0_deg), deg2rad(theta0_deg) + 2*pi, n_total + 1)';
theta = theta(1:end-1);

x = R * cos(theta);
y = R * sin(theta);
z = zeros(n_total, 1);

nodesCoords = [x, y, z];
N = size(nodesCoords, 1);

assert(N == n_total, ...
    'Node count mismatch: expected %d, got %d.', n_total, N);


%% ======================== CONNECTIVITY ===================================

connectivity = [(1:N)', [2:N, 1]'];


%% ======================== QUALITY CHECKS =================================

dx    = diff(nodesCoords([1:end, 1], 1));
dy    = diff(nodesCoords([1:end, 1], 2));
edgeL = sqrt(dx.^2 + dy.^2);

perimeter       = sum(edgeL);
expected_perim  = 2 * pi * R;
perim_err       = abs(perimeter - expected_perim);
mean_edge       = mean(edgeL);
std_edge        = std(edgeL);
cv_edge         = std_edge / mean_edge;

radii = sqrt(nodesCoords(:,1).^2 + nodesCoords(:,2).^2);
radius_mean = mean(radii);
radius_std  = std(radii);
centroid    = mean(nodesCoords(:,1:2), 1);

fprintf('\n--- Quality Report ---\n');
fprintf('Radius        : %.3f mm\n', R);
fprintf('Nodes         : %d\n', N);
fprintf('Perimeter     : %.4f mm  (expected %.4f mm, error %.2e mm)\n', ...
    perimeter, expected_perim, perim_err);
fprintf('Mean edge len : %.4f mm\n', mean_edge);
fprintf('Std  edge len : %.4f mm\n', std_edge);
fprintf('CV (std/mean) : %.4f     (should be near zero)\n', cv_edge);
fprintf('Mean radius   : %.4f mm\n', radius_mean);
fprintf('Std  radius   : %.4e mm  (should be ~0)\n', radius_std);
fprintf('Centroid      : (%.4e, %.4e) mm  (should be ~0, 0)\n', ...
    centroid(1), centroid(2));


%% ======================== SAVE ===========================================

save(outputFile, 'nodesCoords', 'connectivity');
fprintf('\nSaved -> %s\n', outputFile);
fprintf('Fields: nodesCoords [%d x 3], connectivity [%d x 2]\n', N, N);
fprintf('To use: set dataFileName = ''%s'' in the optimizer.\n', outputFile);


%% ======================== PLOTS ==========================================

figure('Name', 'Circle geometry check', 'Position', [100 100 1100 380]);

% --- 1. Shape with cardinal markers --------------------------------------
subplot(1,3,1);
closed = [nodesCoords(:,1:2); nodesCoords(1,1:2)];
plot(closed(:,1), closed(:,2), 'b-', 'LineWidth', 1.4);
hold on;
scatter(nodesCoords(:,1), nodesCoords(:,2), 18, 'b', 'filled');

card_idx = unique(round(linspace(1, N+1, 5)));
card_idx(card_idx > N) = 1;
scatter(nodesCoords(card_idx,1), nodesCoords(card_idx,2), ...
    60, 'r', 'filled', 'DisplayName', 'Quarter points');

step = max(1, floor(N/20));
for k = 1:step:N
    text(nodesCoords(k,1)*1.07, nodesCoords(k,2)*1.07, num2str(k), ...
        'FontSize', 6, 'Color', [0.4 0.4 0.4]);
end

axis equal; grid on;
title(sprintf('Circle  (N=%d, R=%g mm)', N, R));
xlabel('x [mm]'); ylabel('y [mm]');
legend({'Contour','Nodes','Quarter points'}, 'Location','eastoutside');

% --- 2. Edge length profile ----------------------------------------------
subplot(1,3,2);
plot(1:N, edgeL, '.-', 'MarkerSize', 5, 'LineWidth', 0.8);
hold on;
yline(mean_edge, 'k--', sprintf('mean=%.3f mm', mean_edge));
xlabel('Edge index (k -> k+1)');
ylabel('Length [mm]');
title('Edge length profile');
grid on;
ylim([0, max(edgeL)*1.3]);

% --- 3. Approximate RoC preview (finite-difference curvature) ------------
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
yline(R, 'g--', sprintf('expected R = %.1f mm', R));
yline(5,       'r:', 'min clamp (5)');
yline(1221.95, 'r:', 'flat transition');
xlabel('Node index');
ylabel('RoC [mm]  (log scale)');
title('Approximate RoC (FD, not spline)');
grid on;

sgtitle(sprintf('Circle mockup  -  perimeter=%.1f mm, N=%d', perimeter, N));

fprintf('\nPlot rendered. Check:\n');
fprintf('  - Shape is circular and centred at the origin\n');
fprintf('  - Edge-length profile is nearly flat\n');
fprintf('  - RoC stays near the input radius across the contour\n');
