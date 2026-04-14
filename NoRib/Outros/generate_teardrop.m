% generate_teardrop.m
% =========================================================================
% Generates a teardrop cross-section and saves it as a .mat file ready to
% be loaded by the Python optimizer (optimizationModule_patched_v8.py).
%
% Geometry — midplane of thin wall:
%
%            ┌──────── nose arc (radius R_n) ─────────┐
%           ╱                                           ╲
%  tail ───•                                             •─── rightmost nose
%  (sharp  ╲                                           ╱     point (+a+R_n−R_n, 0)
%   or blunt)└──────────────────────────────────────────┘
%
%   • Tail : sharp point at (−a, 0)  when R_t = 0
%            small circular arc, radius R_t, when R_t > 0
%   • Nose : circular arc, radius R_n, centred at (a − R_n, 0)
%   • Sides: two straight lines tangent to both the nose circle
%            and the tail circle (or passing through the sharp tail point)
%
%  The two straight sides are defined by the EXTERNAL tangent between the
%  nose circle (C_n, R_n) and the tail entity (C_t = point or C_t, R_t):
%
%    slope m = (R_n − R_t) / sqrt(d² − (R_n − R_t)²)
%    where d = dist(C_n, C_t)  [d = 2a − R_n for R_t = 0]
%
%  For R_t = 0 this reduces to the standard tangent-from-point formula.
%
% Optimizer compatibility:
%   ✓ Single closed loop  (all nodes degree 2)
%   ✓ Star-shaped from centroid — the teardrop with straight sides is
%     convex, so star-shaped from any interior point.  The script prints
%     the max angular gap test to confirm.
%   ✓ nodesCoords [N×3], connectivity [N×2], 1-based, standard format
%
% RoC behaviour (key physics interest of this geometry):
%   • Sharp tail (R_t = 0) : RoC → 0, clamped to 5 mm  → max crush stress
%   • Straight sides       : RoC → ∞, clamped to flat_transition → min crush stress
%   • Nose arc             : RoC = R_n (tunable intermediate value)
%
%   This creates the widest possible RoC range, making the optimizer
%   exploit or resist the sharp tail depending on the objective.
%
% Node ordering — CCW, "include-start exclude-end" on every segment:
%
%   R_t = 0  (3 segments, starts at sharp tail):
%     Seg 1  upper side  : tail  → upper nose tang pt    [n_us nodes]
%     Seg 2  nose arc    : upper → lower nose tang pt     [n_n  nodes]  CW around C_n
%     Seg 3  lower side  : lower nose tang pt → tail      [n_ls nodes]
%
%   R_t > 0  (4 segments, starts at lower tail tang pt):
%     Seg 1  tail arc    : lower tail tang → upper tail tang  [n_t  nodes]  CW around C_t
%     Seg 2  upper side  : upper tail tang → upper nose tang  [n_us nodes]
%     Seg 3  nose arc    : upper → lower nose tang            [n_n  nodes]  CW around C_n
%     Seg 4  lower side  : lower nose tang → lower tail tang  [n_ls nodes]
%
% Dependencies: none (core MATLAB only)
%
% Francisco Arriscado / FEUP / 2026
% =========================================================================

clear; clc; close all;

%% ======================== USER PARAMETERS ================================

a    = 40;    % [mm]  Half total length — tail at x = −a, nose protrudes to x ≈ +a
R_n  = 15;    % [mm]  Nose circle radius.  Constraint: R_n < a
R_t  = 0;     % [mm]  Tail radius.  0 = sharp point.
              %        If R_t > 0: small blunt tail arc (R_t << R_n typical).
              %        Constraint: R_n + R_t < 2*a  and  R_t < R_n

n_total = 100;  % Total nodes.  Rule of thumb: 80–140 for a well-resolved spline.

outputFile = 'partition_data_teardrop.mat';

% ── Optional manual node-count overrides (all 0 = auto arc-length split)
n_us_override = 0;   % upper side
n_n_override  = 0;   % nose arc
n_ls_override = 0;   % lower side  (= n_us for symmetric geometry)
n_t_override  = 0;   % tail arc  (only used when R_t > 0)


%% ======================== INPUT VALIDATION ================================

assert(a  > 0,          'a must be positive.');
assert(R_n > 0,         'R_n must be positive.');
assert(R_n < a,         'R_n must be < a (%.1f mm).', a);
assert(R_t >= 0,        'R_t must be non-negative.');
if R_t > 0
    assert(R_t < R_n,            'R_t must be < R_n (%.1f mm).', R_n);
    assert(R_n + R_t < 2*a,      'Circles overlap: need R_n + R_t < 2a.');
end
assert(n_total >= 6,    'n_total should be at least 6.');


%% ======================== CORE GEOMETRY ==================================
% All quantities derived analytically; validated against expected values below.

if R_t == 0
    %----------------------------------------------------------------------
    % Case 1: Sharp tail — tangent from external point to nose circle
    %----------------------------------------------------------------------

    C_n  = [a - R_n, 0];      % nose centre
    T    = [-a,      0];       % tail point

    d    = norm(C_n - T);      % = 2a − R_n  (distance tail → nose centre)
    assert(d > R_n, ...
        'Tail point is inside the nose circle. Reduce R_n or increase a.');

    beta    = asin(R_n / d);           % half-angle of tangent at tail
    L_side  = sqrt(d^2 - R_n^2);      % length of each straight side

    % Upper tangent direction from tail
    dir_upper = [cos(beta),  sin(beta)];
    dir_lower = [cos(beta), -sin(beta)];

    P_nose_tang_top = T + L_side * dir_upper;
    P_nose_tang_bot = T + L_side * dir_lower;

    % Angular position of tangent points on nose circle
    theta_n_top = atan2(P_nose_tang_top(2) - C_n(2), ...
                        P_nose_tang_top(1) - C_n(1));   % ∈ (π/2, π)  typical
    theta_n_bot = -theta_n_top;                          % by symmetry

    % Nose arc spans from theta_n_top → theta_n_bot going CW (decreasing θ)
    % Arc length: R_n × (2 × theta_n_top)   [theta_n_top in radians, > 0]
    L_nose  = R_n * 2 * theta_n_top;
    L_total = 2 * L_side + L_nose;

    % For reporting
    nose_arc_deg = rad2deg(2 * theta_n_top);
    fprintf('Sharp tail geometry:\n');
    fprintf('  Tail angle (2β)   : %.2f°\n',   rad2deg(2*beta));
    fprintf('  Side length       : %.3f mm\n',  L_side);
    fprintf('  Nose arc          : %.2f°  (%.3f mm)\n', nose_arc_deg, L_nose);
    fprintf('  Total perimeter   : %.3f mm\n',  L_total);
    fprintf('  Max half-width    : %.3f mm  at x = %.3f mm\n', ...
        P_nose_tang_top(2), P_nose_tang_top(1));

else
    %----------------------------------------------------------------------
    % Case 2: Blunt tail — external tangent between tail circle and nose circle
    %----------------------------------------------------------------------

    C_t = [-a + R_t, 0];      % tail circle centre
    C_n = [ a - R_n, 0];      % nose circle centre
    T   = [-a, 0];             % leftmost point of tail arc

    d = norm(C_n - C_t);       % = 2a − R_n − R_t  (centre-to-centre distance)
    assert(d > abs(R_n - R_t), ...
        'Circles are nested (d ≤ |R_n − R_t|). Increase a or reduce radii.');

    % External tangent slope: m = (R_n − R_t) / sqrt(d² − (R_n − R_t)²)
    %   Note: "external" means both circles lie on the SAME side of the tangent
    %   line (i.e., the tangent does not pass between them).
    num_m   = R_n - R_t;
    den_m   = sqrt(d^2 - num_m^2);
    m       = num_m / den_m;           % slope of upper tangent line

    % Unit normal pointing toward the upper side (perpendicular to tangent, upward)
    tang_dir  = [1, m] / sqrt(1 + m^2);   % unit tangent direction
    norm_up   = [-m, 1] / sqrt(1 + m^2);  % unit normal pointing up

    % Tangent points on each circle (external tangent, upper side)
    P_nose_tang_top = C_n + R_n * norm_up;
    P_nose_tang_bot = C_n - R_n * norm_up;
    P_tail_tang_top = C_t + R_t * norm_up;
    P_tail_tang_bot = C_t - R_t * norm_up;

    % Angular positions on nose circle
    theta_n_top = atan2(P_nose_tang_top(2) - C_n(2), P_nose_tang_top(1) - C_n(1));
    theta_n_bot = -theta_n_top;   % by symmetry

    % Angular positions on tail circle
    % Tail arc goes from P_tail_tang_bot to P_tail_tang_top CW through tip (−a,0)
    theta_t_top = atan2(P_tail_tang_top(2) - C_t(2), P_tail_tang_top(1) - C_t(1));
    % theta_t_top ∈ (π/2, π): the upper tangent point is in the left-upper quadrant

    % Segment arc lengths
    L_nose  = R_n * 2 * theta_n_top;         % nose arc
    L_tail  = R_t * (2*pi - 2*theta_t_top);  % tail arc going CW through −a
    L_side  = norm(P_nose_tang_top - P_tail_tang_top);  % each straight side
    L_total = 2 * L_side + L_nose + L_tail;

    fprintf('Blunt tail geometry (R_t = %.1f mm):\n', R_t);
    fprintf('  Side length       : %.3f mm\n', L_side);
    fprintf('  Nose arc          : %.2f°  (%.3f mm)\n', rad2deg(2*theta_n_top), L_nose);
    fprintf('  Tail arc          : %.2f°  (%.3f mm)\n', rad2deg(2*pi-2*theta_t_top), L_tail);
    fprintf('  Total perimeter   : %.3f mm\n', L_total);
    fprintf('  Max half-width    : %.3f mm  at x = %.3f mm\n', ...
        P_nose_tang_top(2), P_nose_tang_top(1));
end


%% ======================== NODE COUNT PER SEGMENT =========================

if R_t == 0
    if n_us_override > 0 && n_n_override > 0 && n_ls_override > 0
        n_us = n_us_override;
        n_n  = n_n_override;
        n_ls = n_ls_override;
        n_total = n_us + n_n + n_ls;
        fprintf('\nManual split: n_us=%d, n_n=%d, n_ls=%d  (total=%d)\n', ...
            n_us, n_n, n_ls, n_total);
    else
        n_us = max(3, round(n_total * L_side / L_total));
        n_n  = max(3, round(n_total * L_nose / L_total));
        % Absorb rounding error into the nose (largest arc)
        n_ls  = n_us;   % symmetric
        delta = n_total - (n_us + n_n + n_ls);
        n_n   = max(3, n_n + delta);
        n_total = n_us + n_n + n_ls;
        fprintf('\nArc-length split: n_us=%d, n_n=%d, n_ls=%d  (total=%d)\n', ...
            n_us, n_n, n_ls, n_total);
    end
else
    if n_us_override > 0 && n_n_override > 0 && n_ls_override > 0 && n_t_override > 0
        n_t  = n_t_override;
        n_us = n_us_override;
        n_n  = n_n_override;
        n_ls = n_ls_override;
        n_total = n_t + n_us + n_n + n_ls;
        fprintf('\nManual split: n_t=%d, n_us=%d, n_n=%d, n_ls=%d  (total=%d)\n', ...
            n_t, n_us, n_n, n_ls, n_total);
    else
        n_t  = max(3, round(n_total * L_tail / L_total));
        n_us = max(3, round(n_total * L_side / L_total));
        n_n  = max(3, round(n_total * L_nose / L_total));
        n_ls = n_us;   % symmetric
        delta = n_total - (n_t + n_us + n_n + n_ls);
        n_n   = max(3, n_n + delta);
        n_total = n_t + n_us + n_n + n_ls;
        fprintf('\nArc-length split: n_t=%d, n_us=%d, n_n=%d, n_ls=%d  (total=%d)\n', ...
            n_t, n_us, n_n, n_ls, n_total);
    end
end


%% ======================== NODE GENERATION ================================
% Convention: include-start, exclude-end on every segment except the last
% node of the final segment (which is also the start of segment 1, handled
% by the closing edge in connectivity — NOT stored as a duplicate node).

trim = @(pts) pts(1:end-1, :);   % drop last row

if R_t == 0
    % ------------------------------------------------------------------
    % Seg 1: upper side  T → P_nose_tang_top  (straight, n_us nodes)
    t_us  = linspace(0, 1, n_us + 1)';
    seg1  = trim( T + t_us * (P_nose_tang_top - T) );

    % Seg 2: nose arc  theta_n_top → theta_n_bot  CW (decreasing θ), n_n nodes
    %   "CW around C_n" = outer boundary traversal, CCW overall contour
    theta_nose = linspace(theta_n_top, theta_n_bot, n_n + 1)';
    seg2 = trim( [C_n(1) + R_n*cos(theta_nose), C_n(2) + R_n*sin(theta_nose)] );

    % Seg 3: lower side  P_nose_tang_bot → T  (straight, n_ls nodes)
    %   Final segment: includes its last point implicitly via closing edge,
    %   so trim the endpoint (= T = node 1) to avoid duplication.
    t_ls  = linspace(0, 1, n_ls + 1)';
    seg3  = trim( P_nose_tang_bot + t_ls * (T - P_nose_tang_bot) );

    xy = [seg1; seg2; seg3];

else
    % ------------------------------------------------------------------
    % Seg 1: tail arc  P_tail_tang_bot → P_tail_tang_top  CW, n_t nodes
    %   CW around C_t passing through leftmost point (tip at −a)
    %   Angles: from −theta_t_top going CW (decreasing) through −π to +theta_t_top
    %   In practice: linspace(−theta_t_top, theta_t_top − 2π, n_t+1)
    %   At midpoint angle ≈ −π → (C_t.x − R_t, 0) = (−a, 0)  ✓
    theta_tail_start = -theta_t_top;                    % angle of P_tail_tang_bot
    theta_tail_end   =  theta_t_top - 2*pi;             % same angle, one CW revolution
    theta_tail = linspace(theta_tail_start, theta_tail_end, n_t + 1)';
    seg1 = trim( [C_t(1) + R_t*cos(theta_tail), C_t(2) + R_t*sin(theta_tail)] );

    % Seg 2: upper side  P_tail_tang_top → P_nose_tang_top  (straight, n_us)
    t_us  = linspace(0, 1, n_us + 1)';
    seg2  = trim( P_tail_tang_top + t_us * (P_nose_tang_top - P_tail_tang_top) );

    % Seg 3: nose arc  theta_n_top → theta_n_bot  CW, n_n nodes
    theta_nose = linspace(theta_n_top, theta_n_bot, n_n + 1)';
    seg3 = trim( [C_n(1) + R_n*cos(theta_nose), C_n(2) + R_n*sin(theta_nose)] );

    % Seg 4: lower side  P_nose_tang_bot → P_tail_tang_bot  (straight, n_ls)
    t_ls  = linspace(0, 1, n_ls + 1)';
    seg4  = trim( P_nose_tang_bot + t_ls * (P_tail_tang_bot - P_nose_tang_bot) );

    xy = [seg1; seg2; seg3; seg4];
end

% Append Z = 0
nodesCoords = [xy, zeros(size(xy,1), 1)];
N = size(nodesCoords, 1);

assert(N == n_total, ...
    'Node count mismatch: expected %d, assembled %d.', n_total, N);


%% ======================== CONNECTIVITY ===================================

connectivity = [(1:N)', [2:N, 1]'];   % closed loop, 1-based


%% ======================== QUALITY CHECKS =================================

dx    = diff(nodesCoords([1:end,1], 1));
dy    = diff(nodesCoords([1:end,1], 2));
edgeL = sqrt(dx.^2 + dy.^2);

perimeter = sum(edgeL);
mean_edge = mean(edgeL);
cv_edge   = std(edgeL) / mean_edge;

% Star-shaped: no angular gap > 180° from centroid
centroid  = mean(nodesCoords(:,1:2));
phi_deg   = atan2d(nodesCoords(:,2) - centroid(2), ...
                   nodesCoords(:,1) - centroid(1));
phi_s     = sort(phi_deg);
max_gap   = max(diff([phi_s; phi_s(1)+360]));

% Approximate RoC (finite-difference, for the plot)
nc_pad  = [nodesCoords(end-1:end,1:2); nodesCoords(:,1:2); nodesCoords(1:2,1:2)];
d1      = nc_pad(3:end,:)   - nc_pad(1:end-2,:);
d2      = nc_pad(3:end,:) - 2*nc_pad(2:end-1,:) + nc_pad(1:end-2,:);
kappa   = abs(d1(:,1).*d2(:,2) - d1(:,2).*d2(:,1)) ./ ...
          ((d1(:,1).^2 + d1(:,2).^2).^(3/2) + 1e-12);
RoC_fd  = min(max(1./kappa, 5), 1221.95);

fprintf('\n--- Quality Report ---\n');
fprintf('Nodes            : %d\n', N);
fprintf('Perimeter (nodes): %.4f mm   (expected %.4f mm)\n', perimeter, L_total);
fprintf('Mean edge length : %.4f mm\n', mean_edge);
fprintf('CV (std/mean)    : %.4f     (< 0.3 recommended)\n', cv_edge);
fprintf('Centroid         : (%.4f, %.4f) mm\n', centroid(1), centroid(2));
fprintf('Max angular gap  : %.2f°   (< 180° → star-shaped OK)\n', max_gap);
fprintf('Min FD RoC       : %.2f mm  (should be ~5 near tail)\n', min(RoC_fd));
fprintf('Max FD RoC       : %.2f mm  (should be flat_transition near sides)\n', max(RoC_fd));

% Junction continuity checks
if R_t == 0
    gap_tail_start = norm(nodesCoords(1,1:2) - T);
    gap_tang_top   = norm(nodesCoords(n_us,1:2) - P_nose_tang_top);
    gap_tang_bot   = norm(nodesCoords(n_us+n_n,1:2) - P_nose_tang_bot);
    fprintf('\nJunction gaps:\n');
    fprintf('  Tail  (node 1)        : %.2e mm\n', gap_tail_start);
    fprintf('  Upper tang (node %3d) : %.2e mm\n', n_us,       gap_tang_top);
    fprintf('  Lower tang (node %3d) : %.2e mm\n', n_us+n_n,   gap_tang_bot);
end


%% ======================== SAVE ===========================================

save(outputFile, 'nodesCoords', 'connectivity');
fprintf('\nSaved → %s\n', outputFile);
fprintf('Fields: nodesCoords [%d×3], connectivity [%d×2]\n', N, N);
fprintf('To use: set dataFileName = ''%s'' in the optimizer.\n', outputFile);


%% ======================== PLOTS ==========================================

figure('Name', 'Teardrop geometry check', 'Position', [60 60 1300 900]);

% Colour per segment
if R_t == 0
    seg_n    = [n_us,  n_n,  n_ls];
    seg_cols = {[0.12 0.47 0.71], [0.84 0.15 0.16], [0.12 0.47 0.71]};
    seg_names = {'Upper side', 'Nose arc', 'Lower side'};
else
    seg_n    = [n_t,   n_us,  n_n,  n_ls];
    seg_cols = {[0.49 0.18 0.56], [0.12 0.47 0.71], ...
                [0.84 0.15 0.16], [0.12 0.47 0.71]};
    seg_names = {'Tail arc', 'Upper side', 'Nose arc', 'Lower side'};
end
n_segs = numel(seg_n);

% ── Panel 1: Shape --------------------------------------------------------
subplot(2,2,[1 3]);
hold on; axis equal; grid on;

cumN = 0;
for k = 1:n_segs
    nk  = seg_n(k);
    idx = cumN + (1:nk);
    nxt = mod(cumN + nk, N) + 1;
    plot(nodesCoords([idx, nxt], 1), nodesCoords([idx, nxt], 2), ...
        '-', 'Color', seg_cols{k}, 'LineWidth', 2.0, 'DisplayName', seg_names{k});
    scatter(nodesCoords(idx, 1), nodesCoords(idx, 2), ...
        16, seg_cols{k}, 'filled', 'HandleVisibility', 'off');
    cumN = cumN + nk;
end

% Mark key geometry
plot(C_n(1), C_n(2), 'r+', 'MarkerSize', 8, 'LineWidth', 1.2, 'DisplayName', 'Nose centre');
if R_t == 0
    scatter(T(1), T(2), 80, 'k', 'pentagram', 'filled', 'DisplayName', 'Tail (sharp)');
else
    plot(C_t(1), C_t(2), 'm+', 'MarkerSize', 8, 'LineWidth', 1.2, 'DisplayName', 'Tail centre');
end
plot(centroid(1), centroid(2), 'k+', 'MarkerSize', 10, 'LineWidth', 1.5, 'DisplayName', 'Centroid');

% Draw nose circle faintly
theta_circ = linspace(0, 2*pi, 200);
plot(C_n(1) + R_n*cos(theta_circ), C_n(2) + R_n*sin(theta_circ), ...
    ':', 'Color', [0.8 0.4 0.4], 'LineWidth', 0.8, 'DisplayName', 'Nose circle');
if R_t > 0
    plot(C_t(1) + R_t*cos(theta_circ), C_t(2) + R_t*sin(theta_circ), ...
        ':', 'Color', [0.7 0.3 0.7], 'LineWidth', 0.8, 'DisplayName', 'Tail circle');
end

% Node index labels every ~n/20
step = max(1, floor(N/20));
for k = 1:step:N
    text(nodesCoords(k,1)*1.06 + 0.3, nodesCoords(k,2)*1.06, num2str(k), ...
        'FontSize', 6, 'Color', [0.45 0.45 0.45]);
end

% Dimension annotations
y_ann = max(nodesCoords(:,2)) * 1.35;
plot([-a, a-0], [y_ann, y_ann], 'k-', 'LineWidth', 0.8, 'HandleVisibility', 'off');
text(mean([-a, a]), y_ann + 1, sprintf('2a = %.0f mm', 2*a), ...
    'FontSize', 7, 'HorizontalAlignment', 'center');

legend('Location', 'northeast', 'FontSize', 7);
xlabel('x [mm]'); ylabel('y [mm]');
title(sprintf('Teardrop  N=%d   a=%.0f  R_n=%.0f  R_t=%.0f mm', N, a, R_n, R_t));


% ── Panel 2: Edge length profile ------------------------------------------
subplot(2,2,2);
plot(1:N, edgeL, '.-', 'MarkerSize', 5, 'Color', [0.2 0.2 0.2]);
hold on;
cumN = 0;
for k = 1:n_segs
    nk = seg_n(k);
    patch([cumN+1, cumN+nk, cumN+nk, cumN+1], ...
          [0, 0, max(edgeL)*1.35, max(edgeL)*1.35], ...
          seg_cols{k}, 'FaceAlpha', 0.12, 'EdgeColor', 'none');
    xline(cumN + 0.5, ':', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.6);
    cumN = cumN + nk;
end
yline(mean_edge, 'k--', sprintf('mean = %.2f mm', mean_edge), ...
    'LabelHorizontalAlignment', 'left', 'FontSize', 7);
xlabel('Edge index'); ylabel('Length [mm]');
title('Edge length profile'); grid on;
ylim([0, max(edgeL)*1.35]); xlim([1, N]);


% ── Panel 3: Approximate RoC (FD) -----------------------------------------
subplot(2,2,4);
semilogy(1:N, RoC_fd, '.-', 'MarkerSize', 5, 'Color', [0.2 0.2 0.2]);
hold on;
cumN = 0;
for k = 1:n_segs
    nk = seg_n(k);
    patch([cumN+1, cumN+nk, cumN+nk, cumN+1], [1, 1, 3e3, 3e3], ...
          seg_cols{k}, 'FaceAlpha', 0.12, 'EdgeColor', 'none');
    xline(cumN + 0.5, ':', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.6);
    cumN = cumN + nk;
end
yline(5,       'r--', 'min clamp (5 mm)',    'FontSize', 7, 'LabelHorizontalAlignment', 'left');
yline(1221.95, 'r--', 'flat transition',     'FontSize', 7, 'LabelHorizontalAlignment', 'left');
yline(R_n,     'g--', sprintf('R_n = %.0f mm', R_n), 'FontSize', 7, 'LabelHorizontalAlignment', 'right');
if R_t > 0
    yline(R_t, 'm--', sprintf('R_t = %.0f mm', R_t), 'FontSize', 7);
end
xlabel('Node index'); ylabel('RoC [mm]  (log scale, FD approx.)');
title('Approx. RoC — FD only, not spline'); grid on;
ylim([3, 3e3]); xlim([1, N]);

% Expected pattern annotation
if R_t == 0
    n_mid_side_upper = round(n_us / 2);
    n_mid_side_lower = n_us + n_n + round(n_ls / 2);
    text(n_mid_side_upper, 1221.95 * 0.35, 'sides → flat', ...
        'FontSize', 6, 'HorizontalAlignment', 'center', 'Color', [0.3 0.6 0.3]);
    text(n_us + round(n_n/2), R_n * 2.5, 'nose arc', ...
        'FontSize', 6, 'HorizontalAlignment', 'center', 'Color', [0.7 0.2 0.2]);
end

sgtitle(sprintf('Teardrop: a=%.0f  R_n=%.0f  R_t=%.0f mm  |  N=%d  |  perimeter=%.1f mm  |  star-shaped: %.1f° max gap', ...
    a, R_n, R_t, N, perimeter, max_gap));

fprintf('\nExpected RoC pattern (left→right = node 1→N):\n');
if R_t == 0
    fprintf('  Nodes 1 → %d      : sides  → clamped at flat_transition (~1222 mm)\n', n_us);
    fprintf('  Nodes %d → %d  : nose arc → R_n = %.0f mm\n', n_us+1, n_us+n_n, R_n);
    fprintf('  Nodes %d → %d  : sides  → clamped at flat_transition\n', n_us+n_n+1, N);
    fprintf('  Nodes near 1 and %d: tail → clamped at 5 mm (sharp corner)\n', N);
end
