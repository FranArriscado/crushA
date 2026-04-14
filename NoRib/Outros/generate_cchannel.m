% generate_cchannel.m
% =========================================================================
% Generates a C-channel cross-section and saves it as a .mat file.
%
% ╔══════════════════════════════════════════════════════════════════════╗
% ║  OPTIMIZER COMPATIBILITY — READ BEFORE USE                         ║
% ║                                                                    ║
% ║  A C-channel is an OPEN section. The current optimizer             ║
% ║  (optimizationModule_patched_v8.py) has 5 closed-loop assumptions  ║
% ║  that are INCOMPATIBLE with open-chain connectivity:               ║
% ║                                                                    ║
% ║  (1) orderNodes(), line 133                                        ║
% ║      orderedIndices = ordered[:-1] drops the last node.            ║
% ║      For a closed loop the last entry is a start-node duplicate    ║
% ║      so this is correct. For an open chain the last entry is the   ║
% ║      actual second flange tip — dropping it silently loses a node. ║
% ║                                                                    ║
% ║  (2) nodesCoords()                                                 ║
% ║      Builds the spline overlap by appending X[-overlapCount:] at   ║
% ║      the start and X[:overlapCount] at the end — circular wrapping ║
% ║      that splices the two free flange tips together. The spline    ║
% ║      sees a false junction and assigns wrong curvature to both     ║
% ║      end nodes.                                                    ║
% ║                                                                    ║
% ║  (3) perimeterCalc()                                               ║
% ║      Uses torch.roll(orderedNodes,-1) to shift nodes by one step,  ║
% ║      creating a phantom edge from the last node (top flange tip)   ║
% ║      back to the first (bottom flange tip). Perimeter and node-    ║
% ║      length contributions at both tips are wrong.                  ║
% ║                                                                    ║
% ║  (4) smoothnessTerm()                                              ║
% ║      r_wrapped = cat([r[-1:], r, r[:1]]) enforces circular         ║
% ║      smoothness, penalising the angular gap between the two free   ║
% ║      tips as if they were neighbours on the path.                  ║
% ║                                                                    ║
% ║  (5) Cartesian2Spherical — star-shaped assumption                  ║
% ║      The optimizer fixes (theta, phi) per node and only optimises  ║
% ║      r (radial distance from centroid). This requires the section  ║
% ║      to be star-shaped from the centroid — every ray hits the wall ║
% ║      exactly once. The C-channel centroid sits inside the "C" gap; ║
% ║      rays toward the open side never reach the wall, making the    ║
% ║      parameterisation degenerate for those nodes.                  ║
% ║                                                                    ║
% ║  RESULT: feeding the open .mat into the current optimizer will     ║
% ║  silently produce incorrect geometry, forces, and gradients.       ║
% ║  The file is saved for future use once the optimizer is extended.  ║
% ║                                                                    ║
% ║  WORKAROUND (usable today):                                        ║
% ║  Set saveClosedVariant = true. A closed box section is generated   ║
% ║  by adding a back wall between the two flange tips. This is fully  ║
% ║  compatible with the current optimizer and physically represents   ║
% ║  a closed C-channel (common in crashworthiness structures where    ║
% ║  the C is bonded or riveted to a backing plate).                   ║
% ╚══════════════════════════════════════════════════════════════════════╝
%
% Geometry — midplane of thin wall, web on left, flanges pointing right:
%
%           ←── flange_w ───→
%           ┌────────────────  ← top flange tip     (segment 3 end)
%           │                  ← top flange
%   ────────┤                  ← top corner         (TL junction)
%           │ web (x = 0)
%   ────────┤                  ← bottom corner      (BL junction)
%           │                  ← bottom flange
%           └────────────────  ← bottom flange tip  (segment 1 start)
%
% Node ordering — CCW from bottom flange tip:
%   Seg 1  bottom flange : (W, −H/2) → (r_c, −H/2)          rightward→left
%  [Arc 2  bottom corner : arc r_c, θ: −π/2 → −π             CW, 90°]
%   Seg 3  web           : (0, −H/2+r_c) → (0, +H/2−r_c)    upward
%  [Arc 4  top corner    : arc r_c, θ: +π → +π/2             CW, 90°]
%   Seg 5  top flange    : (r_c, +H/2) → (W, +H/2)           leftward→right
%   (Corner arcs active only when r_c > 0)
%
%   Open chain: N nodes, N-1 edges — NO closing edge.
%
% Output .mat fields:
%   nodesCoords   [N×3]    double  — (x,y,0) mm, centroid-shifted
%   connectivity  [N-1×2]  double  — 1-based, open chain (NO closing edge)
%
%   Closed variant (saveClosedVariant = true):
%   nodesCoords_closed   [(N+n_bw)×3]
%   connectivity_closed  [(N+n_bw)×2]  — standard closed loop
%
% Dependencies: none (core MATLAB only)
%
% Francisco Arriscado / FEUP / 2026
% =========================================================================

clear; clc; close all;

%% ======================== USER PARAMETERS ================================

W         = 40;    % [mm]  Flange width (how far each flange extends from web)
H         = 60;    % [mm]  Total height (outer distance between flange midlines)
r_c       = 0;     % [mm]  Corner fillet radius  (0 = sharp corners)
                   %        If r_c > 0: must satisfy r_c < min(W, H/2)

n_total   = 100;   % Total node count across all segments
                   % Rule of thumb: 80–140 nodes

outputFile       = 'partition_data_cchannel.mat';
saveClosedVariant = true;   % Also save a back-wall-closed version
outputFileClosed  = 'partition_data_cchannel_closed.mat';

% Closed variant only: nodes on the back wall (tip-to-tip closing segment)
n_bw_closed = 20;   % Must be ≥ 2


%% ======================== INPUT VALIDATION ================================

assert(W > 0 && H > 0,  'W and H must be positive.');
assert(r_c >= 0,         'r_c must be non-negative.');
if r_c > 0
    assert(r_c < W,      'r_c must be < W (%.1f mm).', W);
    assert(r_c < H/2,    'r_c must be < H/2 (%.1f mm).', H/2);
end
assert(n_total >= 6,     'n_total should be at least 6.');


%% ======================== SEGMENT LENGTHS ================================

if r_c == 0
    % 3 straight segments
    L_bf  = W;        % bottom flange
    L_w   = H;        % web
    L_tf  = W;        % top flange
    L_c   = 0;        % no corner arcs
    n_seg = 3;
else
    % 3 straight + 2 corner arcs = 5 segments
    L_bf  = W - r_c;          % bottom flange (tip to corner tangent)
    L_c   = (pi/2) * r_c;     % each corner arc
    L_w   = H - 2*r_c;        % web (tangent to tangent)
    L_tf  = W - r_c;          % top flange (corner tangent to tip)
    n_seg = 5;
end

L_total = 2*L_bf + L_w + 2*L_c;   % total open-chain length


%% ======================== NODE COUNT PER SEGMENT =========================
% Proportional to arc length; minimum 2 nodes per segment.

if r_c == 0
    n_bf = max(2, round(n_total * L_bf / L_total));
    n_w  = max(2, round(n_total * L_w  / L_total));
    n_tf = max(2, round(n_total * L_tf / L_total));

    % Absorb rounding error into the web (longest segment)
    delta = n_total - (n_bf + n_w + n_tf);
    n_w   = max(2, n_w + delta);
    n_total = n_bf + n_w + n_tf;

    fprintf('Node split (r_c=0): n_bf=%d, n_w=%d, n_tf=%d  (total=%d)\n', ...
        n_bf, n_w, n_tf, n_total);
else
    n_bf = max(2, round(n_total * L_bf / L_total));
    n_c  = max(2, round(n_total * L_c  / L_total));
    n_w  = max(2, round(n_total * L_w  / L_total));
    n_tf = max(2, round(n_total * L_tf / L_total));

    delta = n_total - (n_bf + 2*n_c + n_w + n_tf);
    n_w   = max(2, n_w + delta);
    n_total = n_bf + 2*n_c + n_w + n_tf;

    fprintf('Node split (r_c=%.1f): n_bf=%d, n_c=%d, n_w=%d, n_tf=%d  (total=%d)\n', ...
        r_c, n_bf, n_c, n_w, n_tf, n_total);
end


%% ======================== NODE GENERATION ================================
% Convention: "include start, exclude end" — each segment contributes
% exactly n nodes starting from its first point, stopping one step before
% its last point. This avoids duplicate junction nodes when concatenating.
%
% EXCEPTION: the last segment (top flange) INCLUDES its endpoint (the tip)
% because this is an open chain — the tip is a genuine boundary node.

trim = @(pts) pts(1:end-1, :);   % drop last row (used for all but final segment)

if r_c == 0
    % ── Seg 1: bottom flange  (W, -H/2) → (0, -H/2)  left
    y_bf = -H/2;
    x1   = linspace(W, 0, n_bf + 1)';
    seg1 = trim([x1, repmat(y_bf, n_bf+1, 1)]);

    % ── Seg 2: web  (0, -H/2) → (0, +H/2)  upward
    x_w  = 0;
    y2   = linspace(-H/2, H/2, n_w + 1)';
    seg2 = trim([repmat(x_w, n_w+1, 1), y2]);

    % ── Seg 3: top flange  (0, +H/2) → (W, +H/2)  rightward  [KEEP endpoint]
    y_tf = H/2;
    x3   = linspace(0, W, n_tf + 1)';
    seg3 = [x3, repmat(y_tf, n_tf+1, 1)];   % full n_tf+1 rows, then trim below
    seg3 = seg3(1:n_tf, :);                  % n_tf nodes, endpoint = (W,H/2)

    % The true endpoint must be included for the open chain
    % so we add it explicitly:
    tip_top = [W, H/2];

    xy = [seg1; seg2; seg3; tip_top];

else
    % ── Seg 1: bottom flange  (W, -H/2) → (r_c, -H/2)
    x1   = linspace(W, r_c, n_bf + 1)';
    seg1 = trim([x1, repmat(-H/2, n_bf+1, 1)]);

    % ── Arc 2: bottom-left corner  centre (r_c, -H/2+r_c),  θ: -π/2 → -π  (CW)
    cx2 = r_c;  cy2 = -H/2 + r_c;
    th2 = linspace(-pi/2, -pi, n_c + 1)';
    seg2 = trim([cx2 + r_c*cos(th2), cy2 + r_c*sin(th2)]);

    % ── Seg 3: web  (0, -H/2+r_c) → (0, +H/2-r_c)
    y3   = linspace(-H/2+r_c, H/2-r_c, n_w + 1)';
    seg3 = trim([zeros(n_w+1, 1), y3]);

    % ── Arc 4: top-left corner  centre (r_c, +H/2-r_c),  θ: π → π/2  (CW)
    cx4 = r_c;  cy4 = H/2 - r_c;
    th4 = linspace(pi, pi/2, n_c + 1)';
    seg4 = trim([cx4 + r_c*cos(th4), cy4 + r_c*sin(th4)]);

    % ── Seg 5: top flange  (r_c, +H/2) → (W, +H/2)  [n_tf nodes + tip endpoint]
    x5   = linspace(r_c, W, n_tf + 1)';
    seg5 = [x5, repmat(H/2, n_tf+1, 1)];
    seg5 = seg5(1:n_tf, :);

    tip_top = [W, H/2];   % explicit tip endpoint

    xy = [seg1; seg2; seg3; seg4; seg5; tip_top];
end

% ── Append Z = 0
nodesCoords_raw = [xy, zeros(size(xy,1), 1)];
N_open = size(nodesCoords_raw, 1);


%% ======================== CENTROID SHIFT =================================
% Shift so that mean(nodesCoords) = (0, 0, 0).
% This is the same centroid computation the optimizer does internally, so
% pre-centering has no effect on the optimisation — it only improves
% readability of the geometry in plots and makes the spherical angles
% more interpretable for diagnostic purposes.

centroid_xy  = mean(nodesCoords_raw(:, 1:2), 1);
nodesCoords  = nodesCoords_raw;
nodesCoords(:, 1) = nodesCoords_raw(:, 1) - centroid_xy(1);
nodesCoords(:, 2) = nodesCoords_raw(:, 2) - centroid_xy(2);

fprintf('Centroid shift: (%.4f, %.4f) mm applied\n', ...
    centroid_xy(1), centroid_xy(2));
fprintf('Post-shift centroid: (%.2e, %.2e)\n', ...
    mean(nodesCoords(:,1)), mean(nodesCoords(:,2)));


%% ======================== CONNECTIVITY — OPEN CHAIN =====================
% N-1 edges, NO closing edge from node N to node 1.
% This is what distinguishes an open section from a closed loop.

connectivity = [(1:N_open-1)', (2:N_open)'];   % [N-1 × 2]


%% ======================== QUALITY CHECKS — OPEN CHAIN ===================

dx    = diff(nodesCoords(:, 1));
dy    = diff(nodesCoords(:, 2));
edgeL = sqrt(dx.^2 + dy.^2);   % N-1 edge lengths (no closing edge)

perimeter   = sum(edgeL);       % open-chain arc length
mean_edge   = mean(edgeL);
std_edge    = std(edgeL);
cv_edge     = std_edge / mean_edge;

fprintf('\n--- Quality Report (open chain) ---\n');
fprintf('Nodes         : %d\n', N_open);
fprintf('Edges         : %d  (N-1, open chain)\n', N_open-1);
fprintf('Arc length    : %.4f mm  (expected %.4f mm)\n', perimeter, L_total);
fprintf('Mean edge len : %.4f mm\n', mean_edge);
fprintf('CV (std/mean) : %.4f     (< 0.3 recommended)\n', cv_edge);

% ── Junction continuity
if r_c == 0
    gap_bl = norm(nodesCoords(n_bf,     1:2) - [-centroid_xy(1), -H/2]);
    gap_tl = norm(nodesCoords(n_bf+n_w, 1:2) - [-centroid_xy(1), +H/2]);
    fprintf('Web-flange junction gap (bottom): %.2e mm\n', gap_bl);
    fprintf('Web-flange junction gap (top)   : %.2e mm\n', gap_tl);
end


%% ======================== SAVE — OPEN CHAIN ==============================

save(outputFile, 'nodesCoords', 'connectivity');
fprintf('\n[OPEN CHAIN] Saved → %s\n', outputFile);
fprintf('  nodesCoords  [%d × 3]\n', N_open);
fprintf('  connectivity [%d × 2]  (N-1 edges, no closing edge)\n', N_open-1);
fprintf('\n*** WARNING: incompatible with current optimizer (see header). ***\n');


%% ======================== CLOSED VARIANT =================================
% Adds a back wall from top flange tip → bottom flange tip.
% The resulting section is a closed box — fully compatible with the
% current optimizer (closed-loop connectivity, star-shaped from centroid).
%
% Physical interpretation: a C-channel backed by a flat plate, which is
% a common configuration in sidepod structural ribs and crashworthiness
% test jigs.

if saveClosedVariant

    % Back wall: (W, +H/2) → (W, -H/2)  downward, after centroid shift
    x_bw = W - centroid_xy(1);   % back wall x-position after shift
    y_bw = linspace(H/2, -H/2, n_bw_closed + 1)';
    bw   = [repmat(x_bw, n_bw_closed+1, 1), y_bw, zeros(n_bw_closed+1, 1)];
    bw   = bw(1:n_bw_closed, :);   % trim endpoint (= node 1 of open chain)

    nodesCoords_closed = [nodesCoords; bw];
    N_closed = size(nodesCoords_closed, 1);

    % Standard closed-loop connectivity
    connectivity_closed = [(1:N_closed)', [2:N_closed, 1]'];

    save(outputFileClosed, ...
        'nodesCoords',   'nodesCoords_closed', ...
        'connectivity',  'connectivity_closed');

    fprintf('\n[CLOSED VARIANT] Saved → %s\n', outputFileClosed);
    fprintf('  nodesCoords_closed  [%d × 3]  (open %d + back wall %d)\n', ...
        N_closed, N_open, n_bw_closed);
    fprintf('  connectivity_closed [%d × 2]  (closed loop)\n', N_closed);
    fprintf('  To use: load nodesCoords_closed and connectivity_closed\n');
    fprintf('          set dataFileName = ''%s'' in the optimizer\n', outputFileClosed);
    fprintf('          (load these fields directly — loadData() reads nodesCoords/connectivity)\n');
    fprintf('\n  NOTE: rename the closed fields before passing to optimizer:\n');
    fprintf('    data.nodesCoords  = data.nodesCoords_closed;\n');
    fprintf('    data.connectivity = data.connectivity_closed;\n');

end


%% ======================== PLOTS ==========================================

% ── Segment colours
col_flange = [0.12 0.47 0.71];   % blue
col_web    = [0.84 0.15 0.16];   % red
col_corner = [0.49 0.18 0.56];   % purple
col_bw     = [0.47 0.67 0.19];   % green (back wall)

figure('Name', 'C-channel geometry check', 'Position', [60 60 1300 900]);

% --- Panel 1: Shape comparison (open vs closed) ---------------------------
subplot(2,3,[1 4]);
hold on; axis equal; grid on;

% Collect segment node ranges for colouring
if r_c == 0
    seg_ranges = {1:n_bf, n_bf+1:n_bf+n_w, n_bf+n_w+1:N_open};
    seg_labels_plot = {'Bottom flange','Web','Top flange'};
    seg_colors = {col_flange, col_web, col_flange};
else
    seg_ranges = {1:n_bf, n_bf+1:n_bf+n_c, ...
                  n_bf+n_c+1:n_bf+n_c+n_w, ...
                  n_bf+n_c+n_w+1:n_bf+2*n_c+n_w, ...
                  n_bf+2*n_c+n_w+1:N_open};
    seg_labels_plot = {'Bot flange','Bot corner','Web','Top corner','Top flange'};
    seg_colors = {col_flange, col_corner, col_web, col_corner, col_flange};
end

for k = 1:numel(seg_ranges)
    idx = seg_ranges{k};
    if k < numel(seg_ranges)
        idx_plt = [idx, idx(end)+1];   % connect to next segment start
    else
        idx_plt = idx;
    end
    plot(nodesCoords(idx_plt, 1), nodesCoords(idx_plt, 2), ...
        '-', 'Color', seg_colors{k}, 'LineWidth', 2.0, ...
        'DisplayName', seg_labels_plot{k});
    scatter(nodesCoords(idx, 1), nodesCoords(idx, 2), ...
        18, seg_colors{k}, 'filled', 'HandleVisibility','off');
end

% Highlight free tips
scatter(nodesCoords(1, 1),     nodesCoords(1, 2),     80, 'k', 'p', ...
    'filled', 'DisplayName', 'Free tip (node 1)');
scatter(nodesCoords(end, 1),   nodesCoords(end, 2),   80, 'k', 'd', ...
    'filled', 'DisplayName', sprintf('Free tip (node %d)', N_open));

% Mark centroid
plot(0, 0, 'k+', 'MarkerSize', 10, 'LineWidth', 1.5, 'DisplayName', 'Centroid');

% Overlay closed back wall
if saveClosedVariant
    bw_idx  = N_open + (1:n_bw_closed);
    bw_plot = [nodesCoords_closed(bw_idx, :); nodesCoords_closed(1, :)];
    plot(bw_plot(:,1), bw_plot(:,2), '--', 'Color', col_bw, ...
        'LineWidth', 1.5, 'DisplayName', 'Back wall (closed variant)');
end

% Number every ~8th node
step = max(1, floor(N_open/20));
for k = 1:step:N_open
    text(nodesCoords(k,1)*1.1 + 0.5, nodesCoords(k,2)*1.0, num2str(k), ...
        'FontSize', 6, 'Color', [0.45 0.45 0.45]);
end

legend('Location','northeast', 'FontSize', 7);
title(sprintf('C-channel  N=%d  (r_c=%.0f mm)', N_open, r_c));
xlabel('x [mm]'); ylabel('y [mm]');


% --- Panel 2: Edge length profile -----------------------------------------
subplot(2,3,2);
plot(1:N_open-1, edgeL, '.-', 'MarkerSize', 5, 'Color', col_flange);
hold on;

% Shade segments
cumN = 0;
for k = 1:numel(seg_ranges)
    nk = numel(seg_ranges{k});
    patch([cumN+1, cumN+nk, cumN+nk, cumN+1], ...
          [0,0,max(edgeL)*1.4,max(edgeL)*1.4], ...
          seg_colors{k}, 'FaceAlpha',0.12,'EdgeColor','none');
    xline(cumN+0.5,':', 'Color',[0.7 0.7 0.7], 'LineWidth', 0.6);
    cumN = cumN + nk;
end

yline(mean_edge, 'k--', sprintf('mean=%.2f mm',mean_edge), ...
    'LabelHorizontalAlignment','left','FontSize',7);
xlabel('Edge index'); ylabel('Length [mm]');
title('Edge length profile (open chain, N-1 edges)');
grid on; ylim([0, max(edgeL)*1.4]); xlim([1, N_open-1]);


% --- Panel 3: Approx RoC (FD) --------------------------------------------
subplot(2,3,3);
if N_open >= 5
    % Non-periodic FD for open chain: reflect at endpoints instead of wrap
    nc_pad = [nodesCoords(2,1:2); nodesCoords(:,1:2); nodesCoords(end-1,1:2)];
    d1 = nc_pad(3:end,:)   - nc_pad(1:end-2,:);
    d2 = nc_pad(3:end,:) - 2*nc_pad(2:end-1,:) + nc_pad(1:end-2,:);
    cross_z  = d1(:,1).*d2(:,2) - d1(:,2).*d2(:,1);
    kappa    = abs(cross_z) ./ ((d1(:,1).^2 + d1(:,2).^2).^(3/2) + 1e-12);
    RoC_approx = min(max(1./kappa, 5), 1221.95);

    semilogy(1:N_open, RoC_approx, '.-', 'MarkerSize',5,'Color',col_flange);
    hold on;
    cumN = 0;
    for k = 1:numel(seg_ranges)
        nk = numel(seg_ranges{k});
        patch([cumN+1,cumN+nk,cumN+nk,cumN+1], [1,1,3e3,3e3], ...
            seg_colors{k},'FaceAlpha',0.12,'EdgeColor','none');
        cumN = cumN + nk;
    end
    yline(5,       'r--','min clamp (5)','FontSize',7,'LabelHorizontalAlignment','left');
    yline(1221.95, 'r--','flat transition','FontSize',7,'LabelHorizontalAlignment','left');
    if r_c > 0
        yline(r_c, 'g--', sprintf('r_c=%.0f mm',r_c),'FontSize',7);
    end
    xlabel('Node index'); ylabel('RoC [mm]  (log, FD approx.)');
    title('Approx. RoC — open-chain FD (not spline)');
    grid on; ylim([3 3e3]); xlim([1 N_open]);
end


% --- Panel 4: Spherical angle coverage -----------------------------------
% Shows WHY the star-shaped assumption breaks — phi gap on the open side
subplot(2,3,5);
xy_c = nodesCoords(:,1:2);
phi_deg = atan2d(xy_c(:,2), xy_c(:,1));   % angle from centroid

scatter(xy_c(:,1), xy_c(:,2), 25, phi_deg, 'filled');
colorbar; colormap(gca, hsv);
hold on;
plot(0, 0, 'k+', 'MarkerSize', 10, 'LineWidth', 2);
axis equal; grid on;
title('Angle \phi from centroid (shows open-side gap)');
xlabel('x [mm]'); ylabel('y [mm]');

% Annotations for open side
text(W*0.6 - centroid_xy(1), 0, 'Open side\n(no nodes here)', ...
    'HorizontalAlignment','center','FontSize',7,'Color',[0.5 0.5 0.5]);


% --- Panel 5: phi vs node index — reveals non-monotonic range ------------
subplot(2,3,6);
plot(1:N_open, phi_deg, '.-', 'MarkerSize', 5, 'Color', col_flange);
hold on;
xlabel('Node index');
ylabel('\phi [deg]  (from centroid)');
title('\phi coverage — angular range of open chain');
grid on;

% Mark expected start and end angles
yline(phi_deg(1),   'b--', sprintf('Start  %.0f°', phi_deg(1)),   'FontSize',7);
yline(phi_deg(end), 'r--', sprintf('End    %.0f°', phi_deg(end)), 'FontSize',7);

angular_span = max(phi_deg) - min(phi_deg);
text(N_open*0.05, min(phi_deg) + angular_span*0.85, ...
    sprintf('Span: %.0f°\n(%.0f° missing)', angular_span, 360 - angular_span), ...
    'FontSize', 8, 'Color', [0.4 0.4 0.4]);

xlim([1 N_open]);

% Super-title
sgtitle(sprintf( ...
    'C-channel: W=%.0f H=%.0f r_c=%.0f mm  |  N=%d  |  arc-length=%.1f mm', ...
    W, H, r_c, N_open, perimeter));


fprintf('\n--- Summary ---\n');
fprintf('Open-chain file  : %s  (%d nodes, %d edges)\n', ...
    outputFile, N_open, N_open-1);
if saveClosedVariant
    fprintf('Closed-box file  : %s  (%d nodes, %d edges)\n', ...
        outputFileClosed, N_closed, N_closed);
    fprintf('\nTo run closed variant in optimizer:\n');
    fprintf('  1. In the .mat file, load ''nodesCoords_closed'' and ''connectivity_closed''\n');
    fprintf('  2. Or rename them to ''nodesCoords''/''connectivity'' before saving:\n');
    fprintf('       save(''%s'', ''nodesCoords_closed'', ''connectivity_closed'')\n', ...
        outputFileClosed);
    fprintf('     then in Python loadData() will pick them up if you rename in MATLAB first.\n');
end

fprintf('\nPanel 5 (phi coverage) is the key diagnostic:\n');
fprintf('  - Full 360° = closed section, star-shaped OK\n');
fprintf('  - Gap on open side = angle range missing = star-shaped FAILS\n');
fprintf('  - This script reports %.0f° span, %.0f° missing.\n', ...
    angular_span, 360 - angular_span);
