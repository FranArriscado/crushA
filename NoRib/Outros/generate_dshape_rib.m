% generate_dshape_rib.m
% =========================================================================
% Generates a D-shape with a horizontal internal rib, matching the topology
% of the McLaren F1 Side Impact Structure (SIS): outer D-shell (8 plies) +
% horizontal midrib (4 plies) connecting the flat wall to the semicircle's
% equator point.
%
% From AnalyticalTool.m:
%   ModelFilename = {'SIS_OuterShell.stl', 'SIS_Rib.stl'}
%   initial_ply_count = [8, 4]
%
% ── THREE OUTPUTS ──────────────────────────────────────────────────────────
%
%  (A)  partition_data_dshape_rib_full.mat     BRANCHED — NOT optimizer-compatible
%       The complete D + rib geometry as a single branched graph.
%       Topology: 2 degree-3 junction nodes (J_flat and J_arc).
%       Incompatible with the current optimizer for the same reasons as the
%       C-channel and ribbed cross: orderNodes() walks a linear chain and
%       gets stuck at the first degree-3 node; perimeterCalc/smoothnessTerm
%       assume a single ordered loop; Cartesian2Spherical produces degenerate
%       radii where rib and outer-shell nodes share the same phi angles.
%
%  (B)  partition_data_dshape_rib_upper.mat    CLOSED LOOP — optimizer-compatible
%       The UPPER sub-chamber formed by the rib acting as a dividing chord:
%         upper flat wall   (0,0) → (0,+R)       straight, n_uf nodes
%         upper arc         (0,+R) → (R,0)        quarter-circle, n_au nodes
%         rib (reversed)    (R,0) → (0,0)         straight, n_rb nodes
%       This is a "fan" shape — convex, star-shaped from its centroid. ✓
%       Physically represents the upper D-half as a standalone crush section.
%
%  (C)  partition_data_dshape_rib_lower.mat    CLOSED LOOP — optimizer-compatible
%       The LOWER sub-chamber (symmetric to upper):
%         lower flat wall   (0,-R) → (0,0)        straight, n_lf nodes
%         rib               (0,0) → (R,0)          straight, n_rb nodes
%         lower arc         (R,0) → (0,-R)         quarter-circle, n_al nodes
%       Also convex and star-shaped. ✓
%
% ── WHY THE SUB-CHAMBERS ARE USEFUL ───────────────────────────────────────
%
%   Running each sub-chamber through the optimizer independently gives the
%   optimal shape of each half subject to its own boundary constraint (the
%   rib line is implicitly fixed at y=0, x∈[0,R]). This is the closest
%   approximation to "optimising the ribbed SIS" that is achievable with the
%   current single-loop optimizer.
%
%   Comparing the optimised sub-chamber to the optimised plain D-shape
%   (from generate_dshape.m) quantifies the rib's effect on the optimizer's
%   solution — how much the rib line as a boundary constraint changes the
%   optimal shape of each half.
%
% ── GEOMETRY ───────────────────────────────────────────────────────────────
%
%   Outer shell segments (CCW, starting at bottom junction (0,−R)):
%     Seg 1  lower flat   (0,−R) → (0, 0)     y-axis, upward      [n_lf nodes]
%     Seg 2  upper flat   (0, 0) → (0,+R)     y-axis, upward      [n_uf nodes]
%     Seg 3  upper arc    (0,+R) → (R, 0)     θ: +π/2 → 0         [n_au nodes]
%     Seg 4  lower arc    (R, 0) → (0,−R)     θ: 0 → −π/2         [n_al nodes]
%
%   Junction nodes (shared, degree-3 in full geometry):
%     J_flat = (0,  0)  = first node of Seg 2 = last node of rib
%     J_arc  = (R,  0)  = first node of Seg 4 = first node of rib
%
%   Rib interior nodes (added after outer shell):
%     n_rib interior nodes along y=0, x ∈ (0, R) — endpoints are J_flat & J_arc
%
% ── NODE ORDERING (outer shell) ────────────────────────────────────────────
%   "include-start, exclude-end" on every segment.
%   Indices:
%     idx_J_flat  = n_lf + 1
%     idx_J_arc   = n_lf + n_uf + n_au + 1
%     idx_rib_int = N_outer + (1:n_rib)   [interior only, no endpoints]
%
% ── DIMENSIONS (SIS-representative) ───────────────────────────────────────
%   R = 35 mm  →  D height = 70 mm, width = 35 mm
%   Rib at y = 0 (horizontal, connects (0,0) to (R,0))
%   These are scaled to match typical F1 SIS proportions. Adjust R to
%   match the actual SIS cross-section if exact dimensions are known.
%
% Dependencies: none (core MATLAB only)
%
% Francisco Arriscado / FEUP / 2026
% =========================================================================

clear; clc; close all;

%% ======================== USER PARAMETERS ================================

R       = 35;    % [mm]  Semicircle radius  (D height = 2R, D width = R)
                 %       Rib runs from (0,0) to (R,0) — horizontal midrib

n_total = 120;   % Total nodes on the OUTER SHELL (Parts A + sub-chamber reference)
                 % Sub-chamber node counts are derived proportionally from this.
                 % Rule of thumb: 100–160 for a well-resolved spline.

n_rib   = 10;    % Interior rib nodes  (total rib nodes including endpoints = n_rib+2)
                 % The endpoints J_flat and J_arc are shared with the outer shell.

outputFile_full  = 'partition_data_dshape_rib_full.mat';
outputFile_upper = 'partition_data_dshape_rib_upper.mat';
outputFile_lower = 'partition_data_dshape_rib_lower.mat';


%% ======================== INPUT VALIDATION ================================

assert(R > 0,      'R must be positive.');
assert(n_total >= 16, 'n_total should be at least 16.');
assert(n_rib >= 2, 'n_rib must be at least 2 (2 interior nodes).');


%% ======================== ARC-LENGTH PROPORTIONAL SPLIT ==================
% Outer shell segments and their arc lengths:
%   lower flat : L_lf = R      (half of total flat wall height 2R)
%   upper flat : L_uf = R
%   upper arc  : L_au = π/2·R  (quarter circle)
%   lower arc  : L_al = π/2·R

L_lf = R;
L_uf = R;
L_au = (pi/2) * R;
L_al = (pi/2) * R;
L_outer = L_lf + L_uf + L_au + L_al;   % = 2R + πR  (same as plain D-shape ✓)
L_rib   = R;                             % rib length (horizontal)

% Proportional node counts, minimum 3 per segment
n_lf = max(3, round(n_total * L_lf / L_outer));
n_uf = max(3, round(n_total * L_uf / L_outer));
n_au = max(3, round(n_total * L_au / L_outer));
% Absorb rounding error into the arc segments (longest per segment type)
n_al = max(3, n_total - n_lf - n_uf - n_au);
n_total_actual = n_lf + n_uf + n_au + n_al;

fprintf('Outer shell node split:\n');
fprintf('  n_lf=%d, n_uf=%d, n_au=%d, n_al=%d  (total=%d)\n', ...
    n_lf, n_uf, n_au, n_al, n_total_actual);
fprintf('  Rib interior nodes: %d  (+ 2 shared endpoints = %d total rib nodes)\n', ...
    n_rib, n_rib+2);


%% ======================== SEGMENT NODE GENERATION ========================
% Convention: include-start, exclude-end on every segment.
% "trim" drops the last row (the endpoint that belongs to the next segment).

trim = @(pts) pts(1:end-1, :);

% ── Seg 1: lower flat wall  (0,−R) → (0,0)  upward
y1   = linspace(-R, 0, n_lf + 1)';
seg1 = trim([zeros(n_lf+1,1), y1, zeros(n_lf+1,1)]);   % [n_lf × 3]

% ── Seg 2: upper flat wall  (0,0) → (0,+R)  upward
%   First node = J_flat = (0,0)
y2   = linspace(0, R, n_uf + 1)';
seg2 = trim([zeros(n_uf+1,1), y2, zeros(n_uf+1,1)]);    % [n_uf × 3]

% ── Seg 3: upper arc  θ: +π/2 → 0  (CCW outer boundary = CW around centre)
%   First node = J_top = (0,+R), last approached = J_arc = (R,0)
th3  = linspace(pi/2, 0, n_au + 1)';
seg3 = trim([R*cos(th3), R*sin(th3), zeros(n_au+1,1)]);  % [n_au × 3]

% ── Seg 4: lower arc  θ: 0 → −π/2
%   First node = J_arc = (R,0), last approached = J_bot = (0,-R)
th4  = linspace(0, -pi/2, n_al + 1)';
seg4 = trim([R*cos(th4), R*sin(th4), zeros(n_al+1,1)]);  % [n_al × 3]


%% ======================== OUTER SHELL ASSEMBLY ===========================

nodesCoords_outer = [seg1; seg2; seg3; seg4];   % [N_outer × 3]
N_outer = size(nodesCoords_outer, 1);

assert(N_outer == n_total_actual, ...
    'Outer shell node count mismatch: expected %d, got %d.', n_total_actual, N_outer);

% Junction node indices (1-based)
idx_J_flat = n_lf + 1;                         % start of Seg 2 = (0,0)
idx_J_arc  = n_lf + n_uf + n_au + 1;           % start of Seg 4 = (R,0)

% Verify junction positions
assert(abs(nodesCoords_outer(idx_J_flat, 1)) < 1e-10 && ...
       abs(nodesCoords_outer(idx_J_flat, 2)) < 1e-10, ...
       'J_flat is not at (0,0). Check node generation.');
assert(abs(nodesCoords_outer(idx_J_arc,  1) - R) < 1e-10 && ...
       abs(nodesCoords_outer(idx_J_arc,  2))    < 1e-10, ...
       'J_arc is not at (R,0). Check node generation.');

fprintf('\nJunction indices (outer shell):\n');
fprintf('  idx_J_flat = %d  → (%.4f, %.4f) mm\n', ...
    idx_J_flat, nodesCoords_outer(idx_J_flat,1), nodesCoords_outer(idx_J_flat,2));
fprintf('  idx_J_arc  = %d  → (%.4f, %.4f) mm\n', ...
    idx_J_arc,  nodesCoords_outer(idx_J_arc, 1), nodesCoords_outer(idx_J_arc, 2));


%% ======================== RIB INTERIOR NODES =============================
% n_rib interior nodes along y=0, x ∈ (0, R), EXCLUDING both endpoints.
% Endpoints J_flat (idx_J_flat) and J_arc (idx_J_arc) already exist in
% the outer shell — they are SHARED and NOT duplicated here.

x_rib_int = linspace(0, R, n_rib + 2)';   % n_rib+2 points including endpoints
x_rib_int = x_rib_int(2:end-1);           % strip endpoints → n_rib interior points

rib_int_nodes = [x_rib_int, zeros(n_rib, 1), zeros(n_rib, 1)];   % [n_rib × 3]

% Global indices of rib interior nodes (appended after outer shell)
idx_rib_int = N_outer + (1:n_rib)';        % [n_rib × 1]


%% ========================================================================
%% PART A — Full branched geometry
%% ========================================================================

nodesCoords_full = [nodesCoords_outer; rib_int_nodes];
N_full = size(nodesCoords_full, 1);

% Outer shell: standard closed loop
conn_outer = [(1:N_outer)', [2:N_outer, 1]'];

% Rib: chain  J_flat → rib_int(1) → ... → rib_int(n_rib) → J_arc
rib_chain  = [idx_J_flat; idx_rib_int; idx_J_arc];          % n_rib+2 node IDs
conn_rib   = [rib_chain(1:end-1), rib_chain(2:end)];        % [n_rib+1 × 2]

connectivity_full = [conn_outer; conn_rib];

% Verify topology
G_full = graph(connectivity_full(:,1), connectivity_full(:,2));
deg    = degree(G_full);
n_deg2 = sum(deg == 2);
n_deg3 = sum(deg == 3);
n_deg4 = sum(deg == 4);

fprintf('\n--- Part A topology ---\n');
fprintf('Total nodes : %d  (outer=%d + rib_int=%d)\n', N_full, N_outer, n_rib);
fprintf('Total edges : %d\n', size(connectivity_full,1));
fprintf('Degree-2    : %d  (regular wall/rib nodes)\n', n_deg2);
fprintf('Degree-3    : %d  (J_flat and J_arc — expected 2)\n', n_deg3);
assert(n_deg3 == 2, 'Expected exactly 2 degree-3 junction nodes.');
assert(n_deg4 == 0, 'Unexpected degree-4 nodes.');

save(outputFile_full, 'nodesCoords_full', 'connectivity_full');
fprintf('Saved → %s  (branched, NOT optimizer-compatible)\n', outputFile_full);


%% ========================================================================
%% PART B — Upper sub-chamber  (optimizer-compatible closed loop)
%% ========================================================================
%
%  Shape: upper flat wall + upper quarter-arc + rib (reversed)
%  CCW: (0,0) → (0,+R) → arc → (R,0) → rib → (0,0)
%
%  Sub-chamber arc lengths:
%    upper flat : L_uf = R
%    upper arc  : L_au = π/2·R
%    rib        : L_rb = R
%  Total: R(2 + π/2)

L_rb   = R;
L_sub  = L_uf + L_au + L_rb;

% Node count — proportional, using n_total as budget reference
%   (these are INDEPENDENT nodes, not shared with the outer shell)
n_uf_s = max(3, round(n_total_actual * L_uf / L_sub));
n_au_s = max(3, round(n_total_actual * L_au / L_sub));
n_rb_s = max(3, n_total_actual - n_uf_s - n_au_s);
n_upper = n_uf_s + n_au_s + n_rb_s;

fprintf('\n--- Part B: upper sub-chamber ---\n');
fprintf('  n_uf=%d, n_au=%d, n_rb=%d  (total=%d)\n', n_uf_s, n_au_s, n_rb_s, n_upper);

% ── Upper flat  (0,0) → (0,+R)
y_uf   = linspace(0, R, n_uf_s + 1)';
s_uf   = trim([zeros(n_uf_s+1,1), y_uf, zeros(n_uf_s+1,1)]);

% ── Upper arc  θ: +π/2 → 0
th_au  = linspace(pi/2, 0, n_au_s + 1)';
s_au   = trim([R*cos(th_au), R*sin(th_au), zeros(n_au_s+1,1)]);

% ── Rib reversed  (R,0) → (0,0)
x_rb   = linspace(R, 0, n_rb_s + 1)';
s_rb   = trim([x_rb, zeros(n_rb_s+1,1), zeros(n_rb_s+1,1)]);

nodesCoords_upper = [s_uf; s_au; s_rb];
N_upper_actual    = size(nodesCoords_upper, 1);
assert(N_upper_actual == n_upper, 'Upper sub-chamber node count mismatch.');

connectivity_upper = [(1:N_upper_actual)', [2:N_upper_actual, 1]'];

% Star-shaped check
cen_up = mean(nodesCoords_upper(:,1:2));
phi_up = atan2d(nodesCoords_upper(:,2)-cen_up(2), nodesCoords_upper(:,1)-cen_up(1));
phi_up_s = sort(phi_up);
gap_up = max(diff([phi_up_s; phi_up_s(1)+360]));
fprintf('  Centroid: (%.3f, %.3f) mm\n', cen_up(1), cen_up(2));
fprintf('  Max angular gap: %.1f°  (< 180° → star-shaped OK)\n', gap_up);

% Perimeter
dx_u = diff(nodesCoords_upper([1:end,1],1));
dy_u = diff(nodesCoords_upper([1:end,1],2));
peri_upper = sum(sqrt(dx_u.^2 + dy_u.^2));

% Use exact variable names expected by loadData()
nodesCoords  = nodesCoords_upper;
connectivity = connectivity_upper;
save(outputFile_upper, 'nodesCoords', 'connectivity');
fprintf('  Perimeter: %.3f mm  (expected %.3f mm)\n', peri_upper, L_sub);
fprintf('Saved → %s  (optimizer-compatible ✓)\n', outputFile_upper);


%% ========================================================================
%% PART C — Lower sub-chamber  (optimizer-compatible closed loop)
%% ========================================================================
%
%  Shape: lower flat wall + rib + lower quarter-arc
%  CCW: (0,-R) → lower_flat → (0,0) → rib → (R,0) → lower_arc → (0,-R)

n_lf_s = max(3, round(n_total_actual * L_lf / L_sub));
n_rb_s2 = n_rb_s;   % same rib length/count for symmetry
n_al_s = max(3, n_total_actual - n_lf_s - n_rb_s2);
n_lower = n_lf_s + n_rb_s2 + n_al_s;

fprintf('\n--- Part C: lower sub-chamber ---\n');
fprintf('  n_lf=%d, n_rb=%d, n_al=%d  (total=%d)\n', n_lf_s, n_rb_s2, n_al_s, n_lower);

% ── Lower flat  (0,-R) → (0,0)
y_lf_s = linspace(-R, 0, n_lf_s + 1)';
s_lf   = trim([zeros(n_lf_s+1,1), y_lf_s, zeros(n_lf_s+1,1)]);

% ── Rib  (0,0) → (R,0)
x_rb2  = linspace(0, R, n_rb_s2 + 1)';
s_rb2  = trim([x_rb2, zeros(n_rb_s2+1,1), zeros(n_rb_s2+1,1)]);

% ── Lower arc  θ: 0 → -π/2
th_al  = linspace(0, -pi/2, n_al_s + 1)';
s_al   = trim([R*cos(th_al), R*sin(th_al), zeros(n_al_s+1,1)]);

nodesCoords_lower = [s_lf; s_rb2; s_al];
N_lower_actual    = size(nodesCoords_lower, 1);
assert(N_lower_actual == n_lower, 'Lower sub-chamber node count mismatch.');

connectivity_lower = [(1:N_lower_actual)', [2:N_lower_actual, 1]'];

% Star-shaped check
cen_lo = mean(nodesCoords_lower(:,1:2));
phi_lo = atan2d(nodesCoords_lower(:,2)-cen_lo(2), nodesCoords_lower(:,1)-cen_lo(1));
phi_lo_s = sort(phi_lo);
gap_lo = max(diff([phi_lo_s; phi_lo_s(1)+360]));
fprintf('  Centroid: (%.3f, %.3f) mm\n', cen_lo(1), cen_lo(2));
fprintf('  Max angular gap: %.1f°  (< 180° → star-shaped OK)\n', gap_lo);

dx_l = diff(nodesCoords_lower([1:end,1],1));
dy_l = diff(nodesCoords_lower([1:end,1],2));
peri_lower = sum(sqrt(dx_l.^2 + dy_l.^2));

nodesCoords  = nodesCoords_lower;
connectivity = connectivity_lower;
save(outputFile_lower, 'nodesCoords', 'connectivity');
fprintf('  Perimeter: %.3f mm  (expected %.3f mm)\n', peri_lower, L_sub);
fprintf('Saved → %s  (optimizer-compatible ✓)\n', outputFile_lower);


%% ========================================================================
%% PLOTS
%% ========================================================================

figure('Name','D-shape + rib (SIS-like)', 'Position',[50 50 1400 950]);

col_lf   = [0.12 0.47 0.71];   % blue  — lower flat
col_uf   = [0.20 0.63 0.17];   % green — upper flat
col_au   = [0.84 0.15 0.16];   % red   — upper arc
col_al   = [0.89 0.47 0.76];   % pink  — lower arc
col_rib  = [0.50 0.50 0.50];   % gray  — rib
col_junc = [1.00 0.60 0.00];   % orange — junction nodes
col_up   = [0.13 0.47 0.71];
col_lo   = [0.84 0.15 0.16];

% ── Panel 1: Full geometry (Part A) ──────────────────────────────────────
subplot(2,3,[1 4]);
hold on; axis equal; grid on;

% Plot outer shell segments
seg_defs = {1:n_lf, n_lf+1:n_lf+n_uf, ...
            n_lf+n_uf+1:n_lf+n_uf+n_au, ...
            n_lf+n_uf+n_au+1:N_outer};
seg_cols = {col_lf, col_uf, col_au, col_al};
seg_names = {'Lower flat','Upper flat','Upper arc','Lower arc'};

for k = 1:4
    idx = seg_defs{k};
    nxt = mod(seg_defs{k}(end), N_outer) + 1;
    plot(nodesCoords_full([idx,nxt],1), nodesCoords_full([idx,nxt],2), ...
        '-','Color',seg_cols{k},'LineWidth',2,'DisplayName',seg_names{k});
    scatter(nodesCoords_full(idx,1), nodesCoords_full(idx,2), ...
        14, seg_cols{k}, 'filled', 'HandleVisibility','off');
end

% Plot rib
rib_plot_idx = [idx_J_flat; idx_rib_int; idx_J_arc];
plot(nodesCoords_full(rib_plot_idx,1), nodesCoords_full(rib_plot_idx,2), ...
    '-','Color',col_rib,'LineWidth',2.5,'DisplayName','Rib');
scatter(nodesCoords_full(idx_rib_int,1), nodesCoords_full(idx_rib_int,2), ...
    14, col_rib, 'filled','HandleVisibility','off');

% Junction nodes
scatter(nodesCoords_full(idx_J_flat,1), nodesCoords_full(idx_J_flat,2), ...
    90, col_junc,'pentagram','filled','DisplayName','J\_flat  (deg-3)');
scatter(nodesCoords_full(idx_J_arc, 1), nodesCoords_full(idx_J_arc, 2), ...
    90, col_junc,'d',       'filled','DisplayName','J\_arc   (deg-3)');

% Centroid of outer shell for reference
cen_outer = mean(nodesCoords_outer(:,1:2));
plot(cen_outer(1), cen_outer(2), 'k+','MarkerSize',10,'LineWidth',1.5, ...
    'DisplayName','D-shell centroid');

% Annotations
text(nodesCoords_full(idx_J_flat,1)-1.5, nodesCoords_full(idx_J_flat,2)-2, ...
    'J_{flat}','FontSize',8,'HorizontalAlignment','right','Color',col_junc);
text(nodesCoords_full(idx_J_arc,1)+0.5, nodesCoords_full(idx_J_arc,2)-2, ...
    'J_{arc}','FontSize',8,'Color',col_junc);

% Dimension lines
text(-R*0.15, -R*1.15, sprintf('2R = %.0f mm', 2*R), ...
    'FontSize',7,'HorizontalAlignment','center','Color',[0.4 0.4 0.4]);

legend('Location','northwest','FontSize',7);
title(sprintf('Part A: Full D+rib  (BRANCHED — NOT optimizer-compatible)\nN_{outer}=%d, n_{rib}=%d, deg-3 junctions: J_{flat}(0,0), J_{arc}(R,0)', ...
    N_outer, n_rib));
xlabel('x [mm]'); ylabel('y [mm]');


% ── Panel 2: Upper sub-chamber (Part B) ──────────────────────────────────
subplot(2,3,2);
hold on; axis equal; grid on;

closed_u = [nodesCoords_upper; nodesCoords_upper(1,:)];
seg_u = {1:n_uf_s, n_uf_s+1:n_uf_s+n_au_s, n_uf_s+n_au_s+1:n_upper};
seg_u_cols = {col_uf, col_au, col_rib};
seg_u_names = {'Upper flat','Upper arc','Rib (reversed)'};
for k = 1:3
    idx = seg_u{k};
    nxt = mod(idx(end), n_upper) + 1;
    plot(nodesCoords_upper([idx,nxt],1), nodesCoords_upper([idx,nxt],2), ...
        '-','Color',seg_u_cols{k},'LineWidth',2,'DisplayName',seg_u_names{k});
    scatter(nodesCoords_upper(idx,1), nodesCoords_upper(idx,2), ...
        14, seg_u_cols{k},'filled','HandleVisibility','off');
end
plot(cen_up(1), cen_up(2), 'k+','MarkerSize',10,'LineWidth',1.5,'DisplayName','Centroid');
legend('Location','northeast','FontSize',7);
title(sprintf('Part B: Upper sub-chamber\nN=%d, star-shaped gap=%.1f°  ✓', n_upper, gap_up));
xlabel('x [mm]'); ylabel('y [mm]');


% ── Panel 3: Lower sub-chamber (Part C) ──────────────────────────────────
subplot(2,3,3);
hold on; axis equal; grid on;

seg_l = {1:n_lf_s, n_lf_s+1:n_lf_s+n_rb_s2, n_lf_s+n_rb_s2+1:n_lower};
seg_l_cols = {col_lf, col_rib, col_al};
seg_l_names = {'Lower flat','Rib','Lower arc'};
for k = 1:3
    idx = seg_l{k};
    nxt = mod(idx(end), n_lower) + 1;
    plot(nodesCoords_lower([idx,nxt],1), nodesCoords_lower([idx,nxt],2), ...
        '-','Color',seg_l_cols{k},'LineWidth',2,'DisplayName',seg_l_names{k});
    scatter(nodesCoords_lower(idx,1), nodesCoords_lower(idx,2), ...
        14, seg_l_cols{k},'filled','HandleVisibility','off');
end
plot(cen_lo(1), cen_lo(2), 'k+','MarkerSize',10,'LineWidth',1.5,'DisplayName','Centroid');
legend('Location','northeast','FontSize',7);
title(sprintf('Part C: Lower sub-chamber\nN=%d, star-shaped gap=%.1f°  ✓', n_lower, gap_lo));
xlabel('x [mm]'); ylabel('y [mm]');


% ── Panel 4: Edge length profiles ────────────────────────────────────────
subplot(2,3,5);

% Outer shell edge lengths
dx_o = diff(nodesCoords_outer([1:end,1],1));
dy_o = diff(nodesCoords_outer([1:end,1],2));
eL_o = sqrt(dx_o.^2 + dy_o.^2);

plot(1:N_outer, eL_o, '.-','MarkerSize',4,'Color',[0.2 0.2 0.2],'DisplayName','Outer shell');
hold on;

% Shade segments
cumN = 0;
for k = 1:4
    nk = numel(seg_defs{k});
    patch([cumN+1,cumN+nk,cumN+nk,cumN+1],[0,0,max(eL_o)*1.3,max(eL_o)*1.3], ...
        seg_cols{k},'FaceAlpha',0.12,'EdgeColor','none','HandleVisibility','off');
    xline(cumN+0.5,':','Color',[0.7,0.7,0.7],'LineWidth',0.6,'HandleVisibility','off');
    cumN = cumN + nk;
end
yline(mean(eL_o),'k--',sprintf('mean=%.2f mm',mean(eL_o)), ...
    'LabelHorizontalAlignment','left','FontSize',7,'HandleVisibility','off');
% Rib junction markers
xline(idx_J_flat-0.5,'Color',col_junc,'LineWidth',1.2,'DisplayName','J\_flat');
xline(idx_J_arc-0.5, 'Color',col_junc,'LineWidth',1.2,'DisplayName','J\_arc');

xlabel('Edge index (outer shell)'); ylabel('Length [mm]');
title('Edge lengths — outer shell (Part A)'); grid on;
legend('Location','northeast','FontSize',7);
xlim([1,N_outer]); ylim([0,max(eL_o)*1.3]);


% ── Panel 5: Approximate RoC comparison of sub-chambers ──────────────────
subplot(2,3,6);

fd_roc = @(nc) deal( ...
    min(max( ...
        1 ./ (abs( (nc([3:end,1,2],1)-nc([end,1:end-1],1)).*(nc([3:end,1,2],2)-2*nc(:,2)+nc([end,1:end-1],2)) - ...
                   (nc([3:end,1,2],2)-nc([end,1:end-1],2)).*(nc([3:end,1,2],1)-2*nc(:,1)+nc([end,1:end-1],1))) ./ ...
        ( (nc([3:end,1,2],1)-nc([end,1:end-1],1)).^2 + ...
          (nc([3:end,1,2],2)-nc([end,1:end-1],2)).^2 ).^(3/2) + 1e-12 ), ...
        5), 1221.95) );

% Upper sub-chamber RoC (approximate, FD)
nc_u = nodesCoords_upper(:,1:2);
d1_u = nc_u([3:end,1,2],:) - nc_u([end,1:end-1],:);
d2_u = nc_u([3:end,1,2],:) - 2*nc_u(:,:) + nc_u([end,1:end-1],:);
kap_u = abs(d1_u(:,1).*d2_u(:,2)-d1_u(:,2).*d2_u(:,1)) ./ ...
        ((d1_u(:,1).^2+d1_u(:,2).^2).^(3/2)+1e-12);
RoC_u = min(max(1./kap_u, 5), 1221.95);

% Lower sub-chamber RoC
nc_l = nodesCoords_lower(:,1:2);
d1_l = nc_l([3:end,1,2],:) - nc_l([end,1:end-1],:);
d2_l = nc_l([3:end,1,2],:) - 2*nc_l(:,:) + nc_l([end,1:end-1],:);
kap_l = abs(d1_l(:,1).*d2_l(:,2)-d1_l(:,2).*d2_l(:,1)) ./ ...
        ((d1_l(:,1).^2+d1_l(:,2).^2).^(3/2)+1e-12);
RoC_l = min(max(1./kap_l, 5), 1221.95);

semilogy(1:n_upper, RoC_u,'.-','MarkerSize',4,'Color',col_up,'DisplayName','Upper chamber');
hold on;
semilogy(1:n_lower, RoC_l,'.-','MarkerSize',4,'Color',col_lo,'DisplayName','Lower chamber');

yline(5,       'r--','min clamp','FontSize',7,'LabelHorizontalAlignment','left');
yline(1221.95, 'r--','flat transition','FontSize',7,'LabelHorizontalAlignment','left');
yline(R, 'g--',sprintf('R=%.0f mm',R),'FontSize',7,'LabelHorizontalAlignment','right');
xlabel('Node index'); ylabel('RoC [mm]  (log, FD approx.)');
title('Approx. RoC — sub-chambers (FD, not spline)'); grid on;
ylim([3,3e3]); legend('Location','southeast','FontSize',7);

sgtitle(sprintf('D-shape + rib (SIS-like): R=%.0f mm  |  Full: %d+%d nodes  |  Sub-chambers: %d, %d nodes', ...
    R, N_outer, n_rib+2, n_upper, n_lower));


%% ======================== SUMMARY ========================================
fprintf('\n=========================================================\n');
fprintf('  SUMMARY\n');
fprintf('=========================================================\n');
fprintf('  Full (branched)   %s\n', outputFile_full);
fprintf('    Nodes: %d outer + %d rib interior = %d total\n', N_outer, n_rib, N_full);
fprintf('    Edges: %d (outer loop) + %d (rib) = %d total\n', ...
    N_outer, n_rib+1, size(connectivity_full,1));
fprintf('    *** NOT compatible with current optimizer ***\n\n');
fprintf('  Upper sub-chamber  %s\n', outputFile_upper);
fprintf('    Nodes: %d  |  Perimeter: %.1f mm\n', n_upper, peri_upper);
fprintf('    star-shaped: max gap = %.1f°  ✓\n', gap_up);
fprintf('    dataFileName = ''%s''\n\n', outputFile_upper);
fprintf('  Lower sub-chamber  %s\n', outputFile_lower);
fprintf('    Nodes: %d  |  Perimeter: %.1f mm\n', n_lower, peri_lower);
fprintf('    star-shaped: max gap = %.1f°  ✓\n', gap_lo);
fprintf('    dataFileName = ''%s''\n', outputFile_lower);
fprintf('=========================================================\n');
fprintf('\nPhysical interpretation:\n');
fprintf('  Upper chamber = D-half above rib (flat wall + quarter arc + rib line)\n');
fprintf('  Lower chamber = D-half below rib (flat wall + rib line + quarter arc)\n');
fprintf('  Each sub-chamber is equivalent to a quarter-D with a closing rib chord.\n');
fprintf('  Running both through the optimizer gives the best shape of each half\n');
fprintf('  independently, which is the best approximation of SIS rib optimisation\n');
fprintf('  achievable with the current single-loop optimizer.\n');
