% generate_rrect.m
% =========================================================================
% Generates a rounded-rectangle cross-section and saves it as a .mat file
% ready to be loaded by the Python optimizer (optimizationModule_patched_v8.py).
%
% Geometry — 8 segments in CCW order, starting at (−W/2+r_c, −H/2):
%
%   1. Bottom straight   (−W/2+r_c, −H/2)  →  (+W/2−r_c, −H/2)   rightward
%   2. Bottom-right arc  centre (+W/2−r_c, −H/2+r_c)   θ: −π/2 → 0
%   3. Right straight    (+W/2, −H/2+r_c)  →  (+W/2, +H/2−r_c)   upward
%   4. Top-right arc     centre (+W/2−r_c, +H/2−r_c)   θ:  0   → +π/2
%   5. Top straight      (+W/2−r_c, +H/2)  →  (−W/2+r_c, +H/2)   leftward
%   6. Top-left arc      centre (−W/2+r_c, +H/2−r_c)   θ: +π/2 → +π
%   7. Left straight     (−W/2, +H/2−r_c)  →  (−W/2, −H/2+r_c)   downward
%   8. Bottom-left arc   centre (−W/2+r_c, −H/2+r_c)   θ: +π   → 3π/2
%
%   Every segment includes its START point and EXCLUDES its END point.
%   Concatenating all 8 gives a closed loop with no duplicate junction nodes.
%
% Output .mat fields  (must match what loadData() expects):
%   nodesCoords   [N×3]  double  — (x, y, 0) in mm, row = one node
%   connectivity  [N×2]  double  — 1-based edge pairs, closed loop
%                                   edge k connects node k → node k+1
%                                   last edge connects node N → node 1
%
% Node distribution:
%   Proportional to arc length per segment.  All 4 corners get the same
%   count (n_c), both horizontal walls get n_hw, both vertical walls n_vw.
%   Rounding error is absorbed into the longest individual segment.
%
% Minimum node count per segment: 2  (enforced by max(2, round(...))).
% Recommended total: 80–160 nodes for a well-resolved smoothing spline.
%
% Dependencies: none (core MATLAB only)
%
% Francisco Arriscado / FEUP / 2026
% =========================================================================

clear; clc; close all;

%% ======================== USER PARAMETERS ================================

W       = 80;    % [mm]  Total outer width  (x-direction)
H       = 50;    % [mm]  Total outer height (y-direction)
r_c     = 12;    % [mm]  Corner radius  — must satisfy r_c < min(W,H)/2

n_total = 120;   % Total node count — distributes proportionally across all 8 segments
                 % Rule of thumb: 80–160 nodes for a well-resolved spline

outputFile = 'partition_data_rrect.mat';

% ── Optional: override individual counts (leave as 0 to use arc-length split)
%    n_hw: nodes per horizontal straight (bottom, top)
%    n_vw: nodes per vertical   straight (left, right)
%    n_c:  nodes per corner arc          (all 4 corners equal)
%    Set ALL THREE to non-zero to use the override, or leave all at 0 for auto.
n_hw_override = 0;
n_vw_override = 0;
n_c_override  = 0;


%% ======================== INPUT VALIDATION ================================

assert(W > 0 && H > 0,          'W and H must be positive.');
assert(r_c > 0,                  'r_c must be positive.');
assert(r_c < W/2,                'r_c must be < W/2  (%.1f mm).', W/2);
assert(r_c < H/2,                'r_c must be < H/2  (%.1f mm).', H/2);
assert(n_total >= 16,            'n_total should be at least 16 (2 per segment).');


%% ======================== ARC-LENGTH PROPORTIONAL SPLIT ==================
% Each segment's arc length:
%   horizontal straight : L_hw  = W - 2*r_c      (×2 segments)
%   vertical   straight : L_vw  = H - 2*r_c      (×2 segments)
%   quarter-circle arc  : L_c   = π/2 * r_c      (×4 segments)

L_hw    = W - 2*r_c;
L_vw    = H - 2*r_c;
L_c     = (pi/2) * r_c;
L_total = 2*L_hw + 2*L_vw + 4*L_c;

if n_hw_override > 0 && n_vw_override > 0 && n_c_override > 0
    n_hw = n_hw_override;
    n_vw = n_vw_override;
    n_c  = n_c_override;
    n_total = 2*n_hw + 2*n_vw + 4*n_c;
    fprintf('Using manual split: n_hw=%d, n_vw=%d, n_c=%d  (total=%d)\n', ...
        n_hw, n_vw, n_c, n_total);
else
    n_hw = max(2, round(n_total * L_hw / L_total));
    n_vw = max(2, round(n_total * L_vw / L_total));
    n_c  = max(2, round(n_total * L_c  / L_total));

    % Absorb rounding error into the longest segment type
    n_actual = 2*n_hw + 2*n_vw + 4*n_c;
    delta    = n_total - n_actual;   % may be negative (over-counted) or positive

    if delta ~= 0
        % Lengths of the three representative segments
        lens   = [L_hw, L_vw, L_c];
        counts = [2,    2,    4 ];        % multiplicity of each segment type
        [~, longest_type] = max(lens);    % adjust the longest one

        switch longest_type
            case 1,  n_hw = n_hw + delta;
            case 2,  n_vw = n_vw + delta;
            case 3,  n_c  = n_c  + delta;
        end

        % Sanity check: no segment goes below 2
        n_hw = max(2, n_hw);
        n_vw = max(2, n_vw);
        n_c  = max(2, n_c);
    end

    n_total = 2*n_hw + 2*n_vw + 4*n_c;
    fprintf('Arc-length split: n_hw=%d, n_vw=%d, n_c=%d  (total=%d)\n', ...
        n_hw, n_vw, n_c, n_total);
end

% Segment arc lengths for spacing report
seg_labels = {'bot straight','BR corner','right straight','TR corner', ...
              'top straight','TL corner','left straight', 'BL corner'};
seg_lengths = [L_hw, L_c, L_vw, L_c, L_hw, L_c, L_vw, L_c];
seg_counts  = [n_hw, n_c, n_vw, n_c, n_hw, n_c, n_vw, n_c];


%% ======================== SEGMENT GENERATORS =============================
% Both helpers include the START point and EXCLUDE the END point so that
% concatenating all 8 segments produces exactly n_total nodes with no
% duplicates at any junction.

% Straight segment: p_start → p_end, n nodes [n×2]
seg_straight = @(ps, pe, n) ...
    ps + (linspace(0, 1, n+1)' * (pe - ps));          % (n+1)×1 · 1×2 = (n+1)×2
% The above returns n+1 rows — we trim below during assembly.

% Quarter-arc: centre (cx,cy), radius r_c, theta_start→theta_end, n nodes [n×2]
seg_arc = @(cx, cy, theta_s, theta_e, n) ...
    [cx + r_c * cos(linspace(theta_s, theta_e, n+1)'), ...
     cy + r_c * sin(linspace(theta_s, theta_e, n+1)')];
% Same: n+1 rows, drop last during assembly.

% Helper to drop the last point (the junction endpoint)
trim = @(pts) pts(1:end-1, :);


%% ======================== CORNER CENTRES =================================
% Named clearly for readability
cx_br = +W/2 - r_c;   cy_br = -H/2 + r_c;   % bottom-right
cx_tr = +W/2 - r_c;   cy_tr = +H/2 - r_c;   % top-right
cx_tl = -W/2 + r_c;   cy_tl = +H/2 - r_c;   % top-left
cx_bl = -W/2 + r_c;   cy_bl = -H/2 + r_c;   % bottom-left


%% ======================== NODE GENERATION ================================
% CCW traversal.  Start at bottom-left tangent point of the bottom edge:
%   (-W/2+r_c, -H/2)  =  the point where the BL arc ends / bottom straight begins.

% 1. Bottom straight: left → right along y = -H/2
p1s = [-W/2+r_c, -H/2];   p1e = [+W/2-r_c, -H/2];
seg1 = trim(seg_straight(p1s, p1e, n_hw));

% 2. Bottom-right corner: θ = -π/2 → 0
seg2 = trim(seg_arc(cx_br, cy_br, -pi/2, 0, n_c));

% 3. Right straight: bottom → top along x = +W/2
p3s = [+W/2, -H/2+r_c];   p3e = [+W/2, +H/2-r_c];
seg3 = trim(seg_straight(p3s, p3e, n_vw));

% 4. Top-right corner: θ = 0 → +π/2
seg4 = trim(seg_arc(cx_tr, cy_tr, 0, pi/2, n_c));

% 5. Top straight: right → left along y = +H/2
p5s = [+W/2-r_c, +H/2];   p5e = [-W/2+r_c, +H/2];
seg5 = trim(seg_straight(p5s, p5e, n_hw));

% 6. Top-left corner: θ = +π/2 → π
seg6 = trim(seg_arc(cx_tl, cy_tl, pi/2, pi, n_c));

% 7. Left straight: top → bottom along x = -W/2
p7s = [-W/2, +H/2-r_c];   p7e = [-W/2, -H/2+r_c];
seg7 = trim(seg_straight(p7s, p7e, n_vw));

% 8. Bottom-left corner: θ = π → 3π/2  (ends at start of segment 1)
seg8 = trim(seg_arc(cx_bl, cy_bl, pi, 3*pi/2, n_c));

% Assemble — each row is (x, y)
xy = [seg1; seg2; seg3; seg4; seg5; seg6; seg7; seg8];

% Append Z = 0
nodesCoords = [xy, zeros(size(xy,1), 1)];   % [N×3]
N = size(nodesCoords, 1);

assert(N == n_total, ...
    'Node count mismatch: expected %d, assembled %d. Check rounding logic.', n_total, N);


%% ======================== CONNECTIVITY ===================================
% Closed-loop 1-based edge pairs: k → k+1, last edge N → 1

connectivity = [(1:N)', [2:N, 1]'];   % [N×2]


%% ======================== SEGMENT BOUNDARY INDICES =======================
% Track which node index each segment starts at — used for plots and labels.

seg_start_idx = cumsum([1, n_hw, n_c, n_vw, n_c, n_hw, n_c, n_vw]);
% seg_start_idx(k) = first node index of segment k  (1-based)


%% ======================== QUALITY CHECKS =================================

dx    = diff(nodesCoords([1:end,1], 1));
dy    = diff(nodesCoords([1:end,1], 2));
edgeL = sqrt(dx.^2 + dy.^2);

perimeter   = sum(edgeL);
mean_edge   = mean(edgeL);
std_edge    = std(edgeL);
cv_edge     = std_edge / mean_edge;

expected_perim = L_total;
perim_err      = abs(perimeter - expected_perim);

fprintf('\n--- Quality Report ---\n');
fprintf('Dimensions    : W=%.1f mm,  H=%.1f mm,  r_c=%.1f mm\n', W, H, r_c);
fprintf('Nodes         : %d\n', N);
fprintf('Perimeter     : %.4f mm  (expected %.4f mm,  error %.2e mm)\n', ...
    perimeter, expected_perim, perim_err);
fprintf('Mean edge len : %.4f mm\n', mean_edge);
fprintf('Std  edge len : %.4f mm\n', std_edge);
fprintf('CV (std/mean) : %.4f     (< 0.3 recommended)\n', cv_edge);

% Per-segment spacing check
fprintf('\nPer-segment spacing:\n');
fprintf('  %-20s  nodes  arc_len [mm]  mean_edge [mm]\n', 'Segment');
fprintf('  %s\n', repmat('-',1,58));
cumN = 0;
for k = 1:8
    nk  = seg_counts(k);
    lk  = seg_lengths(k);
    idx = cumN + (1:nk);
    % Edge lengths belonging to this segment (edge k leaves node k)
    ek  = edgeL(idx);
    fprintf('  %-20s  %5d  %12.3f  %13.4f\n', seg_labels{k}, nk, lk, mean(ek));
    cumN = cumN + nk;
end

% Junction continuity — check the 8 seam points
fprintf('\nJunction continuity (expected vs actual corner tangent points):\n');
junctions = struct();
junctions(1).name = 'bot start  (-W/2+r_c, -H/2)';  junctions(1).expected = [-W/2+r_c, -H/2];
junctions(2).name = 'BR tangent (+W/2-r_c, -H/2)';  junctions(2).expected = [+W/2-r_c, -H/2];
junctions(3).name = 'right bot  (+W/2, -H/2+r_c)';  junctions(3).expected = [+W/2,     -H/2+r_c];
junctions(4).name = 'right top  (+W/2, +H/2-r_c)';  junctions(4).expected = [+W/2,     +H/2-r_c];
junctions(5).name = 'TR tangent (+W/2-r_c, +H/2)';  junctions(5).expected = [+W/2-r_c, +H/2];
junctions(6).name = 'TL tangent (-W/2+r_c, +H/2)';  junctions(6).expected = [-W/2+r_c, +H/2];
junctions(7).name = 'left top   (-W/2, +H/2-r_c)';  junctions(7).expected = [-W/2,     +H/2-r_c];
junctions(8).name = 'left bot   (-W/2, -H/2+r_c)';  junctions(8).expected = [-W/2,     -H/2+r_c];

for k = 1:8
    actual_idx = seg_start_idx(k);
    actual_xy  = nodesCoords(actual_idx, 1:2);
    gap = norm(actual_xy - junctions(k).expected);
    flag = '';
    if gap > 1e-8, flag = '  *** GAP DETECTED ***'; end
    fprintf('  Seg %d  %-34s  gap=%.2e mm%s\n', ...
        k, junctions(k).name, gap, flag);
end

fprintf('\nCentroid: (%.4f, %.4f) mm  (expected ≈ (0, 0))\n', ...
    mean(nodesCoords(:,1)), mean(nodesCoords(:,2)));


%% ======================== SAVE ===========================================

save(outputFile, 'nodesCoords', 'connectivity');
fprintf('\nSaved → %s\n', outputFile);
fprintf('Fields: nodesCoords [%d×3], connectivity [%d×2]\n', N, N);
fprintf('To use: set dataFileName = ''%s'' in the optimizer.\n', outputFile);


%% ======================== PLOTS ==========================================

figure('Name', 'Rounded-rectangle geometry check', ...
       'Position', [100 100 1200 400]);

% Assign a colour per segment type for the shape plot
type_color = struct();
type_color.straight = [0.12 0.47 0.71];  % blue
type_color.corner   = [0.84 0.15 0.16];  % red

seg_is_straight = logical([1 0 1 0 1 0 1 0]);

% --- 1. Shape with segment colouring --------------------------------------
subplot(1,3,1);
hold on; axis equal; grid on;
cumN = 0;
for k = 1:8
    nk   = seg_counts(k);
    idx  = cumN + (1:nk);
    % close the segment visually by appending the first node of the NEXT segment
    next_start = mod(cumN + nk, N) + 1;
    idx_closed = [idx, next_start];
    if seg_is_straight(k)
        col = type_color.straight;
        lw  = 1.5;
    else
        col = type_color.corner;
        lw  = 2.0;
    end
    plot(nodesCoords(idx_closed,1), nodesCoords(idx_closed,2), ...
        '-', 'Color', col, 'LineWidth', lw);
    % Mark start-of-segment node
    scatter(nodesCoords(cumN+1, 1), nodesCoords(cumN+1, 2), ...
        36, col, 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 0.5);
    cumN = cumN + nk;
end

% Label every ~8th node
step = max(1, floor(N/25));
for k = 1:step:N
    text(nodesCoords(k,1)*1.07, nodesCoords(k,2)*1.07, num2str(k), ...
        'FontSize', 6, 'Color', [0.45 0.45 0.45]);
end

% Label corner centres
cc = [cx_br cy_br; cx_tr cy_tr; cx_tl cy_tl; cx_bl cy_bl];
cc_names = {'BR','TR','TL','BL'};
for k = 1:4
    plot(cc(k,1), cc(k,2), 'k+', 'MarkerSize', 5);
    text(cc(k,1), cc(k,2)-2.5, cc_names{k}, 'FontSize', 6, ...
        'HorizontalAlignment','center', 'Color', [0.3 0.3 0.3]);
end

legend({'Straight','','Corner',''}, 'Location','southeast', 'FontSize', 7);
title(sprintf('Rounded rectangle  N=%d', N));
xlabel('x [mm]'); ylabel('y [mm]');
xlim([-W/2*1.3, W/2*1.3]); ylim([-H/2*1.4, H/2*1.4]);


% --- 2. Edge length profile with segment boundaries ----------------------
subplot(1,3,2);
plot(1:N, edgeL, '.-', 'MarkerSize', 4, 'LineWidth', 0.8, ...
    'Color', [0.12 0.47 0.71]);
hold on;

% Shade corner segments
cumN = 0;
for k = 1:8
    nk = seg_counts(k);
    if ~seg_is_straight(k)
        patch([cumN+1, cumN+nk, cumN+nk, cumN+1], ...
              [0, 0, max(edgeL)*1.3, max(edgeL)*1.3], ...
              [0.84 0.15 0.16], 'FaceAlpha', 0.1, 'EdgeColor','none');
    end
    xline(cumN + 0.5, ':', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.6);
    cumN = cumN + nk;
end

yline(mean_edge, 'k--', sprintf('mean=%.2f mm', mean_edge), ...
    'LabelHorizontalAlignment','left', 'FontSize', 7);
xlabel('Edge index');
ylabel('Length [mm]');
title('Edge length profile');
grid on;
ylim([0, max(edgeL)*1.3]);
xlim([1, N]);


% --- 3. Approximate RoC preview (finite-difference) ----------------------
% Rough visual only — the optimizer uses spaps smoothing splines internally.
subplot(1,3,3);

wrap2 = @(v) [v(end-1:end,:); v; v(1:2,:)];
nw = wrap2(nodesCoords(:,1:2));
d1 = nw(3:end,:)   - nw(1:end-2,:);
d2 = nw(3:end,:) - 2*nw(2:end-1,:) + nw(1:end-2,:);
cross_z = d1(:,1).*d2(:,2) - d1(:,2).*d2(:,1);
kappa   = abs(cross_z) ./ ((d1(:,1).^2 + d1(:,2).^2).^(3/2) + 1e-12);
RoC_approx = min(max(1./kappa, 5), 1221.95);

semilogy(1:N, RoC_approx, '.-', 'MarkerSize', 4, 'LineWidth', 0.8, ...
    'Color', [0.12 0.47 0.71]);
hold on;

% Shade corner segments
cumN = 0;
for k = 1:8
    nk = seg_counts(k);
    if ~seg_is_straight(k)
        patch([cumN+1, cumN+nk, cumN+nk, cumN+1], ...
              [1, 1, 2e3, 2e3], ...
              [0.84 0.15 0.16], 'FaceAlpha', 0.1, 'EdgeColor','none');
        xline(cumN + 0.5, ':', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.6);
    end
    cumN = cumN + nk;
end

yline(5,       'r--', 'min clamp (5 mm)',    'FontSize', 7, ...
    'LabelHorizontalAlignment','left');
yline(1221.95, 'r--', 'flat transition',      'FontSize', 7, ...
    'LabelHorizontalAlignment','left');
yline(r_c,     'g--', sprintf('r\\_c = %.0f mm', r_c), 'FontSize', 7, ...
    'LabelHorizontalAlignment','right');
xlabel('Node index');
ylabel('RoC [mm]  (log scale)');
title('Approx. RoC  (FD, not spline)');
grid on; ylim([3, 3e3]);
xlim([1, N]);

sgtitle(sprintf('Rounded rectangle: W=%.0f  H=%.0f  r\\_c=%.0f mm  |  N=%d  |  perimeter=%.1f mm', ...
    W, H, r_c, N, perimeter));

fprintf('\nPlot rendered. Checks to confirm:\n');
fprintf('  Shape  : 4 straight sides (blue) connected by 4 arcs (red)\n');
fprintf('  Edges  : uniform along straights; slightly wider at arc midpoints if r_c is small\n');
fprintf('  RoC    : straights → clamped at flat_transition (top dashed)\n');
fprintf('           corners  → near r_c (green dashed), clamped at 5 mm if r_c < 5\n');
fprintf('  Red shading highlights corner segments in all three panels.\n');
