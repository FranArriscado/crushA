% generate_cross.m
% =========================================================================
% Generates two cross-section geometries and saves them as .mat files.
%
% ── PART A  Cross outer perimeter  (OPTIMIZER-COMPATIBLE) ───────────────
%
%   The outer boundary of a plus/cross shape is a 12-corner closed polygon.
%   It IS a single closed loop (every node has degree 2).
%   It IS star-shaped from the centroid (0,0): every radial ray from the
%   origin hits the boundary exactly once across all angles, verified by
%   quadrant analysis:
%     θ ∈ [0°, atan(w/L)]     → right-arm cap wall  (x=L)
%     θ = 45°                  → inner concave corner (w,w)  [exact]
%     θ ∈ [atan(w/L), 45°]    → top wall of right arm (y=w)
%     θ ∈ [45°, atan(L/w)]    → right wall of N arm  (x=w)
%     θ ∈ [atan(L/w), 90°]    → N-arm cap wall       (y=L)
%   The pattern repeats by symmetry across all four quadrants.
%   → FULLY COMPATIBLE with the current optimizer.
%
%   Optional: r_c > 0 rounds the 8 CONVEX arm-tip corners only.
%   The 4 CONCAVE notch corners remain sharp (physically representative;
%   adding concave fillets would require a separate arc geometry with the
%   fillet center placed in the void — see inline comment if needed).
%
%   Segment structure (sharp, r_c = 0) — 12 segments CCW from (w,-L):
%     4 arm-cap segments   : length 2w     (one per arm tip)
%     8 arm-side segments  : length L−w   (two per arm, cap→notch)
%
%   With r_c > 0 — 20 segments:
%     4 shortened caps     : length 2w − 2·r_c
%     8 shortened sides    : length (L−w) − r_c  (only arm-tip end rounded)
%     8 convex quarter-arcs: arc length r_c·π/2
%
% ── PART B  Box + cross ribs  (LAYERSPROP-COMPATIBLE, OPTIMIZER-INCOMPATIBLE)
%
%   An outer square with two internal ribs (H and V) meeting at the centre.
%   Junction topology:
%     4 wall-rib nodes   (degree 3): where each rib endpoint meets the wall
%     1 centre node      (degree 4): where H-rib and V-rib cross
%   This branched topology is handled by the branch-detection algorithm in
%   layersProp.m (degree > 2 detection, branch decomposition, per-branch
%   spline fitting, junction RoC averaging).
%
%   INCOMPATIBLE with the current Python optimizer because:
%     (1) orderNodes() is a linear chain walker — it gets stuck at degree-4
%         centre node and degree-3 wall-rib junctions.
%     (2) nodesCoords(), perimeterCalc(), smoothnessTerm() all assume a
%         single ordered node sequence, not a branched graph.
%     (3) Cartesian2Spherical phi values for rib nodes pointing inward
%         overlap with outer-wall phi values at the same angles.
%
% Output .mat fields:
%   Part A:  nodesCoords [N×3],   connectivity [N×2]   — closed loop
%   Part B:  nodesCoords_ribbed [(N_outer+N_ribs)×3],
%            connectivity_ribbed [M×2]                  — branched graph
%
% Dependencies: none (core MATLAB only)
%
% Francisco Arriscado / FEUP / 2026
% =========================================================================

clear; clc; close all;

%% ======================== USER PARAMETERS ================================

% ── Cross dimensions
L    = 35;    % [mm]  Arm half-length  (arm extends from −L to +L in its axis)
w    = 12;    % [mm]  Arm half-width   (arm width = 2w)
                %       Constraint: 0 < w < L

% ── Corner rounding  (Part A only — convex arm-tip corners)
r_c  = 0;     % [mm]  0 = sharp corners.  If r_c > 0: must be < w and < (L−w)

% ── Node count (Part A)
n_total = 120;  % Total nodes on the cross outer perimeter.
                % Rule of thumb: 80–160 for a well-resolved smoothing spline.

% ── Part B dimensions  (box + cross ribs)
S_box      = 80;   % [mm]  Outer square side length
n_half_wall = 10;  % Nodes per half-wall segment  (wall-corner → wall-rib junction,
                   %   includes start endpoint, excludes end endpoint)
                   %   Each full wall: 2·n_half_wall nodes + 1 corner node
                   %   Outer square total: 8·n_half_wall nodes
n_half_rib  = 8;   % Nodes per half-rib  (wall junction → centre,
                   %   includes start, excludes centre endpoint)

% ── Output files
outputFile_A      = 'partition_data_cross.mat';
outputFile_B      = 'partition_data_cross_ribbed.mat';


%% ======================== INPUT VALIDATION ================================

assert(w > 0 && L > w,        'Need 0 < w < L.');
assert(r_c >= 0,               'r_c must be non-negative.');
if r_c > 0
    assert(r_c < w,            'r_c must be < w (%.1f mm).', w);
    assert(r_c < (L - w),      'r_c must be < L−w (%.1f mm).', L - w);
end
assert(n_total >= 16,          'n_total should be at least 16.');
assert(n_half_wall >= 2,       'n_half_wall must be >= 2.');
assert(n_half_rib  >= 2,       'n_half_rib must be >= 2.');


%% ========================================================================
%% PART A — Cross outer perimeter (optimizer-compatible)
%% ========================================================================

%% A.1  Arc-length proportional node distribution -------------------------

% Segment arc lengths
L_cap   = 2*w;         % arm-end cap             (×4, one per arm)
L_side  = L - w;       % arm side from notch to tip corner (×8)

if r_c > 0
    L_cap_s  = 2*w  - 2*r_c;      % shortened cap
    L_side_s = (L - w) - r_c;     % shortened side (notch end untouched)
    L_arc    = r_c * pi/2;         % convex quarter-arc
    L_total  = 4*L_cap_s + 8*L_side_s + 8*L_arc;
else
    L_total  = 4*L_cap + 8*L_side;
end

if r_c == 0
    % 12 segment types:  cap (×4),  side (×8)
    n_cap  = max(2, round(n_total * L_cap  / L_total));
    n_side = max(2, round(n_total * L_side / L_total));
    % Absorb rounding error into the side segments (longest type)
    delta    = n_total - (4*n_cap + 8*n_side);
    n_side   = max(2, n_side + delta);
    n_total  = 4*n_cap + 8*n_side;
    fprintf('Sharp corners: n_cap=%d (×4), n_side=%d (×8)  → total=%d\n', ...
        n_cap, n_side, n_total);
else
    % 20 segment types: cap_s (×4), side_s (×8), arc (×8)
    n_cap_s  = max(2, round(n_total * L_cap_s  / L_total));
    n_side_s = max(2, round(n_total * L_side_s / L_total));
    n_arc    = max(2, round(n_total * L_arc    / L_total));
    delta    = n_total - (4*n_cap_s + 8*n_side_s + 8*n_arc);
    n_side_s = max(2, n_side_s + delta);
    n_total  = 4*n_cap_s + 8*n_side_s + 8*n_arc;
    fprintf('Rounded (r_c=%.1f): n_cap=%d (×4), n_side=%d (×8), n_arc=%d (×8)  → total=%d\n', ...
        r_c, n_cap_s, n_side_s, n_arc, n_total);
end


%% A.2  Segment assembly helpers ------------------------------------------

trim      = @(pts) pts(1:end-1, :);   % include-start, exclude-end
interp_xy = @(p1, p2, n) ...          % straight segment, n+1 pts, then trim
    p1 + (linspace(0,1,n+1)' * (p2 - p1));

% Quarter-circle arc, n+1 pts, then trim
arc_xy = @(cx, cy, th_s, th_e, n) ...
    [cx + r_c*cos(linspace(th_s,th_e,n+1)'), ...
     cy + r_c*sin(linspace(th_s,th_e,n+1)')];


%% A.3  12 reference corner positions of the cross (CCW from bottom-right
%       of S-arm bottom cap) ------------------------------------------
%
%   Label convention:
%     s = S arm, e = E arm, n = N arm, w_arm = W arm
%     BR/BL = bottom-right/left,  TR/TL = top-right/left
%     notch = inner concave corner

corners = struct();
% S arm
corners.s_br  = [+w, -L];   corners.s_bl  = [-w, -L];
corners.notch_se = [+w, -w]; corners.notch_sw = [-w, -w];
% E arm
corners.e_br  = [+L, -w];   corners.e_tr  = [+L, +w];
corners.notch_ne = [+w, +w]; corners.notch_nw = [-w, +w];
% N arm
corners.n_br  = [+w, +L];   corners.n_bl  = [-w, +L];
% W arm
corners.w_tl  = [-L, +w];   corners.w_bl  = [-L, -w];
% (notch_ne, notch_nw, notch_sw already above)


%% A.4  Build all segments (CCW from bottom-right of S-arm cap) -----------
%
%  The "include-start, exclude-end" rule applies to every segment EXCEPT
%  the last, which must include its endpoint so the loop closes exactly.

if r_c == 0

    seg = cell(12, 1);
    % S arm
    seg{1}  = trim(interp_xy(corners.s_br,   corners.notch_se, n_side));  % S arm right side  ↑
    seg{2}  = trim(interp_xy(corners.notch_se, corners.e_br,   n_side));  % E arm bottom       →
    seg{3}  = trim(interp_xy(corners.e_br,   corners.e_tr,     n_cap));   % E arm right cap    ↑
    seg{4}  = trim(interp_xy(corners.e_tr,   corners.notch_ne, n_side));  % E arm top          ←
    seg{5}  = trim(interp_xy(corners.notch_ne, corners.n_br,   n_side));  % N arm right side   ↑
    seg{6}  = trim(interp_xy(corners.n_br,   corners.n_bl,     n_cap));   % N arm top cap      ←
    seg{7}  = trim(interp_xy(corners.n_bl,   corners.notch_nw, n_side));  % N arm left side    ↓
    seg{8}  = trim(interp_xy(corners.notch_nw, corners.w_tl,   n_side));  % W arm top          ←
    seg{9}  = trim(interp_xy(corners.w_tl,   corners.w_bl,     n_cap));   % W arm left cap     ↓
    seg{10} = trim(interp_xy(corners.w_bl,   corners.notch_sw, n_side));  % W arm bottom       →
    seg{11} = trim(interp_xy(corners.notch_sw, corners.s_bl,   n_side));  % S arm left side    ↓
    seg{12} =      interp_xy(corners.s_bl,   corners.s_br,     n_cap);    % S arm bottom cap   →
    seg{12} = seg{12}(1:n_cap, :);    % include endpoint (= node 1 start)

else
    % With rounded convex corners — 20 segments.
    % Convex corners are at: s_br, s_bl, e_br, e_tr, n_br, n_bl, w_tl, w_bl (8 total).
    % Concave corners (notch_se, notch_ne, notch_nw, notch_sw) remain sharp.
    %
    % For each convex corner at position c, with incoming direction d_in and
    % outgoing direction d_out (both unit vectors):
    %   Tangent point 1 (on incoming segment): c - r_c * d_in
    %   Tangent point 2 (on outgoing segment): c + r_c * d_out
    %   Arc centre                           : c + r_c*(d_out - d_in)/norm(...)
    %   Arc from θ_1 to θ_2 (CCW, increasing)
    %
    % Arm-cap segments now span between two tangent points (shortened by r_c each end).
    % Arm-side segments are shortened by r_c at the arm-tip end only.

    seg = cell(20, 1);
    ns = n_side_s;   nc = n_cap_s;   na = n_arc;

    % Helper: convex arc from point p1 to p2 going CCW, centre (cx,cy)
    mk_arc = @(cx,cy,th_s,th_e) trim(arc_xy(cx,cy,th_s,th_e,na));

    % Arm-tip arc centres and tangent-point offsets (derived from geometry)
    % s_br = (+w,-L): incoming → (+1,0), outgoing → (0,+1)
    %   centre (w-r_c, -L+r_c), arc θ: -π/2 → 0
    % s_bl = (-w,-L): incoming → (0,+1), outgoing → (-1,0)  — wait, re-check path direction
    %
    % Recall the CCW path order: we arrive at s_br going RIGHTWARD (from s_bl cap segment),
    % and leave going UPWARD (along S arm right side).
    % So: incoming dir at s_br = (+1,0), outgoing dir = (0,+1)
    % Tangent in:  s_br - r_c*(+1,0) = (w-r_c, -L)
    % Tangent out: s_br + r_c*(0,+1) = (w, -L+r_c)
    % Centre: (w-r_c, -L+r_c),  θ: -π/2 → 0  (CCW)

    % 1  S arm right side (from notch_se to tangent-in at s_br)
    tp_s_br_in = [w-r_c, -L];
    seg{1}  = trim(interp_xy(corners.notch_se, tp_s_br_in, ns));
    % 2  Arc at s_br
    seg{2}  = mk_arc(w-r_c, -L+r_c, -pi/2, 0);
    % 3  E arm bottom (from tangent-out at s_br to notch_se)
    %    tangent-out at s_br = (w, -L+r_c);  tangent-in at e_br = (L-r_c, -w)
    tp_e_br_in = [L-r_c, -w];
    seg{3}  = trim(interp_xy([w, -L+r_c], tp_e_br_in, ns));
    % 4  Arc at e_br  incoming → (0,+1) wait — incoming dir along E arm bottom = (+1,0)
    %    e_br=(+L,-w): incoming (+1,0), outgoing (0,+1)
    %    centre (L-r_c, -w+r_c), θ: -π/2 → 0
    seg{4}  = mk_arc(L-r_c, -w+r_c, -pi/2, 0);
    % 5  E arm right cap (tangent-out at e_br to tangent-in at e_tr)
    %    tangent-out e_br = (L, -w+r_c);  tangent-in e_tr = (L, w-r_c)
    seg{5}  = trim(interp_xy([L, -w+r_c], [L, w-r_c], nc));
    % 6  Arc at e_tr   e_tr=(+L,+w): incoming (0,+1), outgoing (-1,0)
    %    centre (L-r_c, w-r_c), θ: 0 → π/2
    seg{6}  = mk_arc(L-r_c, w-r_c, 0, pi/2);
    % 7  E arm top (tangent-out at e_tr to notch_ne)
    %    tangent-out e_tr = (L-r_c, w)
    seg{7}  = trim(interp_xy([L-r_c, w], corners.notch_ne, ns));
    % 8  N arm right side (from notch_ne to tangent-in at n_br)
    tp_n_br_in = [w, L-r_c];
    seg{8}  = trim(interp_xy(corners.notch_ne, tp_n_br_in, ns));
    % 9  Arc at n_br   n_br=(+w,+L): incoming (0,+1), outgoing (-1,0)
    %    centre (w-r_c, L-r_c), θ: 0 → π/2  (wait: incoming dir is (0,+1) → along +y)
    %    Actually: to reach n_br we travel up the right side of the N arm (+y direction).
    %    Tangent in at n_br = (w, L-r_c); leaving toward cap we go left (-x).
    %    Centre: (w-r_c, L-r_c), θ: 0 (pointing +x from centre to (w, L-r_c)) → π/2 (pointing +y)
    %    Wait: centre to (w, L-r_c): direction = (w-(w-r_c), (L-r_c)-(L-r_c)) = (r_c, 0) → θ=0
    %    centre to (w-r_c, L): direction = (0, r_c) → θ = π/2
    seg{9}  = mk_arc(w-r_c, L-r_c, 0, pi/2);
    % 10 N arm top cap (tangent-out at n_br to tangent-in at n_bl)
    %    tangent-out n_br = (w-r_c, L);  tangent-in n_bl = (-w+r_c, L)
    seg{10} = trim(interp_xy([w-r_c, L], [-w+r_c, L], nc));
    % 11 Arc at n_bl   n_bl=(-w,+L): incoming (-1,0), outgoing (0,-1)
    %    centre (-w+r_c, L-r_c), θ: π/2 → π
    seg{11} = mk_arc(-w+r_c, L-r_c, pi/2, pi);
    % 12 N arm left side (tangent-out at n_bl to notch_nw)
    %    tangent-out n_bl = (-w, L-r_c)
    seg{12} = trim(interp_xy([-w, L-r_c], corners.notch_nw, ns));
    % 13 W arm top (notch_nw to tangent-in at w_tl)
    tp_w_tl_in = [-L+r_c, w];
    seg{13} = trim(interp_xy(corners.notch_nw, tp_w_tl_in, ns));
    % 14 Arc at w_tl   w_tl=(-L,+w): incoming (-1,0), outgoing (0,-1)
    %    centre (-L+r_c, w-r_c), θ: π/2 → π
    seg{14} = mk_arc(-L+r_c, w-r_c, pi/2, pi);
    % 15 W arm left cap
    seg{15} = trim(interp_xy([-L, w-r_c], [-L, -w+r_c], nc));
    % 16 Arc at w_bl   w_bl=(-L,-w): incoming (0,-1), outgoing (+1,0)
    %    centre (-L+r_c, -w+r_c), θ: π → 3π/2
    seg{16} = mk_arc(-L+r_c, -w+r_c, pi, 3*pi/2);
    % 17 W arm bottom (tangent-out at w_bl to notch_sw)
    %    tangent-out w_bl = (-L+r_c, -w)
    seg{17} = trim(interp_xy([-L+r_c, -w], corners.notch_sw, ns));
    % 18 S arm left side (notch_sw to tangent-in at s_bl)
    tp_s_bl_in = [-w, -L+r_c];
    seg{18} = trim(interp_xy(corners.notch_sw, tp_s_bl_in, ns));
    % 19 Arc at s_bl   s_bl=(-w,-L): incoming (0,-1), outgoing (+1,0)
    %    centre (-w+r_c, -L+r_c), θ: π → 3π/2
    seg{19} = mk_arc(-w+r_c, -L+r_c, pi, 3*pi/2);
    % 20 S arm bottom cap (tangent-out at s_bl to tangent-in at s_br)
    %    tangent-out s_bl = (-w+r_c, -L);  tangent-in s_br = (w-r_c, -L)
    seg{20} = interp_xy([-w+r_c, -L], [w-r_c, -L], nc+1);
    seg{20} = seg{20}(1:nc, :);   % keep all but the last (= tangent-in at s_br = start of arc 2)

end


%% A.5  Assemble -----------------------------------------------------------

xy          = vertcat(seg{:});
nodesCoords = [xy, zeros(size(xy,1), 1)];
N           = size(nodesCoords, 1);

assert(N == n_total, ...
    'Node count mismatch: expected %d, assembled %d. Check rounding logic.', n_total, N);

connectivity = [(1:N)', [2:N, 1]'];   % closed loop, 1-based


%% A.6  Quality checks -----------------------------------------------------

dx    = diff(nodesCoords([1:end,1], 1));
dy    = diff(nodesCoords([1:end,1], 2));
edgeL = sqrt(dx.^2 + dy.^2);
perimeter = sum(edgeL);
cv_edge   = std(edgeL) / mean(edgeL);

fprintf('\n--- Part A Quality Report ---\n');
fprintf('Nodes         : %d\n', N);
fprintf('Perimeter     : %.3f mm   (expected %.3f mm)\n', perimeter, L_total);
fprintf('Mean edge     : %.3f mm\n', mean(edgeL));
fprintf('CV (std/mean) : %.4f     (< 0.3 recommended)\n', cv_edge);

% Verify star-shaped: radii from centroid should be > 0 everywhere
centroid   = mean(nodesCoords(:,1:2));
radii      = sqrt((nodesCoords(:,1)-centroid(1)).^2 + (nodesCoords(:,2)-centroid(2)).^2);
phi_deg    = atan2d(nodesCoords(:,2)-centroid(2), nodesCoords(:,1)-centroid(1));
phi_sorted = sort(phi_deg);
phi_gaps   = diff([phi_sorted; phi_sorted(1)+360]);
max_gap    = max(phi_gaps);
fprintf('Max angular gap (star-shape test): %.1f°  (< 180° → star-shaped OK)\n', max_gap);
fprintf('Min radius from centroid: %.3f mm  (> 0 required)\n', min(radii));


%% A.7  Save ---------------------------------------------------------------

save(outputFile_A, 'nodesCoords', 'connectivity');
fprintf('\nPart A saved → %s   [%d×3 nodes, %d×2 edges]\n', outputFile_A, N, N);
fprintf('To use: set dataFileName = ''%s'' in the optimizer.\n', outputFile_A);


%% ========================================================================
%% PART B — Box + cross ribs (layersProp-compatible)
%% ========================================================================

%% B.1  Build outer square with embedded rib-wall junctions ---------------
%
%  Each side is split at its midpoint into two half-segments.
%  Convention: trim = include start, exclude end.
%
%  Outer square node order (CCW):
%    8 half-sides × n_half_wall nodes = 8·n_half_wall nodes total.
%  Junction node positions within this sequence:
%    J_B at index   n_half_wall + 1   (first node of 2nd half-side)
%    J_R at index 3·n_half_wall + 1
%    J_T at index 5·n_half_wall + 1
%    J_L at index 7·n_half_wall + 1

h  = S_box/2;   % half-side length

C_BL = [-h, -h];  C_BR = [+h, -h];
C_TR = [+h, +h];  C_TL = [-h, +h];
J_B  = [ 0, -h];  J_R  = [+h,  0];  J_T  = [0, +h];  J_L  = [-h, 0];

hs_xy = @(p1,p2,n) p1 + ((0:n-1)' / n) * (p2 - p1);   % n nodes: include start, exclude end

hs{1} = hs_xy(C_BL, J_B,  n_half_wall);   % BL corner → J_B
hs{2} = hs_xy(J_B,  C_BR, n_half_wall);   % J_B → BR corner
hs{3} = hs_xy(C_BR, J_R,  n_half_wall);   % BR corner → J_R
hs{4} = hs_xy(J_R,  C_TR, n_half_wall);   % J_R → TR corner
hs{5} = hs_xy(C_TR, J_T,  n_half_wall);   % TR corner → J_T
hs{6} = hs_xy(J_T,  C_TL, n_half_wall);   % J_T → TL corner
hs{7} = hs_xy(C_TL, J_L,  n_half_wall);   % TL corner → J_L
hs{8} = hs_xy(J_L,  C_BL, n_half_wall);   % J_L → BL corner

outer_xy = vertcat(hs{:});
N_outer  = size(outer_xy, 1);   % = 8·n_half_wall

% Junction indices in the outer sequence (1-based)
idx_JB = 1*n_half_wall + 1;   % first node of hs{2}
idx_JR = 3*n_half_wall + 1;   % first node of hs{4}
idx_JT = 5*n_half_wall + 1;   % first node of hs{6}
idx_JL = 7*n_half_wall + 1;   % first node of hs{8}


%% B.2  Build rib interior nodes  -----------------------------------------
%
%  Half-rib: wall-junction → centre, n_half_rib nodes (include start=junction,
%  exclude end=centre).  Interior nodes = rows 2:n_half_rib (indices 2..n_half_rib).
%  Centre node (degree 4) is added once.

hr_L = hs_xy(J_L, [0,0], n_half_rib);   % JL → centre, trim-end
hr_R = hs_xy([0,0], J_R, n_half_rib);   % centre → JR, trim-end
vr_B = hs_xy(J_B, [0,0], n_half_rib);   % JB → centre, trim-end
vr_T = hs_xy([0,0], J_T, n_half_rib);   % centre → JT, trim-end

% Interior nodes (rows 2:end, excluding junction endpoint at row 1)
hr_L_int = hr_L(2:end, :);   % n_half_rib-1 nodes
hr_R_int = hr_R(2:end, :);
vr_B_int = vr_B(2:end, :);
vr_T_int = vr_T(2:end, :);

centre_xy = [0, 0];   % one node

% Concatenate all rib nodes (new nodes, not shared with outer_xy)
rib_xy = [hr_L_int; centre_xy; hr_R_int; vr_B_int; vr_T_int];
N_ribs = size(rib_xy, 1);

% Index of each key rib node within the FULL node array:
N_full = N_outer + N_ribs;
idx_JC = N_outer + size(hr_L_int,1) + 1;   % centre node

% HR left interior: N_outer+1 … N_outer+n_half_rib-1
% HR right interior: N_outer+n_half_rib+1 … N_outer+2*(n_half_rib-1)
% VR bottom interior: similarly for vr_B_int
% VR top interior: similarly for vr_T_int
nhi = n_half_rib - 1;   % interior count per half-rib
idx_HR_L = N_outer + (1:nhi);
idx_HR_R = N_outer + nhi + 1 + (1:nhi);
idx_VR_B = N_outer + 2*nhi + 1 + (1:nhi);
idx_VR_T = N_outer + 3*nhi + 1 + (1:nhi);


%% B.3  Assemble full node list and connectivity --------------------------

nodesCoords_ribbed = [[outer_xy; rib_xy], zeros(N_full, 1)];

% Outer square: standard closed loop (degree 2 everywhere, EXCEPT at the
% 4 junction nodes which will gain degree-3 from the rib edges added below)
outer_conn = [(1:N_outer)', [2:N_outer, 1]'];

% H-rib: JL → HR_L_int → JC → HR_R_int → JR
hrib_ids = [idx_JL, idx_HR_L, idx_JC, idx_HR_R, idx_JR];
hrib_conn = [hrib_ids(1:end-1)', hrib_ids(2:end)'];

% V-rib: JB → VR_B_int → JC → VR_T_int → JT
vrib_ids = [idx_JB, idx_VR_B, idx_JC, idx_VR_T, idx_JT];
vrib_conn = [vrib_ids(1:end-1)', vrib_ids(2:end)'];

connectivity_ribbed = [outer_conn; hrib_conn; vrib_conn];

% Verify junction degrees
G = graph(connectivity_ribbed(:,1), connectivity_ribbed(:,2));
deg = degree(G);
fprintf('\n--- Part B Topology ---\n');
fprintf('Total nodes   : %d  (outer=%d, ribs=%d)\n', N_full, N_outer, N_ribs);
fprintf('Total edges   : %d\n', size(connectivity_ribbed,1));
fprintf('Degree-2 nodes: %d  (outer wall, non-junction)\n', sum(deg==2));
fprintf('Degree-3 nodes: %d  (wall-rib junctions — expected 4)\n', sum(deg==3));
fprintf('Degree-4 nodes: %d  (rib-rib centre — expected 1)\n', sum(deg==4));
assert(sum(deg==3)==4, 'Expected 4 degree-3 junctions.');
assert(sum(deg==4)==1, 'Expected 1 degree-4 centre junction.');


%% B.4  Save ---------------------------------------------------------------

save(outputFile_B, 'nodesCoords_ribbed', 'connectivity_ribbed');
fprintf('Part B saved → %s   [%d×3 nodes, %d×2 edges]\n', ...
    outputFile_B, N_full, size(connectivity_ribbed,1));
fprintf('*** Incompatible with current optimizer (branched topology). ***\n');
fprintf('    Compatible with layersProp.m (branch-detection algorithm).\n');


%% ========================================================================
%% PLOTS
%% ========================================================================

figure('Name','Cross section geometry', 'Position',[60 60 1300 500]);

%% Plot 1: Cross outer perimeter ------------------------------------------
subplot(1,3,1); hold on; axis equal; grid on;

% Colour by segment type
if r_c == 0
    n_segs = 12;
    seg_counts_A = [n_side, n_side, n_cap, n_side, n_side, n_cap, ...
                    n_side, n_side, n_cap, n_side, n_side, n_cap];
    is_cap = logical([0 0 1 0 0 1 0 0 1 0 0 1]);
else
    n_segs = 20;
    seg_counts_A = [n_side_s, n_arc, n_side_s, n_arc, n_cap_s, n_arc, ...
                    n_side_s, n_side_s, n_arc, n_cap_s, n_arc, n_side_s, ...
                    n_side_s, n_arc, n_cap_s, n_arc, n_side_s, n_side_s, n_arc, n_cap_s];
    is_cap = logical([0 1 0 1 1 1 0 0 1 1 1 0 0 1 1 1 0 0 1 1]);
end

col_side = [0.12 0.47 0.71];   col_cap = [0.84 0.15 0.16];
col_notch = [0.49 0.18 0.56];  % for concave corners (sharp, highlighted differently)

cumN = 0;
for k = 1:n_segs
    nk   = seg_counts_A(k);
    idx  = cumN + (1:nk);
    nxt  = mod(cumN+nk, N) + 1;
    col  = col_side;
    if is_cap(k), col = col_cap; end
    plot(nodesCoords([idx,nxt],1), nodesCoords([idx,nxt],2), ...
        '-','Color',col,'LineWidth',1.6,'HandleVisibility','off');
    scatter(nodesCoords(idx,1), nodesCoords(idx,2), 12, col, 'filled', ...
        'HandleVisibility','off');
    cumN = cumN + nk;
end

% Legend proxies
plot(NaN,NaN,'-','Color',col_side,'LineWidth',2,'DisplayName','Arm sides');
plot(NaN,NaN,'-','Color',col_cap, 'LineWidth',2,'DisplayName','Arm caps');
if r_c == 0
    % Mark concave corners
    notch_pts = [corners.notch_se; corners.notch_ne; corners.notch_nw; corners.notch_sw];
    scatter(notch_pts(:,1),notch_pts(:,2),60,'k','d','filled','DisplayName','Concave corners');
end
plot(0,0,'k+','MarkerSize',10,'LineWidth',1.5,'DisplayName','Centroid');
legend('Location','northeast','FontSize',7);
title(sprintf('Part A: Cross perimeter  N=%d  r_c=%.0f mm', N, r_c));
xlabel('x [mm]'); ylabel('y [mm]');

% Annotate arm dimensions
annotation_arms = true;
if annotation_arms
    text(L*0.5, -L*1.12, sprintf('2L = %.0f mm', 2*L), ...
        'HorizontalAlignment','center','FontSize',7,'Color',[0.4 0.4 0.4]);
    text(-L*1.15, 0, sprintf('2w = %.0f mm', 2*w), ...
        'Rotation',90,'HorizontalAlignment','center','FontSize',7,'Color',[0.4 0.4 0.4]);
end


%% Plot 2: Edge length profile + approx RoC -------------------------------
subplot(1,3,2);
yyaxis left;
plot(1:N, edgeL, '.-','MarkerSize',4,'Color',col_side);
ylabel('Edge length [mm]');
yline(mean(edgeL),'k--',sprintf('mean=%.2f',mean(edgeL)), ...
    'LabelHorizontalAlignment','left','FontSize',6);
xlabel('Edge/node index');
title('Edge lengths & approx. RoC');
grid on; xlim([1 N]);

% Approximate RoC (finite-difference, non-periodic at corners)
yyaxis right;
nc2 = [nodesCoords(end-1:end,1:2); nodesCoords(:,1:2); nodesCoords(1:2,1:2)];
d1  = nc2(3:end,:)   - nc2(1:end-2,:);
d2  = nc2(3:end,:) - 2*nc2(2:end-1,:) + nc2(1:end-2,:);
kap = abs(d1(:,1).*d2(:,2)-d1(:,2).*d2(:,1)) ./ ...
      ((d1(:,1).^2+d1(:,2).^2).^(3/2)+1e-12);
RoC_approx = min(max(1./kap,5),1221.95);
semilogy(1:N, RoC_approx,'.-','MarkerSize',4,'Color',col_cap);
ylabel('RoC [mm]  (log)');
yline(5,       'r:','','FontSize',6);
yline(1221.95, 'r:','','FontSize',6);
if r_c > 0
    yline(r_c,'g:',sprintf('r_c=%.0f',r_c),'FontSize',6);
end


%% Plot 3: Box + cross ribs -----------------------------------------------
subplot(1,3,3); hold on; axis equal; grid on;

% Outer square
outer_closed = [nodesCoords_ribbed(1:N_outer,1:2); nodesCoords_ribbed(1,1:2)];
plot(outer_closed(:,1), outer_closed(:,2), 'b-','LineWidth',1.4,'DisplayName','Outer wall');
scatter(nodesCoords_ribbed(1:N_outer,1), nodesCoords_ribbed(1:N_outer,2), ...
    10,'b','filled','HandleVisibility','off');

% H rib
hrib_xy_plot = nodesCoords_ribbed([idx_JL, idx_HR_L, idx_JC, idx_HR_R, idx_JR], 1:2);
plot(hrib_xy_plot(:,1), hrib_xy_plot(:,2), 'r-o','MarkerSize',3,...
    'LineWidth',1.4,'DisplayName','H rib');

% V rib
vrib_xy_plot = nodesCoords_ribbed([idx_JB, idx_VR_B, idx_JC, idx_VR_T, idx_JT], 1:2);
plot(vrib_xy_plot(:,1), vrib_xy_plot(:,2), 'm-o','MarkerSize',3,...
    'LineWidth',1.4,'DisplayName','V rib');

% Junction nodes
junc_idx = [idx_JB, idx_JR, idx_JT, idx_JL];
scatter(nodesCoords_ribbed(junc_idx,1), nodesCoords_ribbed(junc_idx,2), ...
    60,'g','filled','DisplayName','Deg-3 junctions');
scatter(nodesCoords_ribbed(idx_JC,1), nodesCoords_ribbed(idx_JC,2), ...
    80,'k','pentagram','filled','DisplayName','Deg-4 centre');

legend('Location','northeast','FontSize',7);
title(sprintf('Part B: Box+ribs  N=%d  S=%.0f mm  (layersProp only)', N_full, S_box));
xlabel('x [mm]'); ylabel('y [mm]');

sgtitle(sprintf('Cross section: L=%.0f  w=%.0f  r_c=%.0f mm  |  Part A: %d nodes  |  Part B: %d nodes', ...
    L, w, r_c, N, N_full));

fprintf('\n--- Summary ---\n');
fprintf('Part A  %s  →  optimizer-compatible  (N=%d, closed loop, star-shaped ✓)\n', ...
    outputFile_A, N);
fprintf('Part B  %s  →  layersProp-compatible only  (N=%d, branched: 4×deg3, 1×deg4)\n', ...
    outputFile_B, N_full);
