function cfg = config()
% config.m — Central path configuration for CrushAnalytica (MATLAB side).
%
% Usage in any MATLAB script:
%   addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..'));  % adjust depth
%   cfg = config();
%
%   cfg.data          — input partition_data_*.mat files
%   cfg.results_norib — NoRib optimizer outputs (.mat, .txt, .pdf)
%   cfg.results_rib   — Rib optimizer outputs
%   cfg.validation    — validation logs and diagnostics
%
% To load an original geometry:
%   dataFile = fullfile(cfg.data, 'partition_data_square.mat');
%
% To load an optimized result:
%   logFile  = fullfile(cfg.results_norib, 'square_maxSEA_10k_22-03_(v16).txt');
%   dataFile = strrep(logFile, '.txt', '.mat');

    root = fileparts(mfilename('fullpath'));  % wherever config.m lives

    cfg.root          = root;
    cfg.data          = fullfile(root, 'NoRib', 'data');
    cfg.rib_data      = fullfile(root, 'Rib', 'data');
    cfg.results_norib = fullfile(root, 'results', 'NoRib');
    cfg.results_rib   = fullfile(root, 'results', 'Rib');
    cfg.validation    = cfg.results_norib;  % validation outputs go here

    % Auto-create folders if they don't exist
    folders = {cfg.data, cfg.rib_data, cfg.results_norib, cfg.results_rib};
    for i = 1:numel(folders)
        if ~isfolder(folders{i})
            mkdir(folders{i});
        end
    end
end
