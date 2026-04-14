%
%                  abaqusInpProcess.m
%
%                     Tiago Cardoso
%                      up202006481
%                      FEUP, 2025
%
% SUPPORTING FUNCTION FILE
%==========================================================================
%==========================================================================
function FEModelDataUpdated = abaqusInpProcess(FEModelData, log)

log(sprintf('Processing .inp file...'));
% Unpacking data
FEModelDataUpdated = FEModelData;
numInstances = FEModelData.Assembly(1).CountInstance; % number of instances/parts
numElTypes = FEModelData.Elements(1).Count; % number of types of elements
numElSets = FEModelData.ElementSets(1).Count; % number of element sets

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COORDINATES TRANSFORMATION
partIDOrder = [FEModelData.Assembly.PartID];
for p=1:numInstances

    X = FEModelData.Nodes(partIDOrder(p)).Coordinates; % original part nodes coordinates
    T = FEModelData.Assembly(p,1).TranslationVec; % translation vector
    P1 = FEModelData.Assembly(p,1).RotationPoint1; % rotation point 1
    P2 = FEModelData.Assembly(p,1).RotationPoint2; % rotation point 2
    theta = FEModelData.Assembly(p,1).RotationAngle; % [deg] rotation angle

    % Compute transformed coordinates for each part
    % X_{global} = R . X_{local shifted} + T

    if ~isempty(P1) && ~isempty(P2) && ~isempty(T)
        u = P2 - P1; % Rotation axis vector
        u = u / norm(u); % Rotation axis unit vector

        % Convert angle to radians if needed
        theta = deg2rad(theta);
        % Rodrigues rotation matrix:
        % R = I + sin(theta)K + (1-cos(theta))K^2
        % Skew-symmetric matrix K
        K = [  0   -u(3)  u(2);
            u(3)   0   -u(1);
            -u(2)  u(1)   0 ];
        % Rotation matrix
        R = eye(3) + sin(theta)*K + (1 - cos(theta))*(K*K);

        % Shift nodes so axis passes through origin
        X_shifted = X - P1;

        % Rotate
        X_rotated = (R * X_shifted')';  % transpose twice for row-wise mult

        % Shift back
        X_rotated = X_rotated + P1;

        % X_global is the transformed [N x 3] matrix
        X_global = X_rotated + T';  % Make T row vector for row-wise addition

    elseif isempty(P1) && isempty(P2) && ~isempty(T)

        % X_global is the transformed [N x 3] matrix
        X_global = X + T';  % Make T row vector for row-wise addition

    elseif isempty(P1) && isempty(P2) && isempty(T)
        X_global = X;  % Make T row vector for row-wise addition
    end
    Nodes(partIDOrder(p)).GlobalCoords = X_global;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RENUMBER NODES AND ELEMENTS

% Renumber nodes from multiple parts into a single combined list
combinedNodesCoords = [];  % container for combined nodes
combinedNodesIDs = []; % % container for combined and renumbered nodes IDs
combinedNodesIDsCount = 1;      % combined node counter
elementsIDsCount = 1;      % element counter

for p = 1:numInstances
    partIDOrder = [FEModelData.Assembly.PartID];
    nodesIDs = FEModelData.Nodes(partIDOrder(p)).ID;
    coords =  Nodes(partIDOrder(p)).GlobalCoords;

    Np = size(coords, 1);

    newCombinedNodesIDs = (combinedNodesIDsCount : combinedNodesIDsCount + Np - 1)';
    combinedNodesIDs = [combinedNodesIDs; newCombinedNodesIDs];
    combinedNodesCoords = [combinedNodesCoords; coords];
    nodeMap = containers.Map(nodesIDs, newCombinedNodesIDs);
    combinedNodesIDsCount = combinedNodesIDsCount + Np;

    % Renumber elements ID for each element type
    elementTag = [];
    for e=1:numElTypes
        % detect element types of part p
        if FEModelData.Elements(e).AttachedPartID == partIDOrder(p)
            elementTag = [elementTag; FEModelData.Elements(e).ElementTag];
        end
    end
elementsIDsPart = [];
newElementsIDsPart = [];
    for i = 1:length(elementTag)
        elementsIDs = FEModelData.Elements(elementTag(i)).ID;
        elementsIDsPart = [elementsIDsPart; elementsIDs];
        Ne = size(FEModelData.Elements(elementTag(i)).ID, 1);
        newElementsIDs = (elementsIDsCount : elementsIDsCount + Ne - 1)';
        newElementsIDsPart = [newElementsIDsPart; newElementsIDs];
        
        elementsIDsCount = elementsIDsCount + Ne;
        FEModelDataUpdated.Elements(elementTag(i)).ID = newElementsIDs;

        % Renumber nodes ID in table of connectivities

        newConnectList = zeros(size(FEModelData.Elements(elementTag(i)).NodeIDList));
        for j=1:numel(FEModelData.Elements(elementTag(i)).NodeIDList)
            newConnectList(j) = nodeMap(FEModelData.Elements(elementTag(i)).NodeIDList(j));
        end

        FEModelDataUpdated.Elements(elementTag(i)).NodeIDList = newConnectList;
    end
elementsMap = containers.Map(elementsIDsPart, newElementsIDsPart);
    % Renumber elements ID in element sets data
    elementSetTag = [];
    for e=1:numElSets
        % detect element sets of part p
        if FEModelData.ElementSets(e).AttachedPartID == partIDOrder(p)
            elementSetTag = [elementSetTag; e];
        end
    end

    for i = 1:length(elementSetTag)
        if FEModelData.ElementSets(elementSetTag(i)).Generate == 0
            newElSet = zeros(size(FEModelData.ElementSets(elementSetTag(i)).Data));
            for j=1:numel(FEModelData.ElementSets(elementSetTag(i)).Data)
                newElSet(j) = elementsMap(FEModelData.ElementSets(elementSetTag(i)).Data(j));
            end
        else
            newElSet = zeros(size(FEModelData.ElementSets(elementSetTag(i)).Data));
            % write element set in extended form for easier attribution of
            % elements
            elSetExt = [FEModelData.ElementSets(elementSetTag(i)).Data(1):FEModelDataUpdated.ElementSets(elementSetTag(i)).Data(3):FEModelDataUpdated.ElementSets(elementSetTag(i)).Data(2)]';
            for j=1:numel(elSetExt)
                newElSet(j) = elementsMap(elSetExt(j));
            end
        end
        FEModelDataUpdated.ElementSets(elementSetTag(i)).Data = newElSet;
    end
    
end

% Assign material/layup, thickness and ply count information to each element
for i=1:1:length(FEModelDataUpdated.Sections)
    index_elset = find(arrayfun(@(ElementSets) strcmp(ElementSets.Name, FEModelDataUpdated.Sections(i).ElementSetsName), FEModelData.ElementSets));

        FEModelDataUpdated.Sections(i).ElementID = FEModelDataUpdated.ElementSets(index_elset).Data;
   
    for j=1:length(FEModelDataUpdated.Sections(i).ElementID)
        elements_material{FEModelDataUpdated.Sections(i).ElementID(j),1} = FEModelDataUpdated.Sections(i).LayupName;
        elements_thickness{FEModelDataUpdated.Sections(i).ElementID(j),1} = sum(FEModelDataUpdated.Sections(i).PlyThickness);
        elements_plycount{FEModelDataUpdated.Sections(i).ElementID(j),1} = FEModelDataUpdated.Sections(i).PlyCount;
    end
end

FEModelDataUpdated.Elements(1).LayupName = elements_material;
elements_thickness = cellfun(@(x) ternaryNaN(x), elements_thickness);
elements_plycount = cellfun(@(x) ternaryNaN(x), elements_plycount);
FEModelDataUpdated.Elements(1).Thickness = elements_thickness;
FEModelDataUpdated.Elements(1).PlyCount = elements_plycount;

% compute density in each section of the mesh
strMaterials = {FEModelDataUpdated.Materials.Name};
for i=1:1:length(FEModelDataUpdated.Sections)
    clear plyDensity
    plyMaterials = FEModelDataUpdated.Sections(i).PlyMaterial;
    for j=1:length(FEModelDataUpdated.Sections(i).PlyMaterial)
        % assign densities to each ply based on material
        matIndex = find(strcmp(strMaterials, plyMaterials{j}));
        %if exist('data.Materials(matIndex).Density', 'var')
        plyDensity(j,1) = FEModelDataUpdated.Materials(matIndex).Density;
        %else %PRINT ERROR
            %log(sprintf('[ERROR] No density value assigned to materials'));
        %end
    end
    thickness = FEModelDataUpdated.Sections(i).PlyThickness;
    weightAvgDensity = sum(plyDensity .* thickness) / sum(thickness);
    FEModelDataUpdated.Sections(i).Density = weightAvgDensity;
end

% assign density to each element of the mesh
for i=1:length(FEModelDataUpdated.Sections)
    for j=1:length(FEModelDataUpdated.Sections(i).ElementID)
        elements_density(FEModelDataUpdated.Sections(i).ElementID(j),1) = FEModelDataUpdated.Sections(i).Density;
    end
end
FEModelDataUpdated.Elements(1).Density = elements_density;

FEModelDataUpdated = rmfield(FEModelDataUpdated, 'Nodes');
FEModelDataUpdated.Nodes(1).ID =  combinedNodesIDs;
FEModelDataUpdated.Nodes(1).Coordinates = combinedNodesCoords;

end

%--------------------------------------------------------------------------

function out = ternaryNaN(x)
    if isempty(x)
        out = NaN;
    else
        out = x;
    end
end




