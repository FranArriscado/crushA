%
%                  meshTreatment.m
%
%                     Tiago Cardoso
%                      up202006481
%                      FEUP, 2025
%
% SUPPORTING FUNCTION FILE
%==========================================================================
%==========================================================================
function [elements, FEModelData] = meshTreatment(FEModelData)

% Variables initialization
% QUAD elements properties of original FE Model
quadElements = struct( ...
    'ConnectivityList',[],...
    'CompositeIDs',{},...
    'ID',[],...
    'IDCompositeIDs',{},...
    'Thickness',[],...
    'PlyCount',[],...
    'Density',[]);
% TRI elements properties of original FE Model
triElements = struct( ...
    'ConnectivityList',[],...
    'CompositeIDs',{},...
    'ID',[],...
    'IDCompositeIDs',{},...
    'Thickness',[],...
    'PlyCount',[],...
    'Density',[]);
% converted TRI elements properties of FE Model
triConvElements = struct( ...
    'ConnectivityList',[],...
    'CompositeIDs',{},...
    'Thickness',[],...
    'PlyCount',[],...
    'Density',[]);
% elements properties of converted FE Model (all elements are TRI elements)
elements = struct( ...
    'ConnectivityList',[],...
    'LayupName',{},...
    'Thickness',[],...
    'PlyCount',[],...
    'Density',[]);
countTri = 0;
countQuad = 0;

for i=1:length(FEModelData.Elements)

    [~, columns] = size(FEModelData.Elements(i).NodeIDList);

    if columns == 4
        countQuad = countQuad + 1;
        % store QUAD elements properties
        quadElements(1).ConnectivityList = vertcat(quadElements.ConnectivityList, FEModelData.Elements(i).NodeIDList);
        quadElements(1).ID = vertcat(quadElements.ID, num2cell(FEModelData.Elements(i).ID));
        quadElements(1).CompositeIDs = vertcat(quadElements.CompositeIDs, FEModelData.Elements(1).CompositeIDs(FEModelData.Elements(i).ID));
        quadElements(1).Thickness = vertcat(quadElements.Thickness, FEModelData.Elements(1).Thickness(FEModelData.Elements(i).ID));
        quadElements(1).PlyCount = vertcat(quadElements.PlyCount, FEModelData.Elements(1).PlyCount(FEModelData.Elements(i).ID));
        quadElements(1).Density = vertcat(quadElements.Density, FEModelData.Elements(1).Density(FEModelData.Elements(i).ID));
    elseif columns == 3
        countTri = countTri + 1;
        % store TRI elements properties
        triElements(1).ConnectivityList = vertcat(triElements.ConnectivityList, FEModelData.Elements(i).NodeIDList);
        triElements(1).ID = vertcat(triElements.ID,num2cell(FEModelData.Elements(i).ID));
        triElements(1).CompositeIDs = vertcat(triElements.CompositeIDs, FEModelData.Elements(1).CompositeIDs(FEModelData.Elements(i).ID));
        triElements(1).Thickness = vertcat(triElements.Thickness, FEModelData.Elements(1).Thickness(FEModelData.Elements(i).ID));
        triElements(1).PlyCount = vertcat(triElements.PlyCount, FEModelData.Elements(1).PlyCount(FEModelData.Elements(i).ID));
        triElements(1).Density = vertcat(triElements.Density, FEModelData.Elements(1).Density(FEModelData.Elements(i).ID));
    end

end

% link and store both TRI and QUAD elements 'ID' with 'LayupName'
if countTri == 0
    quadElements.IDCompositeIDs = horzcat(quadElements.ID,quadElements.CompositeIDs);
elseif countQuad == 0
    triElements.IDCompositeIDs = horzcat(triElements.ID,triElements.CompositeIDs);
else
    quadElements.IDCompositeIDs = horzcat(quadElements.ID,quadElements.CompositeIDs);
    triElements.IDCompositeIDs = horzcat(triElements.ID,triElements.CompositeIDs);
end

% create new TRI connectivity list from QUAD connectivity list
if countQuad ~= 0
    triConvElements(1).ConnectivityList = quad2tri(quadElements.ConnectivityList);
    % concatenate new TRI connectivity list with original TRI connectivity list
    elements(1).ConnectivityList = vertcat(triConvElements.ConnectivityList, triElements.ConnectivityList);
else
    elements(1).ConnectivityList = triElements.ConnectivityList;
end

% link QUAD elements properties to new converted TRI elements
if countQuad ~= 0
    for i=1:size(quadElements.ConnectivityList,1)
        triConvElements.CompositeIDs(2*i-1) = quadElements.CompositeIDs(i);
        triConvElements.CompositeIDs(2*i) = quadElements.CompositeIDs(i);
        triConvElements.Thickness(2*i-1) = quadElements.Thickness(i);
        triConvElements.Thickness(2*i) = quadElements.Thickness(i);
        triConvElements.PlyCount(2*i-1) = quadElements.PlyCount(i);
        triConvElements.PlyCount(2*i) = quadElements.PlyCount(i);
        triConvElements.Density(2*i-1) = quadElements.Density(i);
        triConvElements.Density(2*i) = quadElements.Density(i);
    end
    % concatenate new TRI properties with original TRI properties
    elements.CompositeIDs = vertcat(triConvElements.CompositeIDs', triElements.CompositeIDs);
    elements.Thickness = vertcat(triConvElements.Thickness', triElements.Thickness);
    elements.PlyCount = vertcat(triConvElements.PlyCount', triElements.PlyCount);
    elements.Density = vertcat(triConvElements.Density', triElements.Density);
else
    elements.CompositeIDs = triElements.CompositeIDs;
    elements.Thickness = triElements.Thickness;
    elements.PlyCount = triElements.PlyCount;
    elements.Density = triElements.Density;
end

end

%--------------------------------------------------------------------------

function [connect_table_tri] = quad2tri(connect_table_quad)
% Transformation of QUAD elements into TRI elements

% Input check 
if ~any(size(connect_table_quad)==4)
    error('Input the nodal connectivity table for QUAD4 elements')
end
nel_quad = size(connect_table_quad,1); % total number of QUA4 elements 

% Initilaize TRI3 data 
nel_tri = 2*nel_quad;
connect_table_tri = zeros(nel_tri,3);

connect_table_tri(1:2:end,:) = connect_table_quad(:,1:3);
connect_table_tri(2:2:end,:) = connect_table_quad(:,[1 3 4]);

end