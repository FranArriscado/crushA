%
%                  abaqusInpRead.m
%
%                     Tiago Cardoso
%                      up202006481
%                      FEUP, 2025
%
% SUPPORTING FUNCTION FILE
%==========================================================================
%==========================================================================
function [FEModelData, elementOut, eType] = abaqusInpRead(fileName, log)

%==========================================================================
%==========================================================================

% MODIFIED VERSION: TIAGO CARDOSO, up202006481, FEUP 2025

% - Addition of 'Sections' struct to extract the material/layup and plies
% properties (ply thickness, ply material, ply count) corresponding to each
% section of the model
% - Addition of 'Assembly' and related fields in other structures
% - Addtion of 'Elements.MaterialLayup' array with information of material/layup
% name corresponding to each element
% - Addtion of 'Elements.Thickness' array with information of thickness
% corresponding to each element (sum of plies thickness)
% - Addtion of 'Elements.PlyCount' array with information of total number of 
% plies corresponding to each element (sum of number of plies)
% - Modification of 'getMatrixFromString' function: addition of 'myMatrixString'
% variable to obtain the material name corresponding to each ply

%==========================================================================
%==========================================================================

% read abaqus input file, e.g. fileName = 'myExamInpFile.inp'
% coded by Wan Ji, Wuhan University
% update date 2021/07/15
% last version "readinp", refer to
% https://www.mathworks.com/matlabcentral/fileexchange/95343-read-abaqus-input-file-to-get-the-nodes-and-elements
% something new to this version
% (1) fixed the bug which the function can't read c3d20 element
% (2) offer optional output, that is, the struct output

%%
% If the number of output arguments is only 1, then a structure output is
% activated, the struct includes some important information inside the inp
% file, like "Parts", "Nodes", "Elements", "NodeSets", "ElementSets", "Materials",
% etc.
% Example 01
% fileName = 'myExamInpFile.inp';
% data = abaqusInpRead(fileName);
%%
% If the number of output arguments is 3, then the 1st set of node
% coordinates, the 1st set of elements, and element type will be the output
% data.
% Example 02
% fileName = 'myExamInpFile.inp';
% [node, element, elementType] = abaqusInpRead(fileName);
%% Function body

log(sprintf('Reading .inp file...'));

s = fileread(fileName);
s = lower(s);
s = split(s,'*');
% Substructures of output structure
ElementSets = struct(...
    'ElementSetType',[],...
    'Name',[],...
    'Data',[],...
    'Instance',[],...
    'Generate',[],...
    'Internal',[],...
    'Count', 0, ...
    'AttachedPartTag', [],...
    'AttachedPartID',[]);
Parts = struct('Tags', [],...
    'Count', 1,...
    'PartID',[]);
Nodes = struct(...
    'Coordinates', [],...
    'ID', [],...
    'NodeTag',[],...
    'AttachedPartTag',[],...
    'AttachedPartID',[],...
    'Count', 0);
Elements = struct(...
    'ElementType', [],...
    'ID',[],...
    'NodeIDList',[],...
    'LayupName',[],...
    'Thickness',[],...
    'PlyCount',[],...
    'ElementTag',[],...
    'AttachedNodeTag',[],...
    'AttachedPartTag', [],...
    'AttachedPartID',[],...
    'Count', 0);
Materials = struct(...
    'Name',[],...
    'Count', 0, ...
    'Density', 0, ...
    'CrushStress0', [], ...
    'CrushStress45', []);
NodeSets = struct(...
    'NodeSetType',[],...
    'Name',[],...
    'Data',[],...
    'Instance',[],...
    'Generate',[],...
    'Internal',[],...
    'Count',0,...
    'AttachedPartTag',[],...
    'AttachedPartID',[]);
Assembly = struct(...
    'CountInstance',0, ...
    'InstanceName',[], ...
    'Part',[], ...
    'PartID',[], ...
    'TranslationVec',[], ...
    'RotationPoint1',[], ...
    'RotationPoint2',[], ...
    'RotationAngle',[],...
    'RotationAxis',[]);
Sections = struct(...
    'Count', 0,...
    'ID',0,...
    'LayupName',[],...
    'ElementSetsName',[],...
    'ElementID',[],...
    'PlyThickness', [],...
    'PlyMaterial',[],...
    'PlyOrientation',0,...
    'PlyCount', 0,...
    'AttachedPartTag', [],...
    'AttachedPartID',[]);
% the output data
FEModelData = struct(...
    'Parts',Parts, ...
    'Nodes', Nodes, ...
    'Elements', Elements, ...
    'Materials',Materials, ...
    'NodeSets', NodeSets, ...
    'ElementSets', ElementSets,...
    'Assembly', Assembly,...
    'Sections', Sections...
    );
partFlag = true;
for i = 1:1:numel(s)
    [myMatrix, myMatrixString, FLS]= getMatrixFromString(s{i}, newline, ',');
    FLS(FLS==char(32)|FLS==char(13)|FLS==newline) = [];
    p = split(FLS,',');
    q = p{1};
    if(isempty(q))
        continue;
    end
    switch lower(q)
        case {'part'}
            if(partFlag)
                partFlag = ~partFlag;
            else
                Parts(1).Count = Parts(1).Count + 1;
            end
            c = Parts(1).Count;
            Parts(c,1).PartID = c;
            partName = split(p{2},'=');
            Parts(c,1).Tags = partName{2};
        case {'node'}
            Nodes(1).Count = Nodes(1).Count + 1;
            c = Nodes(1).Count;
            Nodes(c,1).AttachedPartTag =  Parts(Parts(1).Count).Tags;
            Nodes(c,1).AttachedPartID =  Parts(1).Count;
            idcoor = cell2mat(myMatrix);
            Nodes(c,1).ID = idcoor(:,1);
            Nodes(c,1).Coordinates = idcoor(:,2:end);
            Nodes(c,1).NodeTag = Nodes(1).Count;
        case {'element'}
            Elements(1).Count = Elements(1).Count + 1;
            c = Elements(1).Count;
            Elements(c,1).AttachedPartTag =  Parts(Parts(1).Count).Tags;
            Elements(c,1).AttachedPartID =  Parts(1).Count;
            eType = split(p{2},'=');
            eType = eType{2};
            eNodeNumber = eType(4:end);
            eNodeNumber(double(eNodeNumber)>double('9')|double(eNodeNumber)<double('0')) = [];
            eNodeNumber = str2double(eNodeNumber);
            Elements(c,1).ElementType = eType;
            if(numel(myMatrix)>=2 && (numel(myMatrix{1})+numel(myMatrix{2})-1==eNodeNumber))
                myMat = cell(numel(myMatrix)/2,1);
                for j = 1:1:numel(myMat)
                    myMat{j,1} = [myMatrix{2*j-1,1}, myMatrix{2*j,1}];
                end
            elseif(numel(myMatrix)>=3&& ...
                    (numel(myMatrix{1})+numel(myMatrix{2})+numel(myMatrix{3})-1==eNodeNumber))
                myMat = cell(numel(myMatrix)/3,1);
                for j = 1:1:numel(myMat)
                    myMat{j,1} = [myMatrix{3*j-2,1}, myMatrix{3*j-1,1}, myMatrix{3*j,1}];
                end
            else
                myMat = myMatrix;
            end
            Elements(c,1).NodeIDList = cell2mat(myMat);
            Elements(c,1).ID = Elements(c,1).NodeIDList(:,1);
            Elements(c,1).NodeIDList = Elements(c,1).NodeIDList(:,2:end);
            Elements(c,1).ElementTag = Elements(1).Count;
            Elements(c,1).AttachedNodeTag = Nodes(1).Count;
        case {'nset','ngen'}
            NodeSets(1).Count = NodeSets(1).Count + 1;
            c = NodeSets(1).Count;
            NodeSets(c,1).Generate = false;
            NodeSets(c,1).Internal = false;
            for j = 1:1:numel(p)
                ep = split(p{j},'=');
                switch ep{1}
                    case {'instance'}
                        NodeSets(c,1).Instance = ep{2};
                    case {'generate'}
                        NodeSets(c,1).Generate = true;
                    case {'internal'}
                        NodeSets(c,1).Internal = true;
                    case {'nset','ngen'}
                        if(numel(ep)>=2)
                            NodeSets(c,1).Name = ep{2};
                        end
                    otherwise
                end
            end
            NodeSets(c,1).Data = cell2mat(myMatrix');
            NodeSets(c,1).NodeSetType = lower(q);
            NodeSets(c,1).AttachedPartTag =  Parts(Parts(1).Count).Tags;
            NodeSets(c,1).AttachedPartID =  Parts(1).Count;
        case {'elset'}
            ElementSets(1).Count = ElementSets(1).Count + 1;
            c = ElementSets(1).Count;
            ElementSets(c,1).Generate = false;
            ElementSets(c,1).Internal = false;
            for j = 1:1:numel(p)
                ep = split(p{j},'=');
                switch ep{1}
                    case {'instance'}
                        ElementSets(c,1).Instance = ep{2};
                    case {'generate'}
                        ElementSets(c,1).Generate = true;
                    case {'internal'}
                        ElementSets(c,1).Internal = true;
                    case {'elset'}
                        if(numel(ep)>=2)
                            ElementSets(c,1).Name = ep{2};
                        end
                    otherwise
                end
            end
            ElementSets(c,1).Data = cell2mat(myMatrix');
            ElementSets(c,1).ElementSetType = lower(q);
            ElementSets(c,1).AttachedPartTag =  Parts(Parts(1).Count).Tags;
            ElementSets(c,1).AttachedPartID =  Parts(1).Count;
        case {'shellsection'}
            Sections(1).Count = Sections(1).Count+1;
            c = Sections(1).Count;
            Sections(c,1).ID = c;
            Sections(c,1).AttachedPartTag =  Parts(Parts(1).Count).Tags;
            Sections(c,1).AttachedPartID =  Parts(1).Count;
            for j=1:1:numel(p)
                ep = split(p{j},'=');
                switch ep{1}
                    case{'elset'}
                        Sections(c,1).ElementSetsName = ep{2};
                    case{'material'}
                        Sections(c,1).MaterialLayup = ep{2};
                    case{'layup'}
                        Sections(c,1).LayupName = ep{2};       
                    case{'composite'}
                        Sections(c,1).PlyCount = numel(myMatrix);
                        for j=1:1:numel(myMatrix)
                            ep = split(sprintf('%.15g,',myMatrix{j}),',');
                            Sections(c,1).PlyThickness(j,1) = str2double(ep{1});
                            Sections(c,1).PlyMaterial{j,1} = string(myMatrixString{j+1}(3));
                            Sections(c,1).PlyOrientation(j,1) = str2double(ep{3});
                        end
                    otherwise
                end
            end

        case {'material'}
            Materials(1).Count = Materials(1).Count + 1;
            c = Materials(1).Count;
            for j = 1:1:numel(p)
                ep = split(p{j},'=');
                switch ep{1}
                    case {'name'}
                        Materials(c,1).Name = ep{2};
                    otherwise
                end
            end

        case {'density'}
            c = Materials(1).Count;
            for j=1:1:numel(myMatrix)
                ep = split(sprintf('%.15g,',myMatrix{j}),',');
                Materials(c,1).Density(j,1) = str2double(ep{1});
                Materials(c,1).Density(j,1) = Materials(c,1).Density(j,1) * 10^12; % tonne/mm^3 to kg/m^3
            end

        case {'crushstress'}
            c = Materials(1).Count;
            for j=1:1:numel(myMatrix)
                ep = split(sprintf('%.15g,',myMatrix{j}),',');
                ep = strtrim(ep);  % Remove extra spaces if needed
                if numel(ep) >= 3 && ~isempty(ep{3})
                    if str2double(ep{2}) == 0 && str2double(ep{3}) == 0
                        Materials(c,1).CrushStress0 = str2double(ep{1});
                    elseif str2double(ep{2}) == 45 && str2double(ep{3}) == 0
                        Materials(c,1).CrushStress45 = str2double(ep{1});
                    end
                else
                    if str2double(ep{2}) == 0
                        Materials(c,1).CrushStress0 = str2double(ep{1});
                    elseif str2double(ep{2}) == 45
                        Materials(c,1).CrushStress45 = str2double(ep{1});
                    end
                end
            end

        case{'instance'}
            Assembly(1).CountInstance = Assembly(1).CountInstance + 1;
            c = Assembly(1).CountInstance;
            for j = 1:1:numel(p)
                ep = split(p{j},'=');
                switch ep{1}
                    case {'name'}
                        Assembly(c,1).InstanceName = ep{2};
                    case {'part'}
                        Assembly(c,1).Part = ep{2};
                        for k=1:Parts(1).Count
                            if strcmp(ep{2}, Parts(k,1).Tags)
                                partID = Parts(k,1).PartID;
                                Assembly(c,1).PartID = partID;
                            end
                        end
                    otherwise
                end
            end
            if numel(myMatrix)>0
                ep = split(sprintf('%.15g,',myMatrix{1}),',');
                if length(ep)<5
                    Assembly(c,1).TranslationVec = [str2double(ep{1});str2double(ep{2});str2double(ep{3})];
                else
                    Assembly(c,1).RotationPoint1 = [str2double(ep{1}), str2double(ep{2}), str2double(ep{3})];
                    Assembly(c,1).RotationPoint2 = [str2double(ep{4}), str2double(ep{5}), str2double(ep{6})];
                    Assembly(c,1).RotationAngle = str2double(ep{7});
                end
                if numel(myMatrix)>1
                    ep = split(sprintf('%.15g,',myMatrix{2}),',');
                    Assembly(c,1).RotationPoint1 = [str2double(ep{1}), str2double(ep{2}), str2double(ep{3})];
                    Assembly(c,1).RotationPoint2 = [str2double(ep{4}), str2double(ep{5}), str2double(ep{6})];
                    Assembly(c,1).RotationAngle = str2double(ep{7});
                    %Assembly(c,1).RotationAxis = Assembly(c,1).RotationPoint2 - Assembly(c,1).RotationPoint1;
                end
            end
        otherwise
            continue

    end
end
FEModelData.Parts = Parts;
FEModelData.Nodes = Nodes;
FEModelData.Elements = Elements;
FEModelData.NodeSets = NodeSets;
FEModelData.ElementSets = ElementSets;
FEModelData.Materials = Materials;
FEModelData.Sections = Sections;
FEModelData.Assembly = Assembly;
if(nargout>=2)
    elementOut = FEModelData.Elements(1).NodeIDList;
    FEModelData = FEModelData.Nodes(1).Coordinates;
    eType = Elements(1).ElementType;
end

% Assign material/layup, thickness and ply count information to each element
for i=1:1:length(FEModelData.Sections)
    index_elset = find(arrayfun(@(ElementSets) strcmp(ElementSets.Name, FEModelData.Sections(i).ElementSetsName), FEModelData.ElementSets));
    if FEModelData.ElementSets(index_elset).Generate == 1
        FEModelData.Sections(i).ElementID = [FEModelData.ElementSets(index_elset).Data(1):FEModelData.ElementSets(index_elset).Data(3):FEModelData.ElementSets(index_elset).Data(2)]';
    else
        FEModelData.Sections(i).ElementID = FEModelData.ElementSets(index_elset).Data;
    end
    for j=1:length(FEModelData.Sections(i).ElementID)
        elements_material{FEModelData.Sections(i).ElementID(j),1} = FEModelData.Sections(i).LayupName;
        elements_thickness{FEModelData.Sections(i).ElementID(j),1} = sum(FEModelData.Sections(i).PlyThickness);
        elements_plycount{FEModelData.Sections(i).ElementID(j),1} = FEModelData.Sections(i).PlyCount;
    end
end

FEModelData.Elements(1).LayupName = elements_material;
elements_thickness = cellfun(@(x) ternaryNaN(x), elements_thickness);
elements_plycount = cellfun(@(x) ternaryNaN(x), elements_plycount);
FEModelData.Elements(1).Thickness = elements_thickness;
FEModelData.Elements(1).PlyCount = elements_plycount;

end

%--------------------------------------------------------------------------

function [myMatrix, myMatrixString, firstLineString] = getMatrixFromString(s, sepRow, sepColumn)
% get the matrix from a string
ssep = split(s, sepRow);
myMatrix = cell(numel(ssep),1);
myMatrixString = cell(numel(ssep),1);
myFlag = true(numel(ssep),1);
columnNumber = zeros(numel(ssep),1);
firstLineString = ssep{1};
for i = 1:1:numel(ssep)
    es = ssep{i};
    es(es==char(32)|es==char(13)|es==newline) = [];
    p = split(es, sepColumn);
    myMatrix{i} = zeros(1, numel(p));
    myMatrixString{i} = cell(1, numel(p));
    columnNumber(i) = numel(p);
    for j = 1:1:numel(p)
        val = str2double(p{j});
        myMatrix{i}(1,j) = val;
        if isnan(val)
            myMatrixString{i}{1,j} = p{j};
        end
    end
    myMatrix{i}( isnan(myMatrix{i}))=[];
    myMatrixString{i}(isempty(myMatrix{i}))=[];
    myMatrixString{1} = [];
    if(isempty(myMatrix{i}))
        myFlag(i) = false;
    end
end
myMatrix = myMatrix(myFlag,1);
end

%--------------------------------------------------------------------------

function out = ternaryNaN(x)
    if isempty(x)
        out = NaN;
    else
        out = x;
    end
end