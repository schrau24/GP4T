function CenterLegend(location)

if nargin==0
    location='northoutside';
end


a=findobj(gcf, 'Type', 'Legend');

for i=1:numel(a)
    a(i).Orientation = 'Horizontal';
    a(i).Location    = location;
end
end