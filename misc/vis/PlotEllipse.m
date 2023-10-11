function PlotEllipse(xbar,ybar,hor_axis,vert_axis,c)

if nargin<5
    c='m';
end

hold on;
phi = linspace(0,2*pi,50);
cosphi = cos(phi);
sinphi = sin(phi);


% xbar = s(k).Centroid(1);
% ybar = s(k).Centroid(2);

% a = s(k).MajorAxisLength/2;
% b = s(k).MinorAxisLength/2;

theta = 0;%pi*s(k).Orientation/180;
R = [ cos(theta)   sin(theta)
     -sin(theta)   cos(theta)];

xy = [hor_axis*cosphi; vert_axis*sinphi];
xy = R*xy;

x = xy(1,:) + xbar;
y = xy(2,:) + ybar;

% plot(x,y,'r','LineWidth',2);

alpha = 0.3;
h=fill(x,y,c);
set(h,'facealpha',alpha);
set(h,'edgecolor','m');
set(h,'EdgeAlpha',alpha);
hold off;

end