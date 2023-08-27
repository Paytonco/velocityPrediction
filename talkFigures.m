clc;clear;
X = readtable("./data/Pancreas/Pancreas.csv");
X = sortrows(X,'t');

pt = X.t;
x = X.x1;
y = X.x2;
vx = X.v1;
vy = X.v2;

%%
figure(1)
scatter(x,y,[],pt,'filled')
cb = colorbar;
hold on 
quiver(x,y,vx,vy,2)

cb.Label.String = "Psuedotime";
cb.Label.Rotation = 270;
cb.Label.VerticalAlignment = "bottom";
set(gca,'XColor', 'none','YColor','none')
set(gca, 'color', 'none');

xlabel('PC_1')
ylabel('PC_2')


%%
figure(2)
poi = [0.4 3];
vec = [2,-3];
points = [0, 4;
          0.125, 2.123;
          0.5, 2.613;
          1, 2.78;
          1.234, 2.214;
          1.2, 2;
          1.1, 1];

t = 1:7;

scatter(points(:,1),points(:,2),200,t,'filled')
hold on
quiver(poi(1),poi(2),vec(1),vec(2),0.25,'g')
scatter(poi(1),poi(2),200,'g','filled')

set(gca,'XColor', 'none','YColor','none')
set(gca, 'color', 'none');

%%
figure(3) 

poi = [0 0];
points = [0 5;
          4 -3;
          -4 -3];

scatter(points(:,1),points(:,2),200,'filled')
hold on
%for n = 1:length(points)
%    Delta = points(n,:) - poi;
%    Delta = Delta / norm(Delta);
%    quiver(poi(1),poi(2),Delta(1),Delta(2),3,'g')
%end
quiver(poi(1),poi(2),1,1,3,'g')
scatter(poi(1),poi(2),200,'filled','g')

set(gca,'XColor', 'none','YColor','none')
set(gca, 'color', 'none');