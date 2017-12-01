x = linspace(-200,200,1000);
y = linspace(-200,200,1000);
[X,Y] = meshgrid(x,y);

p0 = [0 3];
p1 = [-3 0];
p2 = [3 0];

function dist = distance(x, y)
        dist = x*x + x*y + y*y;
end

dist0 = arrayfun(@distance,X-p0(1),Y-p0(1));
dist1 = arrayfun(@distance, X-p1(1), Y-p1(1));
dist2 = arrayfun(@distance.X-p2(1),Y-p2(1));
