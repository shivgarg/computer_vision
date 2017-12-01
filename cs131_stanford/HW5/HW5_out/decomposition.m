x = linspace(-200,200,100);
y = linspace(-200,200,100);
[X,Y] = meshgrid(x,y);

p0 = [0 3];
p1 = [-3 0];
p2 = [3 0];

function dist = distance(x, y)
        dist = x*x + x*y + y*y;
end

dist0 = arrayfun(@distance,X-p0(1),Y-p0(1));
dist1 = arrayfun(@distance, X-p1(1), Y-p1(2));
dist2 = arrayfun(@distance,X-p2(1),Y-p2(2));
size(dist0)

first=[];
second=[];
third=[];
for i=1:100
	disp i
	for j=1:100
		m=min([dist0(i,j),dist1(i,j),dist2(i,j)]);
		if m==dist0(i,j)
			first = [first [i j]];
		elseif m== dist1(i,j)
			second = [second [i j]];
		else
			third = [third [i j]];
		endif
	endfor
endfor

hold on;
scatter(first(:,1),first(:,2),'o');
scatter(second(:,1),second(:,2),'+');
scatter(third(:,1),third(:,2),"*");
hold off;
