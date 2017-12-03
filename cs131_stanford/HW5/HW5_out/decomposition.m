x = linspace(-200,200,100);
y = linspace(-200,200,100);
[X,Y] = meshgrid(x,y);
p0 = [0 3];
p1 = [-3 0];
p2 = [3 0];

function dist = distance(x, y)
        dist = x*x + x*y + y*y;
end

dist0 = arrayfun(@distance,X-p0(1),Y-p0(2));
dist1 = arrayfun(@distance, X-p1(1), Y-p1(2));
dist2 = arrayfun(@distance,X-p2(1),Y-p2(2));

first=[];
second=[];
third=[];
for i=1:100
	for j=1:100
		m=min([dist0(i,j),dist1(i,j),dist2(i,j)]);
		if m==dist0(i,j)
			first = [first; [y(i) x(j)]];
		elseif m== dist1(i,j)
			second = [second ;[y(i) x(j)]];
		else
			third = [third; [y(i) x(j)]];
		endif
	endfor
endfor

hold on;
scatter(first(:,2),first(:,1),[],1);
scatter(second(:,2),second(:,1),[],2);
scatter(third(:,2),third(:,1),[],3);
hold off;

