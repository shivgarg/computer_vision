x=linspace(-5,5,200);

k=2;
y=arrayfun(@(x) (gaussian(k,x) - gaussian(1,x))/(k-1),x);


plot(x,y)