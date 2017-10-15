function output = gaussian_2nd(sig, x)
   output = (-1/(sqrt(2*pi)*sig**3))*e**(-x**2/(2*sig**2))*(1-(x**2/sig**2)); 
end