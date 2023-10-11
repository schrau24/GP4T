function out=demean(input,dimension)

    if nargin<2
        dimension=1;
    end


    out = input-mean(input,dimension);
    
end


