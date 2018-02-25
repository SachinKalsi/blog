
## Distance between two points

- Lets say **A($${x_1}$$, $${y_1}$$) & B($${x_2}$$, $${y_2}$$)** be two points in a 2 dimentional geometry. 

    - Euclidean distance

        $$distance(A,B) = \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}$$
        
        - #### Geometric view
        ![]({{site.baseurl}}data/images/distance.jpg)
        
                
                A = (2, 2)
        
                B = (4, 4)
        
        $$distance(A,B) = \sqrt{(2-4)^2 + (2-4)^2} = \sqrt{8} =2\sqrt{2}$$
        
    - Manhatten distance

        $$distance(A,B) = |x_1-x_2| + |y_1-y_2|$$
    - Minkowski distance (Generalisation)
            
        $$distance(A,B) = ({|x_1-x_2|^p + |y_1-y_2|^p})^{1/p}$$
        
        where p = 1, 2, 3 ...
        
        if p == 1, minkowski distance will converge into Manhatten distance
        
        if p == 2, minkowski distance will converge into Euclidean distance

- If A & B points are in n dimentional geometry, then 

    $$A = ( {x_{a1},x_{a2}, x_{a3}, ....x_{an}} )$$

    $$B = ( {x_{b1},x_{b2}, x_{b3}, ....x_{bn}} )$$
    
    Then,
    
    $$Euclidean\_distance(A,B)  = \sqrt{(x_{a1}-x_{b1})^2 + (x_{a2}-a_{b2})^2+ ... + (x_{an}-x_{bn})^2}$$
    
    $$Manhatten\_distance(A,B) = |x_{a1}-x_{b1}| + |x_{a2}-a_{b2}| + .... + |x_{an}-a_{bn}|$$
    
    $$Minkowski\_distance= ({|x_{a1}-x_{b1}|^p + |x_{a2}-a_{b2}|^p+ ... + |x_{an}-x_{bn}|^p})^{1/p} $$
    
## Dot product and angle between 2 vectors
    
- a & b points are in n dimentional vectors, i.e., 

    $$a = ({a_1,a_2,a_3, ... a_n})$$

    $$b = ({b_1,b_2,b_3, ... b_n})$$
    
    Dot product of a & b = a.b
    
    $$a.b = a_1b_1 + a_2b_2 + a_3b_3 + ... + a_nb_n$$   
    
    $$a.b = \begin{bmatrix}{a_1} & {a_2} & {a_3} & {a_4}  .....  {a_n}\end{bmatrix} * \begin{bmatrix}{b_1}\\{b_2} \\ {b_3} \\ {b_4}  \\ . \\ . \\ {b_n}\end{bmatrix}$$
    
    $$a.b = a^Tb$$
    
    Also we know that 
    
    $$a.b = |a| * |b| * Cos(\theta) $$
    
    where a = distance of a from origin & b = distance of b from origin
    
    $$\theta =arccos(\frac{a^T.b}{|a| * |b|})$$