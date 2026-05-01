#### Method and Purpose
Rejection sampling 
- Spawn a huge pool of electrons, determine how likely it would be to find an electron there, they figure out if it should be there. Filter and we have a result based on previous knowledge

- Used to test out my rendering pipeline before I get into actually calculating things (so I get screwed over one thing at a time)

#### Limitations 
- Absolutely no predictive power (it is completely based on equations that have already been created and cannot be used to determine anything that does not have an equation already. 

#### Things to Implement  + Where to implement them 
rejection_sample.cu - quantum directory 
- It takes teh electron's position and the proton (neutron)'s position from the global environment property, calculates distance, computes the orbital probability density, uses a number and determines if it should be there. 

Monte Carlo Rejection Sampling algorithm
$$$\U \leq \frac{f(X)){Mg(X)}$$$ 
- Candidate X from the proposed distribution of g(x)
- Draws a random number U from interval (0, 1) 
- If the above equation holds, then candidate X is accepted
- If its not met, then it is rejected and continued until we get the desired number of candidates 
