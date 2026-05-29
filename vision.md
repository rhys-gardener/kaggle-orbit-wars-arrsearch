# My vision for this initiative

I was seeing various issues with previous iterations of the kaggle orbit wars entries:
- Training time too long 
- Inefficient exploration of the solution space

This new initiative in this repo aims to solve those things by removing requirements on the kaggle env when training. Instead we can initialise what the observations of planet locations will look like once from the env, store that as an array. Then for every planet for every potential step we will be able to tell which planets will be reachable and when. Fleets are 'just' stored as transformations to apply to the planet at that step rather than needing to go through the env

Essentially, I see a training workflow as something roughly like:
- Use the kaggle env to initialise a solar system with observations
- Use the orbital velocity to predict movements of planets. Store a 500 step array with all these observations
Then during training:
- Train 4 agents in parallel generally
- At each step:
    - Apply results of fleets intersecting with planets
    - Do new actions of launching fleets at other planets 
    - Etc
    But only in the arrays, I.e don't need to use the kaggle game env
    - Do a batch load of training on one game state so we don't need to calculate a new set of obs everytime we do a new game. 

This later converts easily to the real env, since we've already done the hard work in calculating the launch angle. So the 'action' is deciding the planets to launch at, then we just calculate the angle using code already developed. not sure if that's in this repo or ../kaggle_orbit_wars_rg

In theory, this should make training a lot faster. I'm hoping it will open up some more avenues for mcts exploration in future, but initially MLP might have a go. Doing 4 agents concurrently improves solution space search. 
