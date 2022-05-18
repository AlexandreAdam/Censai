# Free-Form Reconstruction of Gravitational Lenses using Recurrent Inference Machine 

[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)

<img src="https://raw.githubusercontent.com/AlexandreAdam/Censai/dev/.github/images/random_sample.jpg" alt="" style="height: 1000px; width:1000px;"/>

## Abstract
Modeling strong gravitational lenses in order to quantify the distortions of the background sources
and reconstruct the mass density in the foreground lens has traditionally been a major computational
challenge. As the quality of gravitational lens images increases, the task of fully exploiting the infor-
mation they contain becomes computationally and algorithmically more difficult. In this work, we use
a neural network based on the Recurrent Inference Machine (RIM) to simultaneously reconstruct an
undistorted image of the background source and the lens mass density distribution as pixelated maps.
The method we present iteratively reconstructs the model parameters (the source and density map
pixels) by learning the process of optimization of their likelihood given the data using the physical
model (a ray tracing simulation), regularized by a prior implicitly learnt by the neural network through
its training data. When compared to more traditional parametric models, the method we propose is
significantly more expressive and can reconstruct complex mass distribution, which we demonstrate
by using realistic lensing galaxies taken from the hydrodynamical IllustrisTNG simulation .


## Contributors
<table>
  <tr>
    <td align="center"><a href="https://github.com/AlexandreAdam"><img src="https://avatars.githubusercontent.com/u/40675952?s=96&v=4" width="100px;" alt=""/><br /><sub><b>Alexandre Adam</b></sub>  </td>
    <td align="center"><a href="https://mila.quebec/en/person/laurence-perreault-levasseur"><img src="https://avatars.githubusercontent.com/u/13594101?v=4" width="100px;" alt=""/><br /><sub><b>Laurence Perreault-Levasseur</b></sub>  </td>
    <td align="center"><a href="https://www.astro.umontreal.ca/~hezaveh/hezaveh/Home.html"><img src="https://avatars.githubusercontent.com/u/4911735?v=4" width="100px;" alt=""/><br /><sub><b>Yashar Hezaveh</b></sub>  </td>
  </tr>
</table>
