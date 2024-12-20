## A Recurrent Reinforcement Learning Strategy with a Parameterized Agent for Online Scheduling of a State Task Network Under Epistemic and Aleatoric Uncertainty

This repository contains the codes used to obtain the results from the manuscript _"A Recurrent Reinforcement Learning Strategy with a Parameterized Agent for Online Scheduling of a State Task Network Under Epistemic and Aleatoric Uncertainty"_. The first 8 experiments (i.e., from Exp1 to Exp8) correspond to section 4.1 and the Exp9 and Exp10 to section 4.2. Below, a brief description of the purpose of each experiment is listed.

To run the environment of the plants for Cases Study 1 and 2 (available at each ExpX file), the user must download gym version 0.26.2 and install the environment for each experiment.

Exp1: MDP hybrid Agent implemented on the deterministic environment of Case Study 1.

Exp2: POMDP hybrid Agent implemented on the deterministic environment of Case Study 1.

Exp3: MDP hybrid Agent implemented on the environment of case Study 1 with epistemic uncertainty in Production at Task 2. 

Exp4: POMDP hybrid Agent implemented on the environment of case Study 1 with epistemic uncertainty in Production at Task 2. 

Exp5: MDP hybrid Agent implemented on the environment of case Study 1 with aleatoric uncertainty applying Gaussian Noise in the measurement of the states

Exp6: POMDP hybrid Agent implemented on the environment of case Study 1 with aleatoric uncertainty applying Gaussian Noise in the measurement of the states

Exp7: MDP hybrid Agent implemented on the environment of case Study 1 with random interference in the input observation window representing aleatoric uncertainty

Exp8: POMDP hybrid Agent implemented on the environment of case Study 1 with random interference in the input observation window representing aleatoric uncertainty

Exp9: POMDP hybrid agent implemented on Case Study 2 without masking method.

Exp10: POMDP hybrid agent implemented on Case Study 2 with masking method.
