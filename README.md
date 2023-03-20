[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-f4981d0f882b2a3f0472912d15f9806d57e124e0fc890972558857b51b24a6f9.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=10464145)
# Assignment 2 - Part 2: LTR from Interactions <a class="anchor" id="toptop"></a>

## Introduction
Welcome to the second part of the LTR assignment. In the previous part, you experimented with offline learning to rank. This assignment is on leanring to rank from interactions. You will learn to train an unbiased model and to estimate propensities using DLA (dual learning algorithm).

**Learning Goals**
- Simulate document clicks
- Implement biased and unbiased counterfactual learning to rank (LTR) and dual learning algorithms
- Evaluate and compare the methods

## Guidelines

### How to proceed?
We have prepared a notebook: `hw2-2.ipynb` including the detailed guidelines of where to start and how to proceed with this part of the assignment. Alternatively, you can use `hw2-2.py` if you prefer a Python script that can easily be run from the command line.
The two files are equivalent.

You can find all the code of this assignment inside the `ltr` package. The structure of the `ltr` package is shown below, containing various modules that you need to implement. For the files that require changes, a :pencil2: icon is added after the file name. This icon is followed by the points you will receive if all unit tests related to this file pass. 

📦ltr (85 points)\
 ┣ 📜dataset.py :pencil2: (10 points)\
 ┣ 📜eval.py\
 ┣ 📜logging_policy.py\
 ┣ 📜loss.py :pencil2: (30 points)\
 ┣ 📜model.py :pencil2: (3 points)\
 ┣ 📜train.py :pencil2: (42 points)\
 ┗ 📜utils.py

This assignment part is worth 130 points total. By completing all required implementations correctly, you can earn a total of 85 points. Another 30 points are rewarded based on your outputs, as indicated in the notebook. You can earn the remaining 15 points by completing the [analysis.md](analysis.md) file.

Make sure to __commit__ and __push__ the results files after training and running your models.

**NOTICE THAT YOU NEED TO PUT YOUR IMPLEMENTATION BETWEEN `BEGIN \ END SOLUTION` TAGS!** All of the functions that need to be implemented in modules directory files have a #TODO tag at the top of their definition. As always, it's fine to create additional code/functions/classes/etc outside of these tags, but make sure not to change the behavior of the existing API.


## Table of Contents

Table of contents:

 - _Chapter 1: Offline LTR (previous assignment)_
 - Chapter 2: Counterfactual LTR
    - Section 1: Dataset and utils
    - Section 2: Biased ListNet
    - Section 3: Unbiased ListNet
    - Section 4: Propensity estimation
    - Section 5: Evaluation
