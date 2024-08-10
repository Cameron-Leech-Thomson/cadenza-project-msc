# Modified Cadenza Project (PyClarity) System for MSc Research Project

### A modification of the baseline project from The Cadenza Project (ICASSP 2024 Grand Challenge), created as part of a MSc Research Project at The University of Leeds.

## Installation

### Requirements

- Numpy
- PyTorch
- MatplotLib
- Librosa
- Audio_DSPy
- re
- Subprocess
- Random
- os
- Soundfile
- DEAP

### System

To install the baseline PyClarity system, see the instructions here: https://github.com/claritychallenge/clarity. Upon installing PyClarity and it's data, simply replace the baseline `enhance.py` file with the one in this repository, and make sure to modify the `config.yaml` file in the baseline code to include the relevant file paths to the data, as well as the file paths inside the modified `enhance.py`.

### Data

Instructions on downloading the MUSDB18-HQ dataset can be found on the Cadenza Project site: https://cadenzachallenge.org/docs/icassp_2024/take_part/download.

## Usage

The system has one of two functions to call in `main()`. First, `enhance()` will run the system on the default Equalisation parameters, if none are provided, or will use an individuals parameters if there are any within the `baseline/eqtraining` directory. The `run_ga()` function will begin the training of the genetic algorithm, with a default population, `pop` of 50 and default number of generations `nGen` of 50, outputting the most successful individual of each generation into `baseline/eqtraining`, as well as a sample of the HAAQI scores they produced. Once the algorithm has trained an individual with effective enough parameters to produce high HAAQI scores, great! It's time to return to `enhance()` and perform the enhancement of the entire dataset, before running the baseline systems `evaluate.py` to generate the HAAQI scores for all of the new enhanced signals en masse.

Running the system is as simple as a call to the terminal:
```
python enhance.py
```

Happy Training!

## Individuals:

The best individual of each generation is saved as a .JSON file, called `scores_<mean of parameters>.JSON`. The .JSON file is in the following format:
```
{
  "Individual": [
  <Filter 1 Q-Factor>,
  <Filter 2 Q-Factor>,
  <Filter 2 Gain>,
  <Filter 3 Q-Factor>,
  <Filter 3 Gain>,
  <Filter 4 Q-Factor>,
  <Filter 4 Gain>,
  <Filter 5 Q-Factor>,
  <Filter 5 Gain>,
  <Filter 6 Q-Factor>,
  <Filter 6 Gain>,
  <Filter 7 Q-Factor>,
  <Filter 7 Gain>,
  <Filter 8 Q-Factor>,
  <Filter 8 Gain>
  ], 
      "HAAQI": {
            "<Scene Label>: <Track Name>,  Listener: <Listener Label>": {
                  "LeftScore": <Left Ear HAAQI Score>, 
                  "RightScore": <Right Ear HAAQI Score>, 
                  "MeanScore": <Mean HAAQI Score>
            },
            ... x 10
            }
}
```
