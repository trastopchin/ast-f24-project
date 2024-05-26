## Analysis scripts

To reproduce the results shown in the paper:

(Do all the steps from the project root)

1. Run `pip install -r analysis/requirements.txt`

2. Run `python analysis/analysis.py`. This generates three files:

   - count_of_bug_types.pdf
   - distribution_of_bug_types.pdf
   - distribution_of_correctness_bug_errors.pdf

   as well as further statistics.

3. Run `python analysis/efficiency_bugs.py`. This generates the statistics about efficiency bugs.
   Flip the commented line around Line 10 to find Determination Bugs instead.
