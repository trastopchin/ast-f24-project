# AST FS24 Project

## Steps to reproduce data and results

1. Run `download_initial_dataset.sh` to download and extract the MIPLIB dataset of
   Mixed Integer Programming optimization problems. This downloads a 3GB zip file
   and extracts it into the `input` directory.

2. Install CPLEX. For this step, one should get their own academic installer
   from the IBM website (available [here](http://ibm.biz/CPLEXonAI)). Running
   `download_cplex.sh` will attempt to download the Linux installer under Nahin's
   account but may fail in the future. The installer is around 650MB in size. Run
   the installer and follow the instructions. In our case, we installed to
   `/opt/ibm/ILOG/CPLEX_Studio2211` on a Linux machine.

3. Install CPLEX Python integration. For this, run
   `python3 /opt/ibm/ILOG/CPLEX_Studio2211/python/setup.py install`

4. Install Gurobi. Just run: `pip install gurobipy`

5. Download the Gurobi (academic) license file. Again, one should get their own
   license from the Gurobi wesbite, and click download to get a `gurobi.lic` file.
   Store it at the root of this project directory (same place as this README file).

6. Install tqdm. Just run: `pip install tqdm`

7. Run `python main.py` to run the automated tests. Let it generate `results.txt`. We ran this step for approximately 24h.

8. Analyze the results using the instructions found in the `analysis/` directory.
