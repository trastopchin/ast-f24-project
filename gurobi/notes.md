```
docker run --env=GRB_CLIENT_LOG=3  \
             --volume=$PWD/gurobi.lic:/opt/gurobi/gurobi.lic:ro \
             --volume=$PWD/scripts:/scripts:ro \
             gurobi/python:11.0.1_3.10  /scripts/main.py
```
