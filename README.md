# Net-MARS 

**Net-MARS (Network Monitoring and Recovery Simulator)** is a simulation 
framework including libraries to visualize and process real computer 
network topologies, simulate disruption scenarios, implement damage 
assessment strategies providing probabilistic and partial knowledge, 
recovery algorithms, and commonly used routing protocols.

The somulator is the result of the work proposed in the paper ***"Tomography-based progressive network recovery
and critical service restoration after massive failures"*** published at IEEE INFOCOM 2023 by Viviana Arrigoni,
Matteo Prata and Novella Bartolini. To replicate the results on the paper, find the ```setup_01.py``` file at ```src.experimental_setup```. 
Compiled files are at ```data.infocom_2023```.

## Getting Started with *Net-MARS*
In order to get started, create and activate an environment running 
Python 3.8. Then run the required libraries in the requirements file

```pip install -r requirements.txt ```

Now you can run your first simulation using an input graph. The simulation 
can be batched as to run multiple recovery algorithms, in different setups 
in parallel. The output of a simulation will be a file.

<image width=800 height=500 src="data/git-images/topology-minnesota.PNG"/>

To run a parallel simulation, add your setup file at ```src.experimental_setup```. 
This must contain the following dictionaries. 

```
comparison_dims = {co.IndependentVariable.SEED: range(5),
                   co.IndependentVariable.ALGORITHM: [
                        co.Algorithm.TOMO_CEDAR,
                        co.Algorithm.ORACLE,
                        co.Algorithm.CEDAR,
                        co.Algorithm.ST_PATH,
                        co.Algorithm.SHP,
                        co.Algorithm.ISR_SP,
                        co.Algorithm.ISR_MULTICOM
                        ]
                   }
```

This dictionary represents the dimensions for comparison, the set of seeds, 
and the set of recovery algorithms described in [REF?]. The set of available algorithms 
is expressed in the enumeration at ```src.constants.Algorithm```.

```
indv_vary = {
    co.IndependentVariable.PROB_BROKEN: [.3, .4, .5, .6, .7, .8],
    co.IndependentVariable.MONITOR_BUDGET: [20, 22, 24, 26, 28, 30],
    co.IndependentVariable.N_DEMAND_EDGES: [4, 5, 6, 7, 8],
    co.IndependentVariable.FLOW_DEMAND: [10, 15, 20, 25, 30],
}
```

This dictionary represents the independent variables to test, that are enumerated at 
```src.constants.IndependentVariable```. ***PROB_BROKEN*** is percentage of broken elements 
in the network, ***MONITOR_BUDGET*** the maximum number of monitors that can be placed,
***N_DEMAND_EDGES*** the numbe of demand edge, ***FLOW_DEMAND*** the demand flow in per demand edge.
Each independent variable varies in the specified domain.

```
indv_fixed = {
    co.IndependentVariable.PROB_BROKEN: .8,
    co.IndependentVariable.MONITOR_BUDGET: 20,
    co.IndependentVariable.N_DEMAND_EDGES: 8,
    co.IndependentVariable.FLOW_DEMAND: 30,
}
```

This dictionary represents the fixed value for each variable, when another varies.

To run a simulation in parallel to every core, run with _-par 1_, with the MINNESOTA graph and setup01.
``` 
python -m src.main -set setup_01 -gn MINNESOTA -par 1
```


## Contacts
For further information contact Matteo Prata at [prata@di.uniroma1.it](mailto:prata@di.uniroma1.it) and 
Viviana Arrigoni at [arrigoni@di.uniroma1.it](mailto:arrigoni@di.uniroma1.it).



