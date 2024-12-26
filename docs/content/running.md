# Running Experiments

The idea behind EMAT is to run a number of experiments, and then analyze the 
results of those experiments.  The number of experiments that needs to be run
is a function of the level of complexity of the EMAT scope, but in general it
is more experiments that a user would want to run manually.  Thus, the EMAT
toolset is designed to automate the process of running experiments.

When working with VisionEval, at least as defined in this demonstration repository,
we will be treating the VisionEval model as a "files-based core model".  Doing so
requires a few steps for each experiment:

- Prepare the input files for the VisionEval model, based on the values of policy 
  levers and exogenous uncertainties defined for the experiment.
- Run the VisionEval model, using the input files that have been prepared.
- (Optional) Run any post-processing steps that are needed to extract the results
  of the experiment from the output files of the VisionEval model. 
- Collect the output files from the VisionEval model and parse then to extract the
  results of the experiment.

Each of these steps is encapsulated in a Python function that is part of the 
[`FilesCoreModel`](https://tmip-emat.github.io/source/emat.models/files-api.html) 
interface.  In the implementation code, you will see a class that is a subclass of
`FilesCoreModel`, and that class will define the specific steps needed to prepare
the input files, run the model, and extract the results.

``` python
from emat.model.core_files import FilesCoreModel

class VEModel(FilesCoreModel): # (1)
	"""
	A class for using Vision Eval as a files core model.
	"""
	...
```

1.  The `VEModel` class is a subclass of `FilesCoreModel`, which defines the specific
    steps needed to prepare the input files, run the model, and extract the results.


## Setting Up an Experiment

Each experiment involves making a complete copy of the VisionEval model in a contained 
environment, and then modifying the input files for that copy of the VisionEval model
to reflect the specific values of policy levers and exogenous uncertainties for that
experiment.  The `FilesCoreModel` interface defines the `setup` method as the place to
create a new copy of the VisionEval model in a contained environment, and then modify 
the input files for that copy of the VisionEval model. The `setup` method needs to be
overloaded in a subclass of `FilesCoreModel` to define the specific steps needed to
modify the input files for the experiment.

``` python
class VEModel(FilesCoreModel):
    ...
	def setup(self, params: dict): # (1)
		"""
		Configure the core model with the experiment variable values.

		Args:
			params (dict):
				experiment variables including both exogenous
				uncertainty and policy levers

		Raises:
			KeyError:
				if a defined experiment variable is not supported
				by the core model
		"""
```

1.  The `setup` method accepts a dictionary of parameters, which includes the values of
    policy levers and exogenous uncertainties for the experiment.

Within the `setup` method, the subclass of `FilesCoreModel` will need to make a complete
copy of the VisionEval model in a contained environment, and then modify the input files
for that copy of the VisionEval model to reflect the specific values of policy levers and
exogenous uncertainties for that experiment. 

There are numerous possible ways to prepare the input files for the VisionEval model, 
depending on the exploratory scope and the types of inputs that need to be modified.
This demo repository includes a few different examples of how to prepare input files 
based on the scope.

### Categorical Drop-In

Many of the input files for VisionEval are in the form of CSV files.  The simplest way
to actuate a change in the input files is to simply select an entire file that has the 
desired values, and copy that file into the requisite input location.  This is limited
to categorical inputs, which are inputs that can be represented as discrete categorical
values.  For example, you may have two different population projections, one that 
represents scenario "A" where a particular brownfield area is cleaned up
and developed, and another that represents scenario "B" where the brownfield
is left as is.  Under this policy lever, it doesn't make sense to have an intermediate
value ("we'll just clean up part of the toxic waste, and let only few people move in").

This is the "categorical drop-in" method.  The VisionEval model will either use the
inputs file "A" or the inputs file "B", but not a mix of the two.  This is expressed
in the code by the `categorical_drop_in` method, which is a method of the `FilesCoreModel`.

``` python
def _manipulate_by_categorical_drop_in(self, params, cat_param, ve_scenario_dir):
    """
    Copy in the relevant input files.

    Args:
        params (dict):
            The parameters for this experiment, including both
            exogenous uncertainties and policy levers.
    """
    scenario_dir = params[cat_param]
    for i in os.scandir(scenario_input(ve_scenario_dir,scenario_dir)):
        if i.is_file():
            shutil.copyfile(
                scenario_input(ve_scenario_dir,scenario_dir,i.name),
                join_norm(self.resolved_model_path, 'inputs', i.name)
            )
```

This method is in turn called from individual `setup` sub-methods, which will 
define the specific input parameters that are categorical drop-ins.  For example,
the `setup_population` method will define the specific input parameters that are
categorical drop-ins for the population inputs.

``` python
def _manipulate_carsvcavail(self, params):
    return self._manipulate_by_categorical_drop_in(
        params, # (1)
        'CARSVCAVAILSCEN', # (2)
        self.scenario_input_dirs.get('CARSVCAVAILSCEN') # (3)
    )
```

1.  The `params` dictionary is passed through to the 
    `_manipulate_by_categorical_drop_in` method.
2.  The second argument to the `_manipulate_by_categorical_drop_in` method is the
    name of the parameter in the `params` dictionary that is the categorical drop-in,
    in this case the `CARSVCAVAILSCEN` parameter.
3.  The third argument to the `_manipulate_by_categorical_drop_in` method is the
    directory where the categorical input files for the categorical drop-in are 
    stored.

You will find this function mirrored in the EMAT exploratory scope definition, 
where the categorical drop-in is defined as an uncertainty.
    
``` yaml 
CARSVCAVAILSCEN:
    shortname: Car Service Availability
    address: CARSVCAVAILSCEN
    ptype: exogenous uncertainty
    dtype: cat # (1)
    desc: Different levels of car service availability
    default: mid
    values: # (2)
        - low    
        - mid    
        - high   
```

1.  The `dtype` is set to `cat` to indicate that this is a categorical input, 
    which can only take on one of a discrete set of values.
2.  The `values` list defines the discrete set of values that this parameter can 
    take on.  These should be strings, so that we can match against sub-directory
    names in the `Scenario-Inputs` directory of the VisionEval model.

This structure also requires each categorical drop-in to have a corresponding
directory in the `inputs` directory of the VisionEval model, where the input file(s)
for each categorical drop-in are stored. Note that there is a directory matching
each categorical value, and within that directory are the input files that are
to be used when that categorical value is selected.  Generally, the names of the 
input files will be the same across all categorical values, as shown here.

``` tree
Scenario-Inputs/
    OTP/
        ANOTHER_PARAMETER/
        CARSVCAVAILSCEN/
            low/
                marea_carsvc_availability.csv
            mid/
                marea_carsvc_availability.csv
            high/
                marea_carsvc_availability.csv
        OTHER_PARAMETER/
```
