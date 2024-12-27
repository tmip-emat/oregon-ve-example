# Running Experiments

!!! abstract "tl;dr"

    When you run a VisionEval model, it takes a bunch of files as input, it does
    some stuff, and then it gives you a bunch of files as output.
    To run an experiment from EMAT, we need to set up the input files to reflect
    the values of policy levers and exogenous uncertainties for that experiment,
    run VisionEval model to get the outputs, then extract whatever performance
    measures we want from those outputs and feed them back to EMAT.


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

class VEModel(FilesCoreModel): # (1)!
	"""
	A class for using Vision Eval as a files core model.
	"""
	...
```

1.  The `VEModel` class is a subclass of `FilesCoreModel`, which defines the specific
    steps needed to prepare the input files, run the model, and extract the results.

You can see some examples of the `FilesCoreModel` interface [here](https://github.com/tmip-emat/tmip-emat-ve/blob/1f62205652d26389d1767a2cdb0ba1fabe057dea/emat_verspm.py#L36)
and [here](https://github.com/tmip-emat/ve-integration/blob/ff9e83fdc5adf4414dd9454dd980d6f32464bf09/emat_ve_wrapper.py#L40).

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
	def setup(self, params: dict): # (1)!
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
based on the scope:

- [Categorical Drop-In](#categorical-drop-in)
- [Mixture of Data Tables](#mixture-of-data-tables)
- [Scaling Data Tables](#scaling-data-tables)
- [Additive Data Tables](#additive-data-tables)
- [Template Injection](#template-injection)
- [Direct Injection](#direct-injection)
- [Custom Methods](#custom-methods)

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

An advantage of this method is that it is simple to implement, and it places no
limits on the format of the input files.  There is no need to have a specific format
or a matching number of rows or columns in the input files.  In the population projection
example considered above, the input files for the two scenarios could have different 
numbers of rows as one of the two scenarios could imply a different zonal structure
within the region.

In this example repository, this approach is called the "categorical drop-in" method.  
The VisionEval model will either use the inputs file "A" or the inputs file "B", 
but not a mix of the two.  This is expressed in the code by the `categorical_drop_in` 
method, which is a method of the `FilesCoreModel`.

``` python
def _manipulate_by_categorical_drop_in(
    self, 
    params: dict, # (1)!
    cat_param: str, # (2)!
    ve_scenario_dir: os.PathLike, # (3)!
):
    scenario_dir = params[cat_param]
    for i in os.scandir(scenario_input(ve_scenario_dir,scenario_dir)): # (4)!
        if i.is_file():
            shutil.copyfile(
                scenario_input(ve_scenario_dir,scenario_dir, i.name),
                join_norm(self.resolved_model_path, 'inputs', i.name)
            )
```

1.  The `params` dictionary is passed through to the `_manipulate_by_categorical_drop_in`
    method.  This dictionary includes the values of all the policy levers and exogenous 
    uncertainties for the experiment.
2.  The `cat_param` argument is the name of the parameter in the `params` dictionary that
    is the categorical drop-in.
3.  The `ve_scenario_dir` argument is the directory where the categorical input files for
    the categorical drop-in are stored.
4.  The `_manipulate_by_categorical_drop_in` method will scan the appropriate directory  
    where the categorical input files are stored, and copy the input files for the selected 
    categorical value into the requisite input location for the VisionEval model.

This method is in turn called from individual `setup` sub-methods, which will 
define the specific input parameters that are categorical drop-ins.  For example,
the `_manipulate_carsvcavail` method can define the specific input parameters that are
categorical drop-ins for car service availability inputs.

``` python
def _manipulate_carsvcavail(self, params):
    return self._manipulate_by_categorical_drop_in(
        params, # (1)!
        'CARSVCAVAILSCEN', # (2)!
        self.scenario_input_dirs.get('CARSVCAVAILSCEN') # (3)!
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
    dtype: cat # (1)!
    desc: Different levels of car service availability
    default: mid # (2)!
    values: # (3)!
        - low    
        - mid    
        - high   
```

1.  The `dtype` is set to `cat` to indicate that this is a categorical input, 
    which can only take on one of a discrete set of values.
2.  The `default` value is set to `mid`, which will be the selected value for 
    this parameter if no other value is specified.
3.  The `values` list defines the discrete set of values that this parameter can 
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

### Mixture of Data Tables

In contrast to the categorical drop-in method, the "mixture of data tables" method
allows for creating "intermediate" input files that are a mix of different input files.
The approach is suitable for continuous inputs, which are inputs that can take on a
range of values.  For example, you may have a land use density projection that has
upper and lower bounds, and you want to explore the effects of different levels of
density between those limits. 

An advantage of this method is that it allows for a more fine-grained exploration of
the input space, and it can be used for continuous inputs.  However, it does require
that the input files have a specific format (a CSV table containing primarily numeric 
data), and that the number of rows and columns in the input files match across both
the input files, which are labeled as "1" and "2" in this example.

Instead of copying an entire file, the mixture of data tables method will read in
both input files, and then linearly interpolate between the two input files based 
on the value of the policy lever or exogenous uncertainty.  This is expressed in the
code by the `_manipulate_by_mixture` method, which is a method of the `FilesCoreModel`.

``` python
def _manipulate_by_mixture(
    self, 
    params, # (1)!
    weight_param, # (2)!
    ve_scenario_dir, # (3)!
    no_mix_cols=('Year', 'Geo',), # (4)!
    float_dtypes=False, # (5)!
):
    weight_2 = params[weight_param]
    weight_1 = 1.0-weight_2

    # Gather list of all files in directory "1", and confirm they
    # are also in directory "2"
    filenames = []
    for i in os.scandir(scenario_input(ve_scenario_dir,'1')):
        if i.is_file():
            filenames.append(i.name)
            f2 = scenario_input(ve_scenario_dir,'2', i.name)
            if not os.path.exists(f2):
                raise FileNotFoundError(f2)

    for filename in filenames:
        df1 = pd.read_csv(scenario_input(ve_scenario_dir,'1',filename))
        isna_ = (df1.isnull().values).any()
        df1.fillna(0, inplace=True) # (6)!
        df2 = pd.read_csv(scenario_input(ve_scenario_dir,'2',filename))
        df2.fillna(0, inplace=True)

        float_mix_cols = list(df1.select_dtypes('float').columns)
        if float_dtypes:
            float_mix_cols = float_mix_cols+list(
                df1.select_dtypes('int').columns
            )
        for j in no_mix_cols:
            if j in float_mix_cols:
                float_mix_cols.remove(j)

        if float_mix_cols:
            df1_float = df1[float_mix_cols]
            df2_float = df2[float_mix_cols]
            df1[float_mix_cols] = df1_float * weight_1 + df2_float * weight_2

        int_mix_cols = list(df1.select_dtypes('int').columns)
        if float_dtypes:
            int_mix_cols = list()
        for j in no_mix_cols:
            if j in int_mix_cols:
                int_mix_cols.remove(j)

        if int_mix_cols:
            df1_int = df1[int_mix_cols]
            df2_int = df2[int_mix_cols]
            df_int_mix = df1_int * weight_1 + df2_int * weight_2
            df1[int_mix_cols] = np.round(df_int_mix).astype(int) # (7)!

        out_filename = join_norm(
            self.resolved_model_path, 'inputs', filename
        )
        if isna_:
            df1.replace(0, np.nan, inplace=True)
        df1.to_csv(out_filename, index=False, float_format="%.5f", na_rep='NA')
```

1.  The `params` dictionary is passed through to the `_manipulate_by_mixture` method.
2.  The `weight_param` argument is the name of the parameter in the `params` dictionary
    that is the weight for the mixture of data tables.
3.  The `ve_scenario_dir` argument is the directory where the input files for the 
    mixture of data tables are stored. There should be two subdirectories, "1" and "2".
4.  The `no_mix_cols` argument is a list of column names that should not be mixed. This 
    is useful for columns that are not numerical, such as year or geography, which should
    not be mixed (or for which there is no reasonable linear interpolation). These columns
    will be copied from the input file in directory "1" to the output file.
5.  The `float_dtypes` argument is a boolean that indicates whether integer columns
    should be treated as float columns for the purposes of mixing. Setting this
    to `True` will treat integer columns as float columns, and will mix them as such,
    which can be problematic if VisionEval is expecting integers.
6.  The `isna_` variable is set to `True` if there are any `NaN` values in the input
    file.  If there are, these will be replaced with zeros for the purposes of mixing,
    and then replaced with `NaN` in the output file, as linear interpolation of `NaN`
    values is not possible.
7.  The `df_int_mix` variable is the linear interpolation of the integer columns, and
    is optionally rounded to the nearest integer. This is done to ensure that the output file
    has integer values, which is important if VisionEval is expecting integers.

This method is in turn called from individual `setup` sub-methods, which will
define the specific input parameters that are mixtures of data tables.  For example,
the [`_manipulate_landuse`](https://github.com/tmip-emat/ve-integration/blob/7b41df9c94b2671ccef0fd214593f3629f8337e3/emat_ve_wrapper.py#L534-L544) 
method can define the specific input parameters that are
mixtures of data tables for land use density inputs.

``` python
def _manipulate_ludensity(self, params):
    return self._manipulate_by_mixture(
        params, # (1)!
        'LUDENSITYMIX', # (2)!
        self.scenario_input_dirs.get('LUDENSITYMIX'), # (3)!
    )
```

1.  The `params` dictionary is passed through to the 
    `__manipulate_by_mixture` method.
2.  The second argument to the `_manipulate_by_mixture` method is the
    name of the parameter in the `params` dictionary that is controlling the,
    mixture, in this case the `LUDENSITYMIX` parameter.
3.  The third argument to the `__manipulate_by_mixture` method is the
    directory where the categorical input files for the mixture bounds are 
    stored.

You will find this function mirrored in the EMAT exploratory 
[scope definition](https://github.com/tmip-emat/ve-integration/blob/7b41df9c94b2671ccef0fd214593f3629f8337e3/EMAT-VE-Configs/odot-otp-scope.yml#L11-L23),
where the mixture of data tables is defined as an exogenous uncertainty.

``` yaml
    LUDENSITYMIX:
        shortname: Urban Mix Prop
        address: LUDENSITYMIX
        ptype: exogenous uncertainty
        dtype: float # (1)!
        desc: Urban proportion for each marea by year
        default: 0
        min: 0 # (2)!
        max: 1 # (3)!
```

1.  The `dtype` is set to `float` to indicate that this is a continuous input, 
    which can take on a range of values.
2.  The `min` value for mixtures is always set to `0`, which represents the lower bound for this 
    parameter, and will set the weight of the "1" input file to `1.0` and the
    weight of the "2" input file to `0.0`.
3.  The `max` value for mixtures is always set to `1`, which represents the upper bound for this 
    parameter, and will set the weight of the "1" input file to `0.0` and the
    weight of the "2" input file to `1.0`.
    
This structure also requires each mixture to have a corresponding
directory in the `inputs` directory of the VisionEval model, where the input file(s)
for each categorical drop-in are stored. Note that there are exactly two sub-directories
in this parameters directory, and they are named "1" and "2", and within those two directories
are the input files that are to be mixed together. The names of the 
input file(s) *must* be the same across all both sub-directories, as shown here, 
and they must be in the same format (a CSV table containing primarily numeric data).

``` tree
Scenario-Inputs/
    OTP/
        ANOTHER_PARAMETER/
        LUDENSITYMIX/
            1/
                marea_mix_targets.csv
            2/
                marea_mix_targets.csv
        OTHER_PARAMETER/
```

### Scaling Data Tables

Forthcoming: documentation of the `_manipulate_by_scale` method, which allows for
scaling of single input files.

### Additive Data Tables

Forthcoming: documentation of the `_manipulate_by_delta` method, which allows 
mixtures based on additive deltas instead of linear interpolation.

### Template Injection

Forthcoming: documentation of the template injection method, which writes 
parameter values directly into the input files based on a template.

### Direct Injection

Forthcoming: documentation of the direct injection method, which writes
parameter values directly into the input files, overwriting existing values.

### Custom Methods

Forthcoming: documentation of how to define custom methods for preparing input files.


## Running an Experiment

Once the input files have been prepared, the VisionEval model can be run.  The
`FilesCoreModel` interface defines the `run` method as the place to run the VisionEval
model. The `run` method needs to be overloaded in a subclass of `FilesCoreModel` to define
the specific steps needed to run the VisionEval model.

In this example, the main thing we do in the `run` method is to set the `path` environment
variable to include the path to the R executable, and then run s small script 
that opens the VisionEval model and runs it with the desired inputs.

``` python
class VEModel(FilesCoreModel):
    ...
    def run(self): # (1)!
        os.environ['path'] = (
            join_norm(self.config['r_executable'])+';'+os.environ['path']   
        )
        cmd = 'Rscript'
    
        # write a small script that opens the model and runs it
        with open(join_norm(self.local_directory, "vemodel_runner.R"), "wt") as script:
            script.write(f"""
            thismodel <- openModel("{r_join_norm(self.local_directory, self.modelname)}")
            thismodel$run("reset")
            """)
    
        self.last_run_result = subprocess.run( # (2)!
            [cmd, 'vemodel_runner.R'],
            cwd=self.local_directory,
            capture_output=True,
        )
            
``` 

1.  The `run` method accepts no arguments. All the information needed to run the
    experiment is stored in files written during the `setup` method.
2.  The subprocess.run command runs a command line tool. The
    name of the command line tool, plus all the command line arguments
    for the tool, are given as a list of strings, not one string.
    The `cwd` argument sets the current working directory from which the
    command line tool is launched.  Setting `capture_output` to True
    will capture both stdout and stderr from the command line tool, and
    make these available in the result object to facilitate debugging.
