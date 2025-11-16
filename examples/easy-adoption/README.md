# Using Perforated AI for KAN Conversion

A major feature of the Perforated AI library is the ability to quickly convert any PyTorch network to a dendritic architecture through the use of initialize_pai and the PAIModule class.  Because the functions involved in this conversion are generalizable the library can also be used to do experimentation with any custom module types.  This can be an enormously beneficial tool for anyone looking to get into experimental new modules.  Often learning about new systems can capture the curiosity of a data scientist, but implementation can feel daunting.  Major overhauls of a codebase can take weeks of effort, and when working with other libraries where one didn't even write the original architecture themselves it can feel insurmountable just to begin.  The Perforated AI library completely eliminates these issues.

To display how simple it can be to work with frontier ideas, we have created this script showing how to convert the PyTorch Official MNIST example into a KAN Network with one line of code.
    
## Preparation
    
While the conversion only takes one line of code, you will have to create a wrapper function and some configuration steps first.  In addition, don't forget to install the library:

    pip install perforatedai

### 1) Imports

Of course, importing the required libraries will be the first step.

    import efficient_kan
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA

### 2) Define The Conversion    

To do conversion with you need to create a custom class, or a wrapper function like this, which takes in your old class and returns the new class.  In this example we want the input arg to be the original nn.Linear and the returned class to be a KANLinear.

    def kan_from_linear(linear_module):
        
        # Adjustable settings here
        grid_size=5
        spline_order=3
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        enable_standalone_scale_spline=True
        base_activation=torch.nn.SiLU
        grid_eps=0.02
        grid_range=[-1, 1]
        
        return efficient_kan.KANLinear(
            in_features=linear_module.in_features,
            out_features=linear_module.out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            enable_standalone_scale_spline=enable_standalone_scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )

### 3) Perforated AI Settings
        
Because this example is just doing KAN conversion and not doing dendrites some settings need to be changed to not also do dendrites.  The system also needs to be told to actually do that Linear to KANLinear conversion.  Details in the comments below of what each step does.
        
    def setupKANonfiguration():
        # Instruct PAI to replace all nn.Linears with a call to kan_from_linear with that nn.Linear as the argument
        GPA.pc.append_modules_to_replace([nn.Linear])
        GPA.pc.append_replacement_modules([kan_from_linear])
        # This line tells the system to not convert the new KANLinears to be dendrite modules (or the other conv2d modules)
        GPA.pc.append_modules_to_track([efficient_kan.KANLinear, nn.Conv2d])
        # These lines tell the system we are not doing dendritic optimization at all and shut off warnings related to dendrites
        GPA.pc.set_max_dendrites(0)
        GPA.pc.set_perforated_backpropagation(False)
        GPA.pc.set_weight_decay_accepted(True)
        GPA.pc.set_checked_skipped_modules(True)
        GPA.pc.set_unwrapped_modules_confirmed(True)
    
## The One Line

With that preparation in place, the initialize_pai function will now do everything that is required to convert all your Linear layers to KANLinears.  Just add the following after your initial model is defined:

    model = UPA.initialize_pai(model).to(device)

## Results

With the default MNIST code final results are as follows:

    Test set: Average loss: 0.0261, Accuracy: 9922/10000 (99%)

With this one line of code new results are:

    Test set: Average loss: 0.0233, Accuracy: 9925/10000 (99%)
    
With the starting test loss already at 0.0261, a reduction of 0.003 is an 11% improvement in loss.  Not bad for one line of code.
    

### ViT Eample

To show an additional use case we also include cifar_ViT_perforated_KAN.py.  The the results show the KAN network overfits on this model and dataset, but this is included as well simply to show the use case for a more complex model.
