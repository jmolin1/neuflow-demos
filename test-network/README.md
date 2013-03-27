# Testing your custom network on neuFlow

This script provides simple and easy interface to test your custom network on neuFlow hardware.
Either you can design a network by modifying this script or point out an existing network.

## Dependencies

To run this script, you will need to install Torch and the following dependencies:

``` sh
nnx
neuflow
image
camera
```

## Testing

Use the command to test the network:

``` sh
$ torch test-custom-network.lua                       # run a network defined in the script
$ torch test-custom-network.lua --network=filename    # run an existing network
```

It feeds an input sequence from camera to the network. 
Images will be displayed on the window if the communication channel between computer and neuFlow is built successfully.
